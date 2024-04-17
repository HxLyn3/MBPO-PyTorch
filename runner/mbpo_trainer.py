import numpy as np
from tqdm import tqdm

from agent import AGENT
from utils import BUFFER
from .sac_trainer import SACTrainer
from components.static_fns import STATICFUNC
from components.dynamics_model import EnsembleDynamicsModel

class MBPOTrainer(SACTrainer):
    """ train model-based policy optimization """
    def __init__(self, args):
        super(MBPOTrainer, self).__init__(args)

        # init dynamics model
        self.dynamics_model = EnsembleDynamicsModel(
            obs_dim=np.prod(args.obs_shape),
            action_dim=args.action_dim,
            hidden_features=args.dynamics_hidden_dim,
            ensemble_size=args.n_ensembles,
            ensemble_select_size=args.n_elites,
            device=args.device
        )

        # init mbpo-agent
        task = args.env_name.split('-')[0]
        static_fns = STATICFUNC[task]
        self.agent = AGENT["mbpo"](
            policy=self.agent,
            dynamics_model=self.dynamics_model,
            static_fns=static_fns,
            batch_size=args.batch_size
        )

        # create memory to store imaginary transitions
        model_rollout_size = args.rollout_batch_size*args.rollout_schedule[2]
        model_buffer_size = int(model_rollout_size*args.model_retain_steps/args.model_update_interval)
        self.model_memory = BUFFER["vanilla"](
            buffer_size=model_buffer_size,
            obs_shape=args.obs_shape,
            action_dim=args.action_dim
        )

        # func 4 calculate new rollout length (x->y over steps a->b)
        a, b, x, y = args.rollout_schedule
        self.make_rollout_len = lambda it: int(min(max(x+(it-a)/(b-a)*(y-x), x), y))
        # func 4 calculate new model buffer size
        self.make_model_buffer_size = lambda it: \
            int(args.rollout_batch_size*self.make_rollout_len(it) * \
            args.model_retain_steps/args.model_update_interval)

        # other parameters
        self.model_update_interval = args.model_update_interval
        self.rollout_batch_size = args.rollout_batch_size
        self.real_ratio = args.real_ratio
        self.updates_per_step = args.updates_per_step

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        # init
        records = {"step": [], "loss": {"model": [], "actor": [], "critic1": [], "critic2": []}, 
                  "alpha": [], "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],
                  "value_bias_mean": [], "value_bias_std": []}
        obs = self._warm_up()

        # model_loss, actor_loss, critic1_loss, critic2_loss, eval_reward = [None]*5
        pbar = tqdm(range(self.n_steps), desc="Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            # update (one-step) dynamics model
            if it%self.model_update_interval == 0:
                transitions = self.memory.sample_all()
                model_loss = self.agent.learn_dynamics(transitions)

                # update imaginary memory
                new_model_buffer_size = self.make_model_buffer_size(it)
                if self.model_memory.capacity != new_model_buffer_size:
                    new_buffer = BUFFER["vanilla"](
                        buffer_size=new_model_buffer_size,
                        obs_shape=self.model_memory.obs_shape,
                        action_dim=self.model_memory.action_dim
                    )
                    old_transitions = self.model_memory.sample_all()
                    new_buffer.store_batch(**old_transitions)
                    self.model_memory = new_buffer

                # rollout
                init_transitions = self.memory.sample(self.rollout_batch_size)
                rollout_len = self.make_rollout_len(it)
                fake_transitions = self.agent.rollout_transitions(init_transitions, rollout_len)
                self.model_memory.store_batch(**fake_transitions)

                print(f"rollout length: {rollout_len},",
                      f"model buffer capacity: {new_model_buffer_size},",
                      f"model buffer size: {self.model_memory.size}")

            # step in env
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            timeout = info.get("TimeLimit.truncated", False)
            self.memory.store(obs, action, reward, next_obs, done, timeout)
            # next state
            obs = next_obs
            if done: obs = self.env.reset()
            # render
            if self.render: self.env.render()

            # update policy
            if it%self.update_interval == 0:
                for _ in range(int(self.update_interval*self.updates_per_step)):
                    real_sample_size = int(self.batch_size*self.real_ratio)
                    fake_sample_size = self.batch_size - real_sample_size
                    real_batch = self.memory.sample(batch_size=real_sample_size)
                    fake_batch = self.model_memory.sample(batch_size=fake_sample_size)
                    transitions = {key: np.concatenate(
                        (real_batch[key], fake_batch[key]), axis=0) for key in real_batch.keys()}
                    learning_info = self.agent.learn_policy(transitions)
                    actor_loss = learning_info["loss"]["actor"]
                    critic1_loss = learning_info["loss"]["critic1"]
                    critic2_loss = learning_info["loss"]["critic2"]
                    alpha = learning_info["alpha"]

            # evaluate policy
            if it%self.eval_interval == 0:
                episode_rewards = self._eval_policy()
                records["step"].append(it)
                records["loss"]["model"].append(model_loss)
                records["loss"]["actor"].append(actor_loss)
                records["loss"]["critic1"].append(critic1_loss)
                records["loss"]["critic2"].append(critic2_loss)
                records["alpha"].append(alpha)
                records["reward_mean"].append(np.mean(episode_rewards))
                records["reward_std"].append(np.std(episode_rewards))
                records["reward_min"].append(np.min(episode_rewards))
                records["reward_max"].append(np.max(episode_rewards))
                eval_reward = records["reward_mean"][-1]
                value_bias_info = self._eval_value_estimation()
                records["value_bias_mean"].append(value_bias_info["value_bias_mean"])
                records["value_bias_std"].append(value_bias_info["value_bias_std"])

            pbar.set_postfix(
                alpha=alpha,
                model_loss=model_loss,
                actor_loss=actor_loss, 
                critic1_loss=critic1_loss, 
                critic2_loss=critic2_loss, 
                eval_reward=eval_reward
            )

            # save
            if it%self.save_interval == 0: self._save(records)

        self._save(records)
    
    def _eval_value_estimation(self):
        """ evaluate value estimation"""
        value_bias_mean, value_bias_std = [], []
        for _ in range(self.eval_n_episodes):
            rewards = []
            log_probs = []
            value_preds = []
            obs = self.eval_env.reset()
            done = False
            while not done:
                action, log_prob = self.agent.act(obs, deterministic=False, return_logprob=True)
                value_preds.append(self.agent.value(obs, action)[0])
                obs, reward, done, info = self.eval_env.step(action)
                rewards.append(reward)
                log_probs.append(log_prob.flatten()[0])
            
            timeout = info.get("TimeLimit.truncated", False)
            returns = []
            if timeout:
                action, log_prob = self.agent.act(obs, deterministic=False, return_logprob=True)
                next_value = self.agent.value(obs, action)[0]
                returns.append(next_value)
                log_probs.append(log_prob.flatten()[0])
            else:
                returns.append(0)
                log_probs.append(0)
            for r in reversed(rewards):
                returns.append(r + self.agent.policy._gamma * (returns[-1] - self.agent.policy._alpha.cpu().item()*log_probs[-1]))
                log_probs.pop()
            
            returns = np.array(list(reversed(returns[1:]))).flatten()
            value_preds = np.array(value_preds).flatten()

            value_bias_mean.append((value_preds - returns).mean() / returns.mean())
            value_bias_std.append((value_preds - returns).std() / returns.mean())
        
        return {
            "value_bias_mean": np.mean(value_bias_mean),
            "value_bias_std": np.mean(value_bias_std)
        }
