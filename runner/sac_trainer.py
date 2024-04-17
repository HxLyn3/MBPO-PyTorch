import os
import json
import numpy as np
from tqdm import tqdm

from agent import AGENT
from utils import BUFFER
from .base_trainer import BASETrainer

class SACTrainer(BASETrainer):
    """ train soft actor-critic """
    def __init__(self, args):
        super(SACTrainer, self).__init__(args)

        # init agent
        self.agent = AGENT["sac"](
            obs_shape=args.obs_shape,
            hidden_dims=args.ac_hidden_dims,
            action_dim=args.action_dim,
            action_space=args.action_space,
            actor_lr=args.actor_lr,
            critic_lr=args.critic_lr,
            tau=args.tau,
            gamma=args.gamma,
            alpha=args.alpha,
            auto_alpha=args.auto_alpha,
            alpha_lr=args.alpha_lr,
            target_entropy=args.target_entropy,
            device=args.device
        )
        self.agent.train()

        # init replay buffer
        self.memory = BUFFER["vanilla"](args.buffer_size, args.obs_shape, args.action_dim)

    def run(self):
        """ train {args.algo} on {args.env} for {args.n_steps} steps"""

        # init
        records = {"step": [], "loss": {"actor": [], "critic1": [], "critic2": []}, 
                  "alpha": [], "reward_mean": [], "reward_std": [], "reward_min": [], "reward_max": [],
                  "value_bias_mean": [], "value_bias_std": []}
        obs = self._warm_up()

        actor_loss, critic1_loss, critic2_loss, alpha, eval_reward = [None]*5
        pbar = tqdm(range(self.n_steps), desc="Training {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        for it in pbar:
            # step in env
            action = self.agent.act(obs)
            next_obs, reward, done, info = self.env.step(action)
            timeout = info.get("TimeLimit.truncated", False)
            self.memory.store(obs, action, reward, next_obs, done, timeout)

            obs = next_obs
            if done: obs = self.env.reset()

            # render
            if self.render: self.env.render()

            # update policy
            if it%self.update_interval == 0:
                transitions = self.memory.sample(self.batch_size)
                learning_info = self.agent.learn(**transitions)
                actor_loss = learning_info["loss"]["actor"]
                critic1_loss = learning_info["loss"]["critic1"]
                critic2_loss = learning_info["loss"]["critic2"]
                alpha = learning_info["alpha"]

            # evaluate policy
            if it%self.eval_interval == 0:
                episode_rewards = self._eval_policy()
                records["step"].append(it)
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
                actor_loss=actor_loss, 
                critic1_loss=critic1_loss, 
                critic2_loss=critic2_loss, 
                eval_reward=eval_reward
            )

            # save
            if it%self.save_interval == 0: self._save(records)

        self._save(records)
