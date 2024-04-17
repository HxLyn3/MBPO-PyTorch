import os
import numpy as np
from tqdm import tqdm

from env import ENV
from agent import AGENT

class SACTester:
    """ test soft actor-critic """
    def __init__(self, args):
        # init env
        self.env = ENV[args.env](args.env_name)
        self.env.seed(args.seed)
        self.env.action_space.seed(args.seed)

        args.obs_shape = self.env.observation_space.shape
        args.action_space = self.env.action_space
        args.action_dim = np.prod(args.action_space.shape)

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
        self.model_dir = "./result/{}/{}/{}/model".format(args.env, args.env_name, args.algo)
        self.agent.load_model(os.path.join(self.model_dir, "model_seed-{}.pth".format(args.seed)))
        self.agent.eval()
        
        # other parameters
        self.test_n_episodes = args.test_n_episodes
        self.render = args.render
        self.device = args.device
        self.seed = args.seed
        self.args = args

    def run(self):
        """ test {args.algo} on {args.env} for {args.test_n_episodes} episodes"""
        pbar = tqdm(range(self.test_n_episodes), desc= "Testing {} on {}.{} (seed: {})".format(
            self.args.algo.upper(), self.args.env.title(), self.args.env_name, self.seed))

        episode_rewards = []
        for _ in pbar:
            # init
            done = False
            episode_rewards.append(0)
            obs = self.env.reset()
            while not done:
                action = self.agent.act(obs, deterministic=True)
                next_obs, reward, done, _ = self.env.step(action)
                episode_rewards[-1] += reward
                obs = next_obs

                # render
                if self.render: self.env.render()

            pbar.set_postfix(test_reward=episode_rewards[-1])

        print("[#] test reward | mean: {:.4f}, min: {:.4f}, max: {:.4f}".format(
            np.mean(episode_rewards), np.min(episode_rewards), np.max(episode_rewards)))
