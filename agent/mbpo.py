import os
import numpy as np
import torch

from components.dynamics import Dynamics, format_samples_for_training

class MBPOAgent:
    """ model-based policy optimization """
    def __init__(
        self, 
        policy,
        dynamics_model,
        static_fns,
        batch_size
    ):
        self.policy = policy
        self.dynamics_model = dynamics_model
        self.static_fns = static_fns
        self.dynamics = Dynamics(self.dynamics_model, self.static_fns)
        self.batch_size = batch_size

    def train(self):
        self.policy.train()

    def eval(self):
        self.policy.eval()

    def act(self, obs, deterministic=False, return_logprob=False):
        return self.policy.act(obs, deterministic, return_logprob)
    
    def value(self, obs, action):
        return self.policy.value(obs, action)

    def rollout_transitions(self, init_transitions, rollout_len):
        """ rollout """
        transitions = {"s": [], "a": [], "r": [], "s_": [], "done": []}

        obs = init_transitions["s"]
        for _ in range(rollout_len):
            # imaginary step
            actions = self.policy.act(obs)
            next_obs, rewards, dones, _ = self.dynamics.step(obs, actions)

            # store
            transitions["s"].append(obs)
            transitions["a"].append(actions)
            transitions["r"].append(rewards)
            transitions["s_"].append(next_obs)
            transitions["done"].append(dones)

            # to next step
            nonterm_mask = (~dones).flatten()
            if nonterm_mask.sum() == 0: break
            obs = next_obs[nonterm_mask]

        transitions = {key: np.concatenate(transitions[key], axis=0) for key in transitions.keys()}
        return transitions

    def learn_dynamics(self, transitions):
        """ learn dynamics model """
        inputs, targets = format_samples_for_training(transitions)
        loss = self.dynamics.train(
            inputs,
            targets,
            batch_size=self.batch_size
        )
        return loss["holdout_loss"].item()

    def learn_policy(self, transitions):
        """ learn policy """
        info = self.policy.learn(**transitions)
        return info

    def save_model(self, filepath):
        """ save model """
        self.policy.save_model(filepath)
        dynamics_dir = filepath.split(".pth")[0]
        if not os.path.exists(dynamics_dir):
            os.makedirs(dynamics_dir)
        self.dynamics.save(dynamics_dir)

    def load_model(self, filepath):
        """ load model """
        self.policy.load_model(filepath)
