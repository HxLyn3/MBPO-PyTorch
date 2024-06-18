import os
import random
import argparse
import setproctitle

import torch
import numpy as np

from runner import RUNNER

def get_args():
    parser = argparse.ArgumentParser(description="MBRL")

    # environment settings
    parser.add_argument("--env", type=str, default="mujoco")
    # mujoco choices: ["InvertedPendulum-v2", "Hopper-v3", "Swimmer-v3", "Walker2d-v3", "HalfCheetah-v3", "AntTruncatedObs-v3", "HumanoidTruncatedObs-v3"]
    parser.add_argument("--env-name", type=str, default="HumanoidTruncatedObs-v3")

    # algorithm parameters
    parser.add_argument("--algo", type=str, default="mbpo")
    parser.add_argument("--ac-hidden-dims", type=list, default=[256, 256])              # dimensions of actor/critic hidden layers
    parser.add_argument("--actor-lr", type=float, default=3e-4)                         # learning rate of actor
    parser.add_argument("--critic-lr", type=float, default=3e-4)                        # learning rate of critic
    parser.add_argument("--gamma", type=float, default=0.99)                            # discount factor
    parser.add_argument("--tau", type=float, default=0.005)                             # update rate of target network
    # (for sac)
    parser.add_argument("--alpha", type=float, default=0.2)                             # weight of entropy
    parser.add_argument("--auto-alpha", type=bool, default=True)                        # auto alpha adjustment
    parser.add_argument("--alpha-lr", type=float, default=3e-4)                         # learning rate of alpha
    # actor update frequency
    parser.add_argument("--actor-freq", type=int, default=1)
    # target entropy
    # InvertedPendulum: -0.05, Hopper, Swimmer: -1, Walker2d, HalfCheetah: -3, Ant: -4, Humanoid: -8
    parser.add_argument("--target-entropy", type=int, default=-8)                       # target entropy

    # replay-buffer parameters
    parser.add_argument("--buffer-size", type=int, default=int(1e6))

    # dynamics-model parameters
    # dynamics-hidden-dim   Humanoid: 400, others: 200
    parser.add_argument("--dynamics-hidden-dim", type=int, default=400)
    parser.add_argument("--dynamics-hidden-layers", type=int, default=4)
    parser.add_argument("--n-ensembles", type=int, default=7)
    parser.add_argument("--n-elites", type=int, default=5)
    parser.add_argument("--rollout-batch-size", type=int, default=int(1e5))
    # rollout schedule (for MPPVE)
    # InvertedPendulum: [0, 1e3, 1, 5], Swimmer, HalfCheetah, Walker2d: [2e4, 1e5, 1, 1]
    # Hopper: [2e4, 5e4, 1, 4], Ant: [2e4, 1.5e5, 1, 20], Humanoid: [2e4, 3e5, 1, 15]
    parser.add_argument("--rollout-schedule", type=list, default=[2e4, 3e5, 1, 15])
    # model-update-interval     Humanoid: 1000, others: 250
    parser.add_argument("--model-update-interval", type=int, default=1000)
    # model-retain-steps        Humanoid: 5000, others: 1000
    parser.add_argument("--model-retain-steps", type=int, default=5000)
    parser.add_argument("--real-ratio", type=float, default=0.05)

    # running parameters
    parser.add_argument("--train", action="store_true", default=True)
    parser.add_argument("--test", action="store_true", default=False)
    parser.add_argument("--n-steps", type=int, default=int(3e5))
    parser.add_argument("--start-learning", type=int, default=int(5e3))
    parser.add_argument("--update-interval", type=int, default=1)
    # UTD
    parser.add_argument("--updates-per-step", type=int, default=20)                     # only use for model-based algos
    parser.add_argument("--batch-size", type=int, default=256)                          # mini-batch size
    parser.add_argument("--eval-interval", type=int, default=int(1e3))
    parser.add_argument("--eval-n-episodes", type=int, default=10)
    parser.add_argument("--test-n-episodes", type=int, default=int(1e3))
    parser.add_argument("--save-interval", type=int, default=int(1e4))
    parser.add_argument("--render", action="store_true", default=False)
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--seed", type=int, default=0)

    args = parser.parse_args()
    return args

def main():
    args = get_args()
    setproctitle.setproctitle("{} {}".format(args.algo.upper(), args.env_name))

    seed_start, seed_end = 0, 3
    if not args.train and not args.test:
        raise ValueError("Argument 'train' and 'test' can't be both False")

    """ main function """
    for seed in range(seed_start, seed_end):
        args.seed = seed
        random.seed(args.seed)
        np.random.seed(args.seed)
        os.environ["PYTHONHASHSEED"] = str(args.seed)

        # set seed of torch
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        args.stage = "train" if args.train else "test"
        runner = RUNNER["{}-{}".format(args.algo, args.stage)](args)
        runner.run()

if __name__ == "__main__":
    main()
