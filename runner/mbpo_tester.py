import os
import json
import numpy as np
from tqdm import tqdm

from agent import AGENT
from .sac_tester import SACTester
from components.static_fns import STATICFUNC
from components.dynamics_model import EnsembleDynamicsModel

class MBPOTester(SACTester):
    """ test model-based policy optimization """
    def __init__(self, args):
        super(MBPOTester, self).__init__(args)

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
