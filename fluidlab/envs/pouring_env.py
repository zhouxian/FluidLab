import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import *

class PouringEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 1000
        self.horizon_action        = 1000
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.02, 0.02])

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_pouring.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        pass

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.6, 0.53, 0.5),
            height=0.2,
            radius=0.18,
            material=MILK,
        )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.6, 0.73, 0.5),
            height=0.2,
            radius=0.18,
            material=WATER,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        self.taichi_env.setup_renderer(
            camera_pos=(0.5, 0.6, 3.5),
            camera_lookat=(0.5, 0.6, 0.5),
            fov=26,
            lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                    {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        )

    def setup_loss(self):
        self.taichi_env.setup_loss(
            target_file=self.target_file,
            weights={'EMD': 1.0, 'chamfer': 0.0}
        )

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=PouringLoss,
            type=self.loss_type,
            weights={'dist': 1.0, 'attraction': 1.0}
        )
        
    def demo_policy(self):
        init_p = np.array([0.6, 0.7, 0.5])
        comp_actions_p = init_p
        return KeyboardPolicy_wz(init_p, v_ang=0.015)

    def trainable_policy(self, optim_cfg, init_range):
        return PouringPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 4])
