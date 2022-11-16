import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from .transporting_env import TransportingEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.engine.taichi_env import TaichiEnv

class StabilizingEnv(TransportingEnv):
    def __init__(self, version, loss=True, seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon = 2000
        self.horizon_action = 2000
        self.target_file = get_tgt_path('Stabilizing-v0.pkl')
        self._n_observed_particles = 200
        self.loss = loss

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, -10.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_stabilizing.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=200000,
            material=WATER,
        )
        self.taichi_env.add_body(
            type='cube',
            lower=(0.6, 0.475, 0.475),
            size=(0.05, 0.05, 0.05),
            euler=(45.0, 45.0, 45.0),
            color=(1.0, 0.5, 0.5, 1.0),
            filling='natural',
            material=RIGID_HEAVY,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.15, 0.45),
            upper=(0.95, 0.95, 0.55),
            restitution=0.0,
            lock_dims=[2],
        )

    def setup_renderer(self):
        self.taichi_env.setup_renderer(
            res=(960, 960),
            camera_pos=(0.5, 0.5, 30),
            camera_lookat=(0.5, 0.5, 0.5),
            fov=2,
            # camera_pos=(-0.15, 2.82, 2.5),
            # camera_lookat=(0.5, 0.5, 0.5),
            # fov=30,
            particle_radius=0.006,
            lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                    {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        )

