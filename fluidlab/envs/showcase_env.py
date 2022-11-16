import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.engine.taichi_env import TaichiEnv

class ShowcaseEnv(FluidEnv):
    '''
    Multi-material env as a showcase.
    '''
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
        agent_cfg.merge_from_file(get_cfg_path('agent_showcase.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    # def setup_statics(self):
    #     self.taichi_env.add_static(
    #         file='pillar.obj',
    #         pos=(0.5, 0.35, 0.5),
    #         euler=(0.0, 0.0, 0.0),
    #         scale=(0.5, 0.7, 0.5),
    #         # scale=(0.25, 0.7, 0.25),
    #         material=PILLAR,
    #         has_dynamics=True,
    #     )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=100000,
            material=WATER,
        )

    def setup_smoke_field(self):
        self.taichi_env.setup_smoke_field(
            res          = 128,
            dt           = 0.03,
            solver_iters = 250,
            decay        = 0.96
        )

    # def setup_boundary(self):
    #     self.taichi_env.setup_boundary(
    #         type='cube',
    #         lower=(0.12, 0.25, 0.18),
    #         upper=(0.88, 0.95, 0.82),
    #     )

    def setup_renderer(self):
        self.taichi_env.setup_renderer(
            res=(1280, 1280),
            camera_pos=(1.5, 2.5, 4.0),
            camera_lookat=(0.5, 0.5, 0.5),
            # camera_pos=(1.32, 4.16, -0.26),
            # camera_lookat=(1.10, 3.20, -0.08),
            fov=25,
            lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                    {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        )

    def setup_loss(self):
        self.taichi_env.setup_loss(
            target_file=self.target_file,
            weights={'EMD': 1.0, 'chamfer': 0.0}
        )

    def demo_policy(self):
        comp_actions_p = np.zeros((1, self.agent.action_dim))
        comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
        init_p = np.array([0.8, 0.7, 0.5])
        y_range_0 = -0.3
        current_p = np.array(init_p)
        horizon_0 = int(self.horizon_action / 3)
        horizon_1 = int(self.horizon_action / 3)
        horizon_2 = self.horizon_action - horizon_0 - horizon_1
        step = 0
        for i in range(horizon_0):
            target_i = i + 1
            target_x = init_p[0]
            target_y = init_p[1] + target_i/horizon_0*y_range_0
            target_z = init_p[2]
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]
        step += horizon_0

        x_range = -0.25
        for i in range(step, step + horizon_1):
            target_i = i + 1 - step
            target_x = init_p[0] + target_i/horizon_1*x_range
            target_y = init_p[1] + y_range_0
            target_z = init_p[2]
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]
        step += horizon_1

        y_range_1 = -y_range_0
        for i in range(step, step + horizon_2):
            target_i = i + 1 - step
            target_x = init_p[0] + x_range
            target_y = init_p[1] + y_range_0 + target_i/horizon_2*y_range_1
            target_z = init_p[2]
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]

        comp_actions_p[0] = init_p
        comp_actions_v = np.zeros_like(comp_actions_v)
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)
