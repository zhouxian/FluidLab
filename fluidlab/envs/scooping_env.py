import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.engine.taichi_env import TaichiEnv

class ScoopingEnv(FluidEnv):
    def __init__(self, version, loss=True, seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 420
        self.horizon_action        = 300
        self.target_file           = None
        self._n_obs_ptcls_per_body = 200
        self.loss                  = loss
        self.action_range          = np.array([-1.0, 1.0])

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=50,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_scooping.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='tank.obj',
            pos=(0.5, 0.4, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.85, 0.92, 0.92),
            material=TANK,
            has_dynamics=False,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.12, 0.25, 0.18),
            upper=(0.88, 0.45, 0.82),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.5, 0.6, 0.55),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -90.0, 0.0),
            color=(1.0, 1.0, 0.3, 1.0),
            filling='grid',
            material=ELASTIC,
        )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.2, 0.5, 0.55),
            height=0.08,
            radius=0.02,
            euler=(0.0, 0.0, -45.0),
            color=(1.0, 0.5, 0.5, 1.0),
            filling='natural',
            material=ELASTIC,
        )
        self.taichi_env.add_body(
            type='ball',
            center=(0.3, 0.5, 0.4),
            radius=0.03,
            color=(0.4, 1.0, 0.4, 1.0),
            filling='natural',
            material=ELASTIC,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.12, 0.25, 0.18),
            upper=(0.88, 0.95, 0.82),
        )

    def setup_renderer(self):
        gl_render = False
        # gl_render = True
        if gl_render:
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                res=(960, 960),
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(0.5, 5.0, 0.55),
                light_lookat=(0.5, 0.5, 0.49),
            )
        else:
            self.taichi_env.setup_renderer(
                type='GGUI',
                # render_particle=True,
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
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
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)
