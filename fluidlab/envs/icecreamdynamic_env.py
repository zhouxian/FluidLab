import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import IceCreamDynamicLoss

class IceCreamDynamicEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI'):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 900
        self.horizon_action        = 900
        self.target_file           = get_tgt_path('IceCreamDynamic-v0.pkl')
        self._n_obs_ptcls_per_body = 2000
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.005, 0.005])
        self.renderer_type         = renderer_type

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=40,
            gravity=(0.0, -10.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_icecreamdynamic.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='icecream_dispenser.obj',
            pos=(-0.32, 0.96, 0.24),
            euler=(0.0, 0.0, 0.0),
            scale=(2.5, 2.5, 2.5),
            material=DISPENSER,
            has_dynamics=False,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=100000,
            material=ICECREAM,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                res=(960, 960),
                camera_pos=(3.96, 1.72, 3.99),
                camera_lookat=(3.24, 1.53, 3.32),
                # camera_pos=(4.62, 2.37, 0.28),
                # camera_lookat=(3.81, 1.95, 0.28),
                # camera_pos=(0.74, 5.65, 0.42),
                # camera_lookat=(0.6, 1.65, 0.42),
                fov=30,
                lights=[{'pos': (0.5, 3.5, 3.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 0.5, 3.5), 'color': (0.35, 0.35, 0.35)},
                        {'pos': (-5.0, 1.5, 0.5), 'color': (0.35, 0.35, 0.35)},
                        {'pos': (5.0, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(3.96, 1.72, 3.99),
                camera_lookat=(3.24, 1.53, 3.32),
                fov=30,
                light_pos=(0.5, 10.0, 5.55),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=60,
                floor_height=-1.0,
                camera_far=20,
            )
        else:
            raise NotImplementedError

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=IceCreamDynamicLoss,
            type=self.loss_type,
            target_file=self.target_file,
            weights={'chamfer': 1.0}
        )

    def demo_policy(self, user_input=False):
        if user_input:
            raise NotImplementedError

        comp_actions_p = np.zeros((1, self.agent.action_dim))
        comp_actions_v = np.zeros((900, self.agent.action_dim))
        init_center = np.array([0.5, 0.3, 0.5])
        y_range = 0.0
        rad_v_lin = 0.0042
        init_radius = 0.15
        theta = np.pi
        init_p = init_center + np.array([init_radius * np.cos(theta), 0, init_radius * np.sin(theta)])
        current_p = np.array(init_p)
        radius_v = 4e-5

        horizon_0 = 168
        horizon_2 = 20
        horizon_0_ = horizon_0
        horizon_1 = 900 - horizon_0 - horizon_2
        horizon_1_ = horizon_0 + horizon_1
        horizon_2_ = horizon_1_ + horizon_2
        for i in range(horizon_0):
            target_x = init_p[0]
            target_y = init_p[1]
            target_z = init_p[2]
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]

        for i in range(horizon_0_, horizon_1_):
            t = i - horizon_0_
            r =  init_radius - radius_v * t
            rad_v = rad_v_lin / r
            theta += rad_v

            target_x = init_center[0] + r * np.cos(theta)
            target_y = init_center[1] - y_range * t / horizon_1
            target_z = init_center[2] + r * np.sin(theta)
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]

        # cooling down
        for i in range(horizon_1_, horizon_2_):
            t = i - horizon_1_
            rad_v = rad_v_lin / r * (1 - t / horizon_2)
            theta += rad_v
            target_x = init_center[0] + r * np.cos(theta)
            target_y = init_center[1] - y_range
            target_z = init_center[2] + r * np.sin(theta)
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]

        comp_actions_p[0] = init_p
        comp_actions_v = comp_actions_v[:self.horizon_action]
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)

    def trainable_policy(self, optim_cfg, init_range):
        # return IceCreamDynamicPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[1])
        return IceCreamDynamicPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=None)
    