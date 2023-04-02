import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import IceCreamStaticLoss

class IceCreamStaticEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI'):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 550
        self.horizon_action        = 550
        self.target_file           = get_tgt_path('IceCreamStatic-v0.pkl')
        self._n_obs_ptcls_per_body = 2000
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.005, 0.005])
        self.renderer_type         = renderer_type

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, -5.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_icecreamstatic.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='cone.obj',
            pos=(0.5, 0.1, 0.5),
            euler=(-90.0, 0.0, 30.0),
            scale=(0.435, 0.435, 0.435),
            material=CONE,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=100000,
            material=ICECREAM1,
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
                camera_pos=(4.48, 2.41, -0.84),
                camera_lookat=(3.64, 1.95, -0.56),
                fov=15,
                lights=[{'pos': (0.5, 3.5, 3.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 0.5, 3.5), 'color': (0.35, 0.35, 0.35)},
                        {'pos': (-5.0, 1.5, 0.5), 'color': (0.35, 0.35, 0.35)},
                        {'pos': (5.0, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
            self.taichi_env.setup_renderer(
                type='GL',
                camera_pos=(4.48, 2.41, -0.84),
                camera_lookat=(3.64, 1.95, -0.56),
                fov=18,
                light_pos=(0.5, 10.0, 5.55),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=60,
                floor_height=-0.5,
                camera_far=20,
            )
        else:
            raise NotImplementedError


    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=IceCreamStaticLoss,
            target_file=self.target_file,
            type=self.loss_type,
            weights={'chamfer': 1.0}
        )


    def demo_policy(self, user_input=False):
        if user_input:
            raise NotImplementedError

        comp_actions_p = np.zeros((1, self.agent.action_dim))
        comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
        init_center = np.array([0.5, 0.36, 0.5])
        y_range = 0.26
        init_radius = 0.12
        final_radius = 0.01
        init_p = init_center + np.array([init_radius, 0, 0])
        current_p = np.array(init_p)

        init_rad_v = 0.01
        final_rad_v = init_rad_v * init_radius / final_radius
        theta = 0.0
        horizon_0 = 0
        horizon_0_ = horizon_0
        horizon_1 = 700
        horizon_1_ = horizon_0 + horizon_1
        for i in range(self.horizon_action):
            t = i - horizon_0_
            rad_v = (final_rad_v - init_rad_v) * t / horizon_1 + init_rad_v
            theta += rad_v
            r =  t/horizon_1 * (final_radius - init_radius) + init_radius
            target_x = init_center[0] + r * np.cos(theta)
            target_y = init_center[1] + y_range * t / horizon_1
            target_z = init_center[2] + r * np.sin(theta)
            target_p = np.array([target_x, target_y, target_z])

            comp_actions_v[i] = target_p - current_p
            current_p += comp_actions_v[i]


        comp_actions_p[0] = init_p
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)
        
    def trainable_policy(self, optim_cfg, init_range):
        return IceCreamStaticPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=None)
