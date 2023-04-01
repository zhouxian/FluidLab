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

class TransportingEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 1000
        self.horizon_action        = 1000
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.01, 0.01])

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=20,
            gravity=(0.0, 0.0, 0.0),
            horizon=self.horizon,
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_transporting.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        pass

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=200000,
            material=WATER,
        )
        self.taichi_env.add_body(
            type='cube',
            lower=(0.275, 0.475, 0.475),
            size=(0.05, 0.05, 0.05),
            euler=(45.0, 45.0, 45.0),
            color=(1.0, 0.5, 0.5, 1.0),
            filling='natural',
            material=RIGID_HEAVY,
        )
        # self.taichi_env.add_body(
        #     type='cylinder',
        #     center=(0.2, 0.5, 0.5),
        #     height=0.08,
        #     radius=0.02,
        #     euler=(0.0, 0.0, -45.0),
        #     color=(0.4, 1.0, 0.4, 1.0),
        #     filling='natural',
        #     material=RIGID,
        # )
        # self.taichi_env.add_body(
        #     type='ball',
        #     center=(0.7, 0.7, 0.5),
        #     radius=0.025,
        #     color=(1.0, 1.0, 0.3, 1.0),
        #     filling='natural',
        #     material=RIGID,
        # )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.05, 0.45),
            upper=(0.95, 0.95, 0.55),
            restitution=0.0,
            lock_dims=[2],
        )

    def setup_renderer(self):
        # self.taichi_env.setup_renderer(
        #     res=(960, 960),
        #     camera_pos=(0.5, 0.5, 30),
        #     camera_lookat=(0.5, 0.5, 0.5),
        #     fov=2,
        #     # camera_pos=(-0.15, 2.82, 2.5),
        #     # camera_lookat=(0.5, 0.5, 0.5),
        #     # fov=30,
        #     particle_radius=0.006,
        #     lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
        #             {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        # )

        self.taichi_env.setup_renderer(
            type='GL',
            # render_particle=True,
            # res=(960, 960),
            camera_pos=(0.5, 0.5, 3),
            camera_lookat=(0.5, 0.5, 0.5),
            camera_far=50.0,
            floor_height=-10.0,
            scene_radius=4.0,
            fov=22,
            light_pos=(0.5, 5.0, 5.0),
            light_lookat=(0.5, 0.5, 0.49),
            light_fov=120,
            _smoothing=0.0
        )

    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=TransportingLoss,
            type=self.loss_type,
            weights={'dist': 1.0}
        )


    def demo_policy(self):
        init_p = np.array([0.5, 0.2, 0.5, 0.0, 0.0, 0.0])
        comp_actions_p = init_p
        return KeyboardPolicy_vxy_wz(init_p, v_ang=0.003)

    def trainable_policy(self, optim_cfg, init_range):
        return TransportingPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[1, 2, 3, 4])
