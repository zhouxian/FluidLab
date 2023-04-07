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

class LatteArtStirEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 500
        self.horizon_action        = 500
        self.target_file           = get_tgt_path('LatteArtStir-v0.pkl')
        self._n_obs_ptcls_per_body = 1000
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.01, 0.01])

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
        agent_cfg.merge_from_file(get_cfg_path('agent_latteartstir.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='cup.obj',
            pos=(0.63, 0.42, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.2, 1.2, 1.2),
            material=CUP,
            has_dynamics=False,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.56, 0.5),
            height=0.02,
            radius=0.42,
            material=MILK_VIS,
        )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.475, 0.5),
            height=0.15,
            radius=0.42,
            material=COFFEE_VIS,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cylinder',
            xz_radius=0.42,
            xz_center=(0.5, 0.5),
            y_range=(0.4, 0.95),
        )

    def setup_renderer(self):
        # self.taichi_env.setup_renderer(
        #     camera_pos=(-0.15, 2.82, 2.5),
        #     camera_lookat=(0.5, 0.5, 0.5),
        #     fov=30,
        #     lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
        #             {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        # )

        self.taichi_env.setup_renderer(
            type='GL',
            render_particle=True,
            camera_pos=(-0.15, 2.82, 2.5),
            camera_lookat=(0.5, 0.5, 0.5),
            fov=30,
            light_pos=(3.5, 15.0, 0.55),
            light_lookat=(0.5, 0.5, 0.49),
            light_fov=20,
        )
        
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                     
    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=LatteArtStirLoss,
            type=self.loss_type,
            target_file=self.target_file,
            weights={'chamfer': 1.0}
        )

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']

        if self.loss is not None:
            tgt_particles = self.loss.tgt_particles_x_f32
        else:
            tgt_particles = None
            
        return self.taichi_env.render(mode)
        
    def demo_policy(self):
        init_p = np.array([0.5, 0.73, 0.5])
        comp_actions_p = init_p
        return MousePolicy_vxz(init_p)

    def trainable_policy(self, optim_cfg, init_range):
        return LatteArtStirPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[1])
