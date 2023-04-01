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

class CirculationEnv(FluidEnv):
    '''
    Indoor air circulation.
    '''
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 1000
        self.horizon_action        = 1000
        self.target_file           = None
        self._n_obs_ptcls_per_body = 200
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.1, 0.1])

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            particle_density=1e6,
            max_substeps_local=100,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
            ckpt_dest='cpu',
        )
        self.build_env()
        self.gym_misc()

    def setup_agent(self):
        agent_cfg = CfgNode(new_allowed=True)
        agent_cfg.merge_from_file(get_cfg_path('agent_circulation.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='room.obj',
            # file_vis='room_vis.obj',
            pos=(0.5, 0.5, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.4, 3.0, 1.4),
            material=PILLAR,
            sdf_res=128,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=10,
            material=WATER,
        )

    def setup_smoke_field(self):
        self.taichi_env.setup_smoke_field(
            res          = 128,
            dt           = 0.03,
            solver_iters = 50,
            decay        = 0.99,
            q_dim        = 1,
        )

    # def setup_boundary(self):
    #     self.taichi_env.setup_boundary(
    #         type='cube',
    #         lower=(0.12, 0.25, 0.18),
    #         upper=(0.88, 0.95, 0.82),
    #     )

    def setup_renderer(self):
        # self.taichi_env.setup_renderer(
        #     res=(1280, 1280),
        #     camera_pos=(0.5, 12.0, 0.501),
        #     camera_lookat=(0.5, 0.5, 0.5),
        #     # camera_pos=(0.5, 0.5, 2.5),
        #     # camera_lookat=(0.5, 0.5, 0.5),
        #     fov=8,
        #     lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
        #             {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
        # )

        self.taichi_env.setup_renderer(
            type='GL',
            # render_particle=True,
            camera_pos=(0.5, 12.0, 0.501),
            camera_lookat=(0.5, 0.5, 0.5),
            camera_far=15.0,
            scene_radius=15.0,
            fov=5,
            light_pos=(0.5, 5.0, 1.5),
            light_lookat=(0.5, 0.5, 0.49),
        )
        
    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=CirculationLoss,
            type=self.loss_type,
            weights={'temp': 1.0}
        )

    def demo_policy(self):
        comp_actions_p = np.zeros((1, self.agent.action_dim))
        comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
        init_p = np.array([0.55, 0.5, 0.27, 0.0, 0.0, 0.0, 0.0, 0.0])
        comp_actions_p[0] = init_p
        comp_actions_v[:] = np.array([0.0, 0.0, 0.0, 0.0, 0.1, 0.0, 0.02, 0.04])
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)

    def trainable_policy(self, optim_cfg, init_range):
        return CirculationPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 5, 6, 7])

    def cmaes_policy(self, init_range):
        return CirculationCMAESPolicy(init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 5, 6, 7])
