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

class GatheringOEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 3600
        self.horizon_action        = 3600
        self.target_file           = None
        self._n_obs_ptcls_per_body = 500
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.003, 0.003])

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
        agent_cfg.merge_from_file(get_cfg_path('agent_gatheringO.yaml'))
        self.taichi_env.setup_agent(agent_cfg)
        self.agent = self.taichi_env.agent

    def setup_statics(self):
        self.taichi_env.add_static(
            file='tank_O.obj',
            pos=(0.5, 0.4, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(1.0, 0.92, 0.92),
            material=TANK,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',
            lower=(0.05, 0.3, 0.17),
            upper=(0.95, 0.45, 0.83),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.88, 0.5, 0.45),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -75.0, 0.0),
            color=(1.0, 1.0, 0.3, 1.0),
            filling='grid',
            material=RIGID,
        )
        self.taichi_env.add_body(
            type='mesh',
            file='duck.obj',
            pos=(0.25, 0.5, 0.78),
            scale=(0.10, 0.10, 0.10),
            euler=(0, -95.0, 0.0),
            color=(1.0, 0.5, 0.5, 1.0),
            filling='grid',
            material=RIGID,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.06, 0.3, 0.18),
            upper=(0.94, 0.95, 0.82),
        )

    def setup_renderer(self):
        gl_render = False
        gl_render = True
        if gl_render:
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(0.15, 2.82, 1.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                light_pos=(5.0, 5.0, 0.5),
                light_lookat=(0.5, 0.5, 0.5),
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
            loss_cls=GatheringOLoss,
            type=self.loss_type,
            matching_mat=RIGID,
            weights={'dist': 1.0}
        )

    def demo_policy(self):
        comp_actions_p = np.zeros((1, self.agent.action_dim))
        comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
        init_p = np.array([0.5, 0.45, 0.5])
        comp_actions_v[:] = np.array([0.003, 0.0, 0.0])
        comp_actions_p[0] = init_p
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)

    def trainable_policy(self, optim_cfg, init_range):
        return GatheringOPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[1, 2])

    def cmaes_policy(self, init_range):
        return GatheringOCMAESPolicy(init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[1, 2])
