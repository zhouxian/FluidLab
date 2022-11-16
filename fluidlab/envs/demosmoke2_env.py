import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.engine.taichi_env import TaichiEnv
from fluidlab.engine.losses import *
from fluidlab.engine.simulators import SmokeFieldDemo2

class DemoSmoke2Env(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 250
        self.horizon_action        = 250
        self.target_file           = None
        self._n_obs_ptcls_per_body = 200
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.1, 0.1])

        # create a taichi env
        self.taichi_env = TaichiEnv(
            dim=3,
            quality=2.0,
            particle_density=4e6,
            max_substeps_local=10,
            gravity=(0.0, -20.0, 0.0),
            horizon=self.horizon,
            ckpt_dest='cpu',
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
    #         file='bowl.obj',
    #         pos=(0.5, 0.2, 0.5),
    #         euler=(0.0, 0.0, 0.0),
    #         scale=(0.85, 0.85, 0.85),
    #         material=BOWL,
    #         has_dynamics=True,
    #     )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='nowhere',
            n_particles=100000,
            material=WATER,
        )

    def setup_smoke_field(self):
        self.taichi_env.smoke_field = SmokeFieldDemo2(
            dim=self.taichi_env.dim,
            ckpt_dest=self.taichi_env.ckpt_dest,
            res          = 128,
            dt           = 0.03,
            solver_iters = 100,
            decay        = 0.97,
            q_dim        = 3,
        )


    def setup_renderer(self):
        ggui = True
        ggui = False
        if ggui:
            self.taichi_env.setup_renderer(
                camera_pos=(1.5, 2.5, 4.0),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        else:
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                camera_pos=(-0.5, 1.5, 4.0),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=25,
                scene_radius=3.0,
                light_pos=(0.5, 5.0, 1.5),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=100,
                cam_rotate_v=7.5e-4,
                _smoothing=0.0
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
        init_p = np.array([0.8, 0.7, 0.5])
        comp_actions_p[0] = init_p
        comp_actions = np.vstack([comp_actions_v, comp_actions_p])
        return ActionsPolicy(comp_actions)

    def trainable_policy(self, optim_cfg, init_range):
        return CirculationPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 5, 6, 7])

    def cmaes_policy(self, init_range):
        return CirculationCMAESPolicy(init_range, self.agent.action_dim, self.horizon_action, self.action_range, fix_dim=[0, 1, 2, 3, 5, 6, 7])
