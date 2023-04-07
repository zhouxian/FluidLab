import os
import gym
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.fluidengine.taichi_env import TaichiEnv
from fluidlab.fluidengine.losses import LatteArtLoss

class LatteArtEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None, renderer_type='GGUI'):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 330
        self.horizon_action        = 250
        self.target_file           = get_tgt_path('LatteArt-v0.pkl')
        self._n_obs_ptcls_per_body = 1000
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-0.05, 0.05])
        self.renderer_type         = renderer_type

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
        agent_cfg.merge_from_file(get_cfg_path('agent_latteart.yaml'))
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
            type='nowhere',
            n_particles=60000,
            material=MILK,
        )
        self.taichi_env.add_body(
            type='cylinder',
            center=(0.5, 0.55, 0.5),
            height=0.1,
            radius=0.42,
            material=COFFEE,
        )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cylinder',
            xz_radius=0.42,
            xz_center=(0.5, 0.5),
            y_range=(0.5, 0.95),
        )

    def setup_renderer(self):
        if self.renderer_type == 'GGUI':
            self.taichi_env.setup_renderer(
                res=(960, 960),
                camera_pos=(-0.15, 2.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=30,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )
        elif self.renderer_type == 'GL':
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
        else:
            raise NotImplementedError
        
    def setup_loss(self):
        self.taichi_env.setup_loss(
            loss_cls=LatteArtLoss,
            type=self.loss_type,
            target_file=self.target_file,
            weights={'chamfer': 1.0}
        )

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
            
        return self.taichi_env.render(mode)
        
    def demo_policy(self, user_input=False):
        if user_input:
            init_p = np.array([0.5, 0.73, 0.5])
            comp_actions_p = init_p
            return MousePolicy_vxz(init_p)
        else:
            comp_actions_p = np.zeros((1, self.agent.action_dim))
            comp_actions_v = np.zeros((self.horizon_action, self.agent.action_dim))
            init_p = np.array([0.15, 0.65, 0.5])
            x_range = 0.7
            current_p = np.array(init_p)
            amp_range = np.array([0.15, 0.25])
            for i in range(self.horizon_action):
                target_i = i + 1
                target_x = init_p[0] + target_i/self.horizon_action*x_range
                target_y = init_p[1]
                cycles = 3
                target_rad = target_i/self.horizon_action*(np.pi*2)*cycles
                target_amp = amp_range[1] - np.abs((target_i*2/self.horizon_action) - 1) * (amp_range[1] - amp_range[0])
                target_z = np.sin(target_rad)*target_amp+0.5
                target_p = np.array([target_x, target_y, target_z])

                comp_actions_v[i] = target_p - current_p
                current_p += comp_actions_v[i]

            comp_actions_p[0] = init_p
            comp_actions = np.vstack([comp_actions_v, comp_actions_p])
            return ActionsPolicy(comp_actions)

    def trainable_policy(self, optim_cfg, init_range):
        return LatteArtPolicy(optim_cfg, init_range, self.agent.action_dim, self.horizon_action, self.action_range)
