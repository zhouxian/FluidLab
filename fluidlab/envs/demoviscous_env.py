import os
import gym
import colorsys
import numpy as np
from .fluid_env import FluidEnv
from yacs.config import CfgNode
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.optimizer.policies import *
from fluidlab.engine.taichi_env import TaichiEnv

class DemoViscousEnv(FluidEnv):
    def __init__(self, version, loss=True, loss_type='diff', seed=None):

        if seed is not None:
            self.seed(seed)

        self.horizon               = 700
        self.horizon_action        = 700
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
        )
        self.build_env()
        self.gym_misc()

    def setup_statics(self):
        self.taichi_env.add_static(
            file='bowl.obj',
            pos=(0.5, 0.2, 0.5),
            euler=(0.0, 0.0, 0.0),
            scale=(0.85, 0.85, 0.85),
            material=BOWL,
            has_dynamics=True,
        )

    def setup_bodies(self):
        self.taichi_env.add_body(
            type='cube',    
            lower=(0.3, 0.3, 0.3),  
            upper=(0.7, 0.7, 0.4),  
            filling='random',  
            material=VISCOUS_DEMO,
        )
        for i in range(15):
            type = np.random.choice(['mesh', 'cube', 'ball', 'cylinder'])
            material = RIGID
            euler = np.random.uniform(size=3) * 360 - 180
            color = random_color(h_low=0.0, h_high=1.0, s=0.6, v=0.90)
            range_low = np.array([0.2, 0.3, 0.4])
            range_high = np.array([0.8, 0.4, 0.8])

            if type == 'mesh':
                file = 'duck.obj'
                pos = np.random.uniform(range_low, range_high)
                scale = np.random.uniform([0.08, 0.08, 0.08], [0.12, 0.12, 0.12])

                self.taichi_env.add_body(
                    type='mesh',
                    file=file,
                    pos=pos,
                    scale=scale,
                    euler=euler,
                    color=color,
                    filling='grid',
                    material=material,
                )
            elif type == 'cube':
                lower = np.random.uniform(range_low, range_high)
                upper = lower + np.random.uniform(0.03, 0.05)
                self.taichi_env.add_body(
                    type='cube',
                    lower=lower,
                    upper=upper,
                    euler=euler,
                    color=color,
                    filling='natural',
                    material=material,
                )
            elif type == 'ball':
                center = np.random.uniform(range_low, range_high)
                radius = np.random.uniform(0.02, 0.03)
                self.taichi_env.add_body(
                    type='ball',
                    center=center,
                    radius=radius,
                    euler=euler,
                    color=color,
                    filling='natural',
                    material=material,
                )
            elif type == 'cylinder':
                center = np.random.uniform(range_low, range_high)
                radius = np.random.uniform(0.02, 0.025)
                height = np.random.uniform(0.06, 0.10)
                self.taichi_env.add_body(
                    type='cylinder',
                    center=center,
                    height=height,
                    radius=radius,
                    euler=euler,
                    color=color,
                    filling='natural',
                    material=material,
                )

    def setup_boundary(self):
        self.taichi_env.setup_boundary(
            type='cube',
            lower=(0.05, 0.115, 0.05),
            upper=(0.95, 0.95, 0.95),
        )

    def setup_renderer(self):
        gl_render = False
        gl_render = True
        if gl_render:
            self.taichi_env.setup_renderer(
                type='GL',
                # render_particle=True,
                particle_radius=0.0075,
                camera_pos=(-0.15, 1.92, 2.5),
                camera_lookat=(0.5, 0.35, 0.5),
                fov=32,
                scene_radius=2.0,
                light_pos=(0.5, 5.0, 1.5),
                light_lookat=(0.5, 0.5, 0.49),
                light_fov=40,
                cam_rotate_v=5e-4,
                _smoothing=0.25,
            )
        else:
            self.taichi_env.setup_renderer(
                type='GGUI',
                # render_particle=True,
                particle_radius=0.005,
                camera_pos=(-0.15, 1.82, 2.5),
                camera_lookat=(0.5, 0.5, 0.5),
                fov=40,
                lights=[{'pos': (0.5, 1.5, 0.5), 'color': (0.5, 0.5, 0.5)},
                        {'pos': (0.5, 1.5, 1.5), 'color': (0.5, 0.5, 0.5)}],
            )

    def demo_policy(self):
        comp_actions = [None] * (self.horizon_action + 1)
        return ActionsPolicy(comp_actions)
