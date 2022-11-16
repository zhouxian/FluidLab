import os
import gym
import numpy as np
from gym.spaces import Box
from fluidlab.configs.macros import *
from fluidlab.engine.taichi_env import TaichiEnv
import fluidlab.utils.misc as misc_utils

class FluidEnv(gym.Env):
    '''
    Base env class.
    '''    
    def __init__(self, version, loss=True, loss_type='diff', seed=None):
        if seed is not None:
            self.seed(seed)

        self.horizon               = 500
        self.horizon_action        = 500
        self.target_file           = None
        self._n_obs_ptcls_per_body = 200
        self.loss                  = loss
        self.loss_type             = loss_type
        self.action_range          = np.array([-1.0, 1.0])

        # create a taichi env
        self.taichi_env = TaichiEnv()
        self.build_env()
        self.gym_misc()

    def seed(self, seed):
        super(FluidEnv, self).seed(seed)
        misc_utils.set_random_seed(seed)

    def build_env(self):
        self.setup_agent()
        self.setup_statics()
        self.setup_bodies()
        self.setup_smoke_field()
        self.setup_boundary()
        if not misc_utils.is_on_server():
            self.setup_renderer()
        if self.loss:
            self.setup_loss()
            
        self.taichi_env.build()
        self._init_state = self.taichi_env.get_state()
        
        print(f'===>  {type(self).__name__} built successfully.')

    def setup_agent(self):
        pass

    def setup_statics(self):
        # add static mesh-based objects in the scene
        pass

    def setup_bodies(self):
        # add fluid/object bodies
        self.taichi_env.add_body(
            type='cube',
            lower=(0.2, 0.2, 0.2),
            upper=(0.4, 0.4, 0.4),
            material=WATER,
        )
        self.taichi_env.add_body(
            type='ball',
            center=(0.6, 0.3, 0.6),
            radius=0.1,
            material=WATER,
        )

    def setup_smoke_field(self):
        pass

    def setup_boundary(self):
        pass

    def setup_renderer(self):
        self.taichi_env.setup_renderer()

    def setup_loss(self):
        pass

    def gym_misc(self):
        if self.loss_type == 'default':
            self.horizon = self.horizon_action
        obs = self.reset()
        self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), obs.shape, dtype=DTYPE_NP)
        if self.taichi_env.agent is not None:
            self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]), (self.taichi_env.agent.action_dim,), dtype=DTYPE_NP)
        else:
            self.action_space = None

    def reset(self):
        self.taichi_env.set_state(**self._init_state)
        return self._get_obs()

    def _get_obs(self):
        state = self.taichi_env.get_state_RL()
        obs   = []

        if 'x' in state:
            for body_id in range(self.taichi_env.particles['bodies']['n']):
                body_n_particles  = self.taichi_env.particles['bodies']['n_particles'][body_id]
                body_particle_ids = self.taichi_env.particles['bodies']['particle_ids'][body_id]

                step_size = max(1, body_n_particles // self._n_obs_ptcls_per_body)
                body_x    = state['x'][body_particle_ids][::step_size]
                body_v    = state['v'][body_particle_ids][::step_size]
                body_used = state['used'][body_particle_ids][::step_size]

                obs.append(body_x.flatten())
                obs.append(body_v.flatten())
                obs.append(body_used.flatten())

        if 'agent' in state:
            obs += state['agent']

        if 'smoke_field' in state:
            obs.append(state['smoke_field']['v'][::10, 60:68, ::10].flatten())
            obs.append(state['smoke_field']['q'][::10, 60:68, ::10].flatten())

        obs = np.concatenate(obs)
        return obs

    def _get_reward(self):
        loss_info = self.taichi_env.get_step_loss()
        return loss_info['reward']

    def step(self, action):
        action = action.clip(self.action_range[0], self.action_range[1])
        
        self.taichi_env.step(action)

        obs    = self._get_obs()
        reward = self._get_reward()

        assert self.t <= self.horizon
        if self.t == self.horizon:
            done = True
        else:
            done = False

        if np.isnan(reward):
            reward = -1000
            done = True

        info = dict()
        return obs, reward, done, info

    def render(self, mode='human'):
        assert mode in ['human', 'rgb_array']
        return self.taichi_env.render(mode)

    @property
    def t(self):
        return self.taichi_env.t
