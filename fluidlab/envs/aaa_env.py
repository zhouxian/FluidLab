import os
import gym
import numpy as np
from gym.spaces import Box
from fluidlab.configs.macros import *

class AAAEnv(gym.Env):
    def __init__(self, version, loss=True, seed=None):

        self.horizon = 400
        self.action_range          = np.array([-0.005, 0.005])

        obs = self.reset()
        self.observation_space = Box(DTYPE_NP(-np.inf), DTYPE_NP(np.inf), obs.shape, dtype=DTYPE_NP)
        self.action_space = Box(DTYPE_NP(self.action_range[0]), DTYPE_NP(self.action_range[1]), (3,), dtype=DTYPE_NP)

    def reset(self):
        self.pos = np.array([0.5, 0.5, 0.5])
        self.t = 0
        return self._get_obs()

    def setup_loss(self):
        self.taichi_env.setup_loss(RLTestLoss)

    def step(self, action):
        self.t += 1
        self.action = action
        self.pos += self.action

        obs    = self._get_obs()
        reward = self._get_reward()

        assert self.t <= self.horizon
        if self.t == self.horizon:
            done = True
        else:
            done = False

        info = dict()
        return obs, reward, done, info

    def _get_obs(self):
        obs = self.pos
        return obs

    def _get_reward(self):
        # diff = np.linalg.norm(self.action - np.array([0.5,0,0]))
        # r = 10 * (-diff)
        diff = np.linalg.norm(self.pos - np.array([0.8, 0.5, 0.5]))
        r = 10.0 * (0.5 - diff)
        # diff = np.abs(self.action[0] - 0.5)
        # r = 10 * (1.0-diff)

        return r
