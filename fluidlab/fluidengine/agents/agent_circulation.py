import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentCirculation(Agent):
    # Agent for air circulation env
    def __init__(self, **kwargs):
        super(AgentCirculation, self).__init__(**kwargs)

    def build(self, sim):
        super(AgentCirculation, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], AirCon)
        self.aircon = self.effectors[0]

    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        return mat_v
