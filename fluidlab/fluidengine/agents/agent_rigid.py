import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentRigid(Agent):
    # Agent with one Rigid
    
    def __init__(self, **kwargs):
        super(AgentRigid, self).__init__(**kwargs)

    def build(self, sim):
        super(AgentRigid, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Rigid)
        self.rigid = self.effectors[0]

    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        return self.rigid.collide(f, pos_world, mat_v, dt)
