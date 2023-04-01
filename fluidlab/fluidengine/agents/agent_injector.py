import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentInjector(Agent):
    # Agent with one Injector

    def __init__(self, **kwargs):
        super(AgentInjector, self).__init__(**kwargs)
        
    def build(self, sim):
        super(AgentInjector, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Injector)
        self.injector = self.effectors[0]

        self.injector.set_act_range(self.sim.particles_ng.used.to_numpy()[0])

    def act(self, f, f_global):
        self.act_kernel(f, f_global)
        self.check_act_range(f)

    def act_grad(self, f, f_global):
        self.act_kernel.grad(f, f_global)

    @ti.kernel
    def act_kernel(self, f: ti.i32, f_global: ti.i32):
        self.injector.act(f, f_global, self.sim.particles_ng.used, self.sim.particles.x, self.sim.particles.v)

    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        return mat_v

    def check_act_range(self, f):
        assert self.injector.act_id[f+1] <= self.injector.act_range.shape[0], 'too many particles added'
