import yaml
import taichi as ti
import numpy as np
from fluidlab.fluidengine.effectors import *
from fluidlab.configs.macros import *
from .agent_injector import AgentInjector
from fluidlab.fluidengine.boundaries import create_boundary

@ti.data_oriented
class AgentJetBot(AgentInjector):
    # Agent with one Injector and a collector

    def __init__(self, collector_boundary, **kwargs):
        super(AgentJetBot, self).__init__(**kwargs)
        self.collector_boundary = create_boundary(**collector_boundary)
        self.nowhere = ti.Vector(NOWHERE, dt=DTYPE_TI)

    def act(self, f, f_global):
        self.injector_act_kernel(f, f_global)
        self.collector_act_kernel(f, f_global)

    def act_grad(self, f, f_global):
        self.collector_act_kernel.grad(f, f_global)
        self.injector_act_kernel.grad(f, f_global)

    @ti.kernel
    def injector_act_kernel(self, f: ti.i32, f_global: ti.i32):
        self.injector.act(f, f_global, self.sim.particles_ng.used, self.sim.particles.x, self.sim.particles.v)

    @ti.kernel
    def collector_act_kernel(self, f: ti.i32, f_global: ti.i32):
        # collect out-of-boundary particles
        for p in range(self.sim.n_particles):
            if self.sim.particles_ng[f, p].used and self.sim.particles_i[p].mat == WATER:
                if self.collector_boundary.is_out(self.sim.particles[f, p].x):
                    self.sim.particles_ng[f+1, p].used = 0
                    self.sim.particles[f+1, p].x = self.nowhere
                    
                    # set current step used to false to avoid later options in simulator's substep()
                    self.sim.particles_ng[f, p].used = 0
