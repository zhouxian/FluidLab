import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.fluidengine.effectors import *

@ti.data_oriented
class AgentIceCream(Agent):
    # Agent for icecream env, with one static Injector and one controllable Rigid
    def __init__(self, inject_till=0, **kwargs):
        super(AgentIceCream, self).__init__(**kwargs)
        self.inject_till = inject_till

    def build(self, sim):
        super(AgentIceCream, self).build(sim)

        assert self.n_effectors == 2
        assert isinstance(self.effectors[0], Injector)
        self.injector = self.effectors[0]
        assert isinstance(self.effectors[1], Rigid)
        self.rigid = self.effectors[1]

        self.injector.set_act_range(self.sim.particles_ng.used.to_numpy()[0])

    def act(self, f, f_global):
        if f_global < self.inject_till:
            self.act_kernel(f, f_global)
            self.check_act_range(f)

    def act_grad(self, f, f_global):
        if f_global < self.inject_till:
            self.act_kernel.grad(f, f_global)

    @ti.kernel
    def act_kernel(self, f: ti.i32, f_global: ti.i32):
        self.injector.act(f, f_global, self.sim.particles_ng.used, self.sim.particles.x, self.sim.particles.v)

    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        ret_v = mat_v
        if pos_world[1] > 0.25:
            ret_v = self.rigid.collide(f, pos_world, mat_v, dt)
        return ret_v

    def check_act_range(self, f):
        assert self.injector.act_id[f+1] <= self.injector.act_range.shape[0], 'too many particles added'
            
    @property
    def action_dim(self):
        return self.rigid.action_dim

    @property
    def state_dim(self):
        return self.rigid.state_dim

    def set_action(self, s, s_global, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.rigid.action_dim
        self.rigid.set_action(s, s_global, n_substeps, action)

    def set_action_grad(self, s, s_global, n_substeps, action):
        action = np.asarray(action).reshape(-1).clip(-1, 1)
        assert len(action) == self.rigid.action_dim
        self.rigid.set_action_grad(s, s_global, n_substeps, action)

    def apply_action_p(self, action_p):
        action_p = np.asarray(action_p).reshape(-1).clip(0.05, 0.95)
        self.rigid.apply_action_p(action_p)
            
    def apply_action_p_grad(self, action_p):
        action_p = np.asarray(action_p).reshape(-1).clip(0.05, 0.95)
        self.rigid.apply_action_p_grad(action_p)

    def get_grad(self, n):
        grad = self.rigid.get_action_grad(0, n)
        return grad

    def move(self, f):
        for i in range(self.n_effectors):
            self.effectors[i].move(f)

    def move_grad(self, f):
        self.rigid.move_grad(f)
