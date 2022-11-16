import yaml
import taichi as ti
import numpy as np
from .agent import Agent
from fluidlab.engine.effectors import *

@ti.data_oriented
class AgentShowcase(Agent):
    # Agent for showcase env
    def __init__(self, inject_till=0, **kwargs):
        super(AgentShowcase, self).__init__(**kwargs)
        self.inject_till = inject_till

    def build(self, sim):
        super(AgentShowcase, self).build(sim)

        assert self.n_effectors == 1
        assert isinstance(self.effectors[0], Injector)
        self.injector = self.effectors[0]

        self.injector.set_act_range(self.sim.particles_ng.used.to_numpy()[0])

    def set_action(self, s, s_global, n_substeps, action):
        # super(AgentShowcase, self).set_action(s, s_global, n_substeps, action)
        injector_action = np.zeros(3)
        self.injector.set_action(s, s_global, n_substeps, injector_action)

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
        return mat_v

    def check_act_range(self, f):
        assert self.injector.act_id[f+1] <= self.injector.act_range.shape[0], 'too many particles added'
            
    # @property
    # def action_dim(self):
    #     return self.rigid.action_dim

    # @property
    # def state_dim(self):
    #     return self.rigid.state_dim

    def set_action(self, s, s_global, n_substeps, action):
        pass
        # action = np.asarray(action).reshape(-1).clip(-1, 1)
        # assert len(action) == self.rigid.action_dim
        # self.rigid.set_action(s, s_global, n_substeps, action)

    def set_action_grad(self, s, s_global, n_substeps, action):
        pass
        # action = np.asarray(action).reshape(-1).clip(-1, 1)
        # assert len(action) == self.rigid.action_dim
        # self.rigid.set_action_grad(s, s_global, n_substeps, action)

    def apply_action_p(self, action_p):
        pass
        # action_p = np.asarray(action_p).reshape(-1).clip(0.05, 0.95)
        # self.rigid.apply_action_p(action_p)
            
    def apply_action_p_grad(self, action_p):
        pass
        # action_p = np.asarray(action_p).reshape(-1).clip(0.05, 0.95)
        # self.rigid.apply_action_p_grad(action_p)

    def get_grad(self, n):
        grad = self.rigid.get_action_grad(0, n)
        return grad

    def move(self, f):
        for i in range(self.n_effectors):
            self.effectors[i].move(f)

    def move_grad(self, f):
        self.injector.move_grad(f)

    def get_state(self, f):
        out = []
        out.append(self.injector.get_state(f))
        return out

    def set_state(self, f, state):
        self.injector.set_state(f, state[0])

    def get_ckpt(self, ckpt_dest, ckpt_name=None):
        if ckpt_dest == 'disk':
            return self.injector.get_ckpt()
        elif ckpt_dest in ['cpu', 'gpu']:
            self.injector.get_ckpt(ckpt_name)

    def set_ckpt(self, ckpt_dest, ckpt=None, ckpt_name=None):
        if ckpt_dest == 'disk':
            self.injector.set_ckpt(ckpt=ckpt)
        elif ckpt_dest in ['cpu', 'gpu']:
            self.injector.set_ckpt(ckpt_name=ckpt_name)
