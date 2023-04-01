import taichi as ti
import numpy as np
import yaml
import torch
from fluidlab.fluidengine.effectors import *
from fluidlab.utils.misc import *

@ti.data_oriented
class Agent:
    # Agent with (possibly) multiple effectors.
    def __init__(
        self,
        max_substeps_local,
        max_substeps_global,
        max_action_steps_global,
        ckpt_dest,
        collide_type='particle' # ['particle', 'grid', 'both']
    ):

        self.max_substeps_local = max_substeps_local
        self.max_substeps_global = max_substeps_global
        self.max_action_steps_global = max_action_steps_global
        self.ckpt_dest = ckpt_dest

        self.collide_type = collide_type
        assert self.collide_type in ['particle', 'grid', 'both']

        self.effectors = []
        self.action_dims = [0]

    def add_effector(self, type, params, mesh_cfg, boundary_cfg):
        effector = eval(type)(
            max_substeps_local=self.max_substeps_local,
            max_substeps_global=self.max_substeps_global,
            max_action_steps_global=self.max_action_steps_global,
            ckpt_dest=self.ckpt_dest,
            **params,
        )
        if mesh_cfg is not None:
            effector.setup_mesh(**mesh_cfg)
        effector.setup_boundary(**boundary_cfg)

        self.effectors.append(effector)
        self.action_dims.append(self.action_dims[-1] + effector.action_dim)

    def build(self, sim):
        self.n_effectors = len(self.effectors)
        self.sim = sim

        for effector in self.effectors:
            effector.build()

    def reset_grad(self):
        for i in range(self.n_effectors):
            self.effectors[i].reset_grad()

    def act(self, f, f_global):
        return

    def act_grad(self, f, f_global):
        return

    @property
    def action_dim(self):
        return self.action_dims[-1]

    @property
    def state_dim(self):
        return sum([i.state_dim for i in self.effectors])

    def set_action(self, s, s_global, n_substeps, action):
        action = np.asarray(action).reshape(-1)
        assert len(action) == self.action_dims[-1], 'Action length does not match agent specifications.'
        for i in range(self.n_effectors):
            self.effectors[i].set_action(s, s_global, n_substeps, action[self.action_dims[i]:self.action_dims[i+1]])

    def set_action_grad(self, s, s_global, n_substeps, action):
        action = np.asarray(action).reshape(-1)
        assert len(action) == self.action_dims[-1]
        for i in range(self.n_effectors-1, -1, -1):
            self.effectors[i].set_action_grad(s, s_global, n_substeps, action[self.action_dims[i]:self.action_dims[i+1]])

    def apply_action_p(self, action_p):
        action_p = np.asarray(action_p).reshape(-1)
        for i in range(self.n_effectors):
            self.effectors[i].apply_action_p(action_p[self.action_dims[i]:self.action_dims[i+1]])
            
    def apply_action_p_grad(self, action_p):
        action_p = np.asarray(action_p).reshape(-1)
        for i in range(self.n_effectors-1, -1, -1):
            self.effectors[i].apply_action_p_grad(action_p[self.action_dims[i]:self.action_dims[i+1]])

    def get_grad(self, n):
        grads = []
        for i in range(self.n_effectors):
            grad = self.effectors[i].get_action_grad(0, n)
            if grad is not None:
                grads.append(grad)
        return np.concatenate(grads, axis=1)

    def move(self, f):
        for i in range(self.n_effectors):
            self.effectors[i].move(f)

    def move_grad(self, f):
        for i in range(self.n_effectors-1, -1, -1):
            self.effectors[i].move_grad(f)

    def get_state(self, f):
        out = []
        for i in range(self.n_effectors):
            out.append(self.effectors[i].get_state(f))
        return out

    def set_state(self, f, state):
        for i in range(self.n_effectors):
            self.effectors[i].set_state(f, state[i])

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].copy_frame(source, target)

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].copy_grad(source, target)

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i in ti.static(range(self.n_effectors)):
            self.effectors[i].reset_grad_till_frame(f)

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            out = []
            for effector in self.effectors:
                out.append(effector.get_ckpt())
            return out

        elif self.ckpt_dest in ['cpu', 'gpu']:
            for effector in self.effectors:
                effector.get_ckpt(ckpt_name)

    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            for effector, ckpt_effector in zip(self.effectors, ckpt):
                effector.set_ckpt(ckpt=ckpt_effector)

        elif self.ckpt_dest in ['cpu', 'gpu']:
            for effector in self.effectors:
                effector.set_ckpt(ckpt_name=ckpt_name)