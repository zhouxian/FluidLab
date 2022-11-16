import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from sklearn.neighbors import KDTree
from fluidlab.engine.simulators import MPMSimulator
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
import matplotlib.pyplot as plt
from .loss import Loss

@ti.data_oriented
class RLTestLoss(Loss):
    def __init__(self, **kwargs):
        super(RLTestLoss, self).__init__(**kwargs)

    def build(self, sim):
        self.test_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        super(RLTestLoss, self).build(sim)

    def reset_grad(self):
        pass

    def load_target(self, path):
        pass

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.test_loss[s]

    @ti.kernel
    def clear_losses(self):
        self.test_loss.fill(0)
        self.test_loss.grad.fill(0)

    @ti.ad.grad_replaced
    def compute_step_loss(self, s, f):
        self.compute_test_loss_kernel(s, f)
        self.sum_up_loss_kernel(s)

    @ti.ad.grad_for(compute_step_loss)
    def compute_step_loss_grad(self, f):
        pass

    @ti.kernel
    def compute_test_loss_kernel(self, s: ti.i32, f: ti.i32):
        goal = ti.Vector([0.8, 0.5, 0.5], dt=DTYPE_TI)
        self.test_loss[s] = (self.agent.rigid.pos[f] - goal).norm(EPS)

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 10.0 * (0.5 - cur_step_loss)

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = cur_step_loss
        return loss_info
