import os
import torch
import numpy as np
import taichi as ti
import pickle as pkl
from sklearn.neighbors import KDTree
from fluidlab.fluidengine.simulators import MPMSimulator
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
import matplotlib.pyplot as plt
from .shapematching_loss import ShapeMatchingLoss

@ti.data_oriented
class LatteArtStirLoss(ShapeMatchingLoss):
    def __init__(self, type, **kwargs):
        if type == 'diff':
            super(LatteArtStirLoss, self).__init__(
                matching_mat            = MILK_VIS,
                temporal_init_range_end = 50,
                temporal_range_type     = 'expand',
                plateau_count_limit     = 5,
                temporal_expand_speed   = 10,
                plateau_thresh          = [0.01, 0.1],
                **kwargs
            )
        elif type == 'default':
            super(LatteArtStirLoss, self).__init__(
                matching_mat            = MILK_VIS,
                temporal_range_type     = 'all',
                **kwargs
            )
        else:
            assert False

        self.step_loss_milk = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=False)
        self.total_loss_milk = ti.field(dtype=DTYPE_TI, shape=(), needs_grad=False)

    @ti.kernel
    def clear_loss(self):
        self.step_loss.fill(0)
        self.step_loss.grad.fill(0)

        self.total_loss.fill(0)
        self.total_loss.grad.fill(1)

        self.step_loss_milk.fill(0)
        self.total_loss_milk.fill(0)

    def build(self, sim):
        self.chamfer_loss_milk = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=False)
        super(LatteArtStirLoss, self).build(sim)

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.chamfer_loss[s] * self.chamfer_weight
        self.step_loss_milk[s] += self.chamfer_loss_milk[s] * self.chamfer_weight

    @ti.kernel
    def compute_total_loss_kernel(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            self.total_loss[None] += self.step_loss[s]
            self.total_loss_milk[None] += self.step_loss_milk[s]

    @ti.kernel
    def compute_chamfer_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_used[f, p]:
                loss_p = ti.pow(self.particle_x[f, p] - self.tgt_particles_x[p], 2).sum()
                if self.particle_mat[p] == self.matching_mat:
                    self.chamfer_loss_milk[s] += loss_p
                self.chamfer_loss[s] += loss_p

    @ti.kernel
    def clear_losses(self):
        self.chamfer_loss.fill(0)
        self.chamfer_loss.grad.fill(0)
        self.chamfer_loss_milk.fill(0)

    def get_final_loss(self):
        self.compute_total_loss_kernel(self.temporal_range[0], self.temporal_range[1])
        self.expand_temporal_range()
        
        loss_info = {
            'loss': self.total_loss[None],
            'loss_milk': self.total_loss_milk[None],
            'last_step_loss': self.step_loss[self.max_loss_steps-1],
            'temporal_range': self.temporal_range[1],
            'reward': np.sum((1000 - self.step_loss.to_numpy()) * 0.002)
        }

        return loss_info

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 0.002 * (1000 - cur_step_loss)
        loss = 0.002 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info