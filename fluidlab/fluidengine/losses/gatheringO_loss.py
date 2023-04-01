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
from .loss import Loss

@ti.data_oriented
class GatheringOLoss(Loss):
    def __init__(
            self,
            type,
            matching_mat,
            **kwargs,
        ):
        super(GatheringOLoss, self).__init__(**kwargs)

        self.matching_mat = matching_mat

        if type == 'diff':
            self.plateau_count_limit     = 10
            self.temporal_expand_speed   = 120
            self.temporal_init_range_end = 120
            self.temporal_range_type     = 'expand'
            self.plateau_thresh          = [1e-6, 0.1]
        elif type == 'default':
            self.temporal_range_type     = 'all'
        else:
            assert False


    def build(self, sim):
        self.dist_weight = self.weights['dist']
        self.dist_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)

        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        super(GatheringOLoss, self).build(sim)

    def reset_grad(self):
        super(GatheringOLoss, self).reset_grad()
        self.dist_loss.grad.fill(0)
        
    @ti.kernel
    def clear_losses(self):
        self.dist_loss.fill(0)
        self.dist_loss.grad.fill(0)

    def compute_step_loss(self, s, f):
        self.compute_dist_loss(s, f)
        self.sum_up_loss_kernel(s)

    def compute_step_loss_grad(self, s, f):
        self.sum_up_loss_kernel.grad(s)
        self.compute_dist_loss_grad(s, f)

    def compute_dist_loss(self, s, f):
        self.compute_dist_loss_kernel(s, f)

    def compute_dist_loss_grad(self, s, f):
        self.compute_dist_loss_kernel.grad(s, f)

    @ti.kernel
    def compute_dist_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_used[f, p] and self.particle_mat[p] == self.matching_mat:
                self.dist_loss[s] += ti.pow(self.particle_x[f, p][0] - 0.88, 2) + ti.pow(self.particle_x[f, p][2] - 0.78, 2)

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.dist_loss[s] * self.dist_weight

    @ti.kernel
    def compute_total_loss_kernel(self, s_start: ti.i32, s_end: ti.i32):
        for s in range(s_start, s_end):
            self.total_loss[None] += self.step_loss[s]

    def get_final_loss(self):
        self.compute_total_loss_kernel(self.temporal_range[0], self.temporal_range[1])
        self.expand_temporal_range()
        
        loss_info = {
            'loss': self.total_loss[None],
            'last_step_loss': self.step_loss[self.max_loss_steps-1],
            'temporal_range': self.temporal_range[1],
        }

        return loss_info

    def get_final_loss_grad(self):
        self.compute_total_loss_kernel.grad(self.temporal_range[0], self.temporal_range[1])

    def expand_temporal_range(self):
        if self.temporal_range_type == 'expand':
            loss_improved = (self.best_loss - self.total_loss[None])
            loss_improved_rate = loss_improved / self.best_loss
            if loss_improved_rate < self.plateau_thresh[0] or loss_improved < self.plateau_thresh[1]:
                self.plateau_count += 1
                print('Plateaued!!!', self.plateau_count)
            else:
                self.plateau_count = 0

            if self.best_loss > self.total_loss[None]:
                self.best_loss = self.total_loss[None]

            if self.plateau_count >= self.plateau_count_limit:
                self.plateau_count = 0
                self.best_loss = self.inf

                self.temporal_range[1] = min(self.max_loss_steps, self.temporal_range[1] + self.temporal_expand_speed)
                print(f'temporal range expanded to {self.temporal_range}')
            
    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 0.01 * (65 - cur_step_loss)
        loss = 0.01 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info