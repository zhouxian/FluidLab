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
class ShapeMatchingLoss(Loss):
    def __init__(
            self,
            matching_mat,
            temporal_range_type     = 'expand',
            temporal_init_range_end = 50,
            plateau_count_limit     = 5,
            temporal_expand_speed   = 50,
            plateau_thresh          = [0.01, 0.5],
            **kwargs,
        ):
        super(ShapeMatchingLoss, self).__init__(**kwargs)

        self.matching_mat            = matching_mat
        self.temporal_range_type     = temporal_range_type
        self.temporal_init_range_end = temporal_init_range_end
        self.plateau_count_limit     = plateau_count_limit
        self.temporal_expand_speed   = temporal_expand_speed
        self.plateau_thresh          = plateau_thresh

    def build(self, sim):
        self.chamfer_weight = self.weights['chamfer']
        self.chamfer_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        super(ShapeMatchingLoss, self).build(sim)

    def reset_grad(self):
        super(ShapeMatchingLoss, self).reset_grad()
        self.chamfer_loss.grad.fill(0)
        
    def load_target(self, path):
        self.target = pkl.load(open(path, 'rb'))
        assert self.max_loss_steps == len(self.target['x'])
        assert self.n_particles == len(self.target['x'][0])
        self.tgt_particles_x = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=self.n_particles)
        print(f'===>  Target loaded from {path}.')

    @ti.kernel
    def clear_losses(self):
        self.chamfer_loss.fill(0)
        self.chamfer_loss.grad.fill(0)

    def compute_step_loss(self, s, f):
        self.compute_chamfer_loss(s, f)
        self.sum_up_loss_kernel(s)

    def compute_step_loss_grad(self, s, f):
        self.sum_up_loss_kernel.grad(s)
        self.compute_chamfer_loss_grad(s, f)

    def compute_chamfer_loss(self, s, f):
        self.tgt_particles_x.from_numpy(self.target['x'][s])
        self.compute_chamfer_loss_kernel(s, f)

    def compute_chamfer_loss_grad(self, s, f):
        self.tgt_particles_x.from_numpy(self.target['x'][s])
        self.compute_chamfer_loss_kernel.grad(s, f)

    @ti.kernel
    def compute_chamfer_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_used[f, p] and self.particle_mat[p] == self.matching_mat:
                self.chamfer_loss[s] += ti.pow(self.particle_x[f, p] - self.tgt_particles_x[p], 2).sum()

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.chamfer_loss[s] * self.chamfer_weight

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
            