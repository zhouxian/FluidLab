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
class PouringLoss(Loss):
    def __init__(
            self,
            type,
            **kwargs,
        ):
        super(PouringLoss, self).__init__(**kwargs)

        self.plateau_count_limit     = 10
        self.temporal_expand_speed   = 0
        self.temporal_init_range_end = 0
        self.temporal_range_type     = 'all'
        self.plateau_thresh          = [1e-6, 0.1]
        self.type                    = type

        if self.type == 'diff':
            self.dist_scale = 0.0
        elif self.type == 'default':
            self.dist_scale = 0.2
        else:
            assert False

    def build(self, sim):
        self.dist_weight = self.weights['dist']
        self.dist_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        self.attraction_weight = self.weights['attraction']
        self.attraction_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        self.attraction_loss_ = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)
        self.best_particle_pos = ti.Vector.field(3, dtype=DTYPE_TI, shape=())

        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        super(PouringLoss, self).build(sim)
        self.init_particle_pos = ti.Vector.field(3, dtype=DTYPE_TI, shape=(self.sim.n_particles))
        self.attraction_score = ti.field(dtype=DTYPE_TI, shape=(self.sim.n_particles))
        self.attraction_scale = ti.field(dtype=DTYPE_TI, shape=())

    def reset_grad(self):
        super(PouringLoss, self).reset_grad()
        self.dist_loss.grad.fill(0)
        self.attraction_loss.grad.fill(0)
        self.attraction_loss_.grad.fill(0)
        
    @ti.kernel
    def clear_losses(self):
        self.dist_loss.fill(0)
        self.dist_loss.grad.fill(0)
        self.attraction_loss.fill(0)
        self.attraction_loss.grad.fill(0)
        self.attraction_loss_.fill(0)
        self.attraction_loss_.grad.fill(0)

    def compute_step_loss(self, s, f):
        if s == 0:
            self.get_init_particles(s, f)

        if self.type == 'diff' and s == self.max_loss_steps - 1:
            self.find_best_particle(s, f)
            self.compute_attraction_loss(s, f)

        self.compute_dist_loss(s, f)
        self.sum_up_loss_kernel(s)

    def compute_step_loss_grad(self, s, f):
        self.sum_up_loss_kernel.grad(s)
        self.compute_dist_loss_grad(s, f)

        if self.type == 'diff' and s == self.max_loss_steps - 1:
            self.find_best_particle(s, f)
            self.compute_attraction_loss_grad(s, f)

    def compute_dist_loss(self, s, f):
        self.compute_dist_loss_kernel(s, f)

    def compute_dist_loss_grad(self, s, f):
        self.compute_dist_loss_kernel.grad(s, f)

    def get_init_particles(self, s, f):
        x = self.sim.get_x()
        self.init_particle_pos.from_numpy(x)

    def find_best_particle(self, s, f):
        x = self.sim.get_x()
        used = self.sim.get_used()
        mat = self.particle_mat.to_numpy()
        loss = np.abs(x[:, 1] - 0.05)
        loss[np.logical_not(used)] = 1000
        loss[mat != WATER] = 1000
        best_particle_id = np.argmin(loss)
        self.best_particle_pos.from_numpy(x[best_particle_id])

        dist = np.linalg.norm(x - x[best_particle_id], axis=1)
        dist[np.logical_not(used)] = 1000
        dist[mat != WATER] = 1000
        score = np.argsort(np.argsort(dist))
        self.attraction_score.from_numpy(score)

        x[mat != WATER] = np.array([-100, -100, -100])
        x[np.logical_not(used)] = np.array([-100, -100, -100])
        self.attraction_scale[None] = np.sum(x[:, 1] > 0.55) / 12500


    def compute_attraction_loss(self, s, f):
        self.compute_attraction_loss_kernel(s, f)
        self.scale_attraction_loss_kernel(s, f)
        print(self.attraction_loss[s])

    def compute_attraction_loss_grad(self, s, f):
        self.scale_attraction_loss_kernel.grad(s, f)
        self.compute_attraction_loss_kernel.grad(s, f)

    @ti.kernel
    def compute_dist_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.particle_used[f, p] and self.particle_mat[p] == WATER:
                self.dist_loss[s] += ti.abs(self.particle_x[f, p][1] - 0.05) * self.dist_scale
            if self.particle_used[f, p] and self.particle_mat[p] == MILK:
                self.dist_loss[s] += ti.abs(self.particle_x[f, p] - self.init_particle_pos[p]).sum()

    @ti.kernel
    def compute_attraction_loss_kernel(self, s: ti.i32, f: ti.i32):
        for p in range(self.n_particles):
            if self.attraction_score[p] < 100:
                if self.particle_used[f, p] and self.particle_mat[p] == WATER:
                    self.attraction_loss_[s] += ti.abs(self.particle_x[f, p] - self.best_particle_pos[None]).sum() * 5000

    @ti.kernel
    def scale_attraction_loss_kernel(self, s: ti.i32, f: ti.i32):
        self.attraction_loss[s] = self.attraction_loss_[s] * self.attraction_scale[None]

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.dist_loss[s] * self.dist_weight + self.attraction_loss[s] + self.attraction_weight

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
        reward = 0.001 * (5000 - cur_step_loss)
        loss =  0.001 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info