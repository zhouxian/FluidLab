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
class CirculationLoss(Loss):
    def __init__(
            self,
            type,
            **kwargs,
        ):
        super(CirculationLoss, self).__init__(**kwargs)

        self.plateau_count_limit     = 10
        self.temporal_expand_speed   = 0
        self.temporal_init_range_end = 0
        self.temporal_range_type     = 'all'
        self.plateau_thresh          = [1e-6, 0.1]

    def build(self, sim):
        self.temp_weight = self.weights['temp']
        self.temp_loss = ti.field(dtype=DTYPE_TI, shape=(self.max_loss_steps,), needs_grad=True)

        if self.temporal_range_type == 'last':
            self.temporal_range = [self.max_loss_steps-1, self.max_loss_steps]
        elif self.temporal_range_type == 'all':
            self.temporal_range = [0, self.max_loss_steps]
        elif self.temporal_range_type == 'expand':
            self.temporal_range = [0, self.temporal_init_range_end]
            self.best_loss = self.inf
            self.plateau_count = 0

        self.target_temp = 0.0
        self.detector_array_N = 15
        self.detector_array = ti.Vector.field(3, dtype=ti.i32, shape=(self.detector_array_N,))
        self.detector_h = 64
        self.detector_array.from_numpy(np.array([
            [25, self.detector_h, 85],
            [35, self.detector_h, 85],
            [15, self.detector_h, 85],
            [25, self.detector_h, 75],
            [25, self.detector_h, 95],

            [25, self.detector_h, 42],
            [35, self.detector_h, 42],
            [15, self.detector_h, 42],
            [25, self.detector_h, 32],
            [25, self.detector_h, 52],

            [107, self.detector_h, 65],
            [115, self.detector_h, 65],
            [99, self.detector_h, 65],
            [107, self.detector_h, 45],
            [107, self.detector_h, 85],
        ], dtype=np.int32))

        super(CirculationLoss, self).build(sim)

    def reset_grad(self):
        super(CirculationLoss, self).reset_grad()
        self.temp_loss.grad.fill(0)
        
    @ti.kernel
    def clear_losses(self):
        self.temp_loss.fill(0)
        self.temp_loss.grad.fill(0)

    def step(self):
        # compute loss for step self.sim.cur_step_global-1
        self.compute_step_loss(self.sim.cur_step_global-1, self.sim.cur_step_local)

    def step_grad(self):
        # compute loss for step self.sim.cur_step_global-1
        self.compute_step_loss_grad(self.sim.cur_step_global-1, self.sim.cur_step_local)

    def compute_step_loss(self, s_global, s_local):
        self.compute_temp_loss(s_global, s_local)
        self.sum_up_loss_kernel(s_global)

    def compute_step_loss_grad(self, s_global, s_local):
        self.sum_up_loss_kernel.grad(s_global)
        self.compute_temp_loss_grad(s_global, s_local)

    def compute_temp_loss(self, s_global, s_local):
        self.compute_temp_loss_kernel(s_global, s_local)

    def compute_temp_loss_grad(self, s_global, s_local):
        self.compute_temp_loss_kernel.grad(s_global, s_local)

    @ti.kernel
    def compute_temp_loss_kernel(self, s_global: ti.i32, s_local: ti.i32):
        for i in range(self.detector_array_N):
            x, y, z = self.detector_array[i]
            if i < 5:
                self.temp_loss[s_global] += ti.abs(self.smoke_field.grid.q[s_local, x, y, z][0] - 1.0)
            else:
                self.temp_loss[s_global] += ti.abs(self.smoke_field.grid.q[s_local, x, y, z][0] - self.target_temp)

    @ti.kernel
    def sum_up_loss_kernel(self, s: ti.i32):
        self.step_loss[s] += self.temp_loss[s] * self.temp_weight

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
        pass

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 1.0 * (11 - cur_step_loss)
        loss = 1.0 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info