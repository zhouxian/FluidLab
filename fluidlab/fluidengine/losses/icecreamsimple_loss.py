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
class IceCreamSimpleLoss(ShapeMatchingLoss):
    def __init__(self, type, **kwargs):
        if type == 'diff':
            super(IceCreamSimpleLoss, self).__init__(
                matching_mat=ICECREAM1,
                temporal_init_range_end=100,
                temporal_range_type='expand',
                **kwargs
            )
        elif type == 'default':
            super(IceCreamSimpleLoss, self).__init__(
                matching_mat        = ICECREAM1,
                temporal_range_type = 'all',
                **kwargs
            )
        
    def build(self, sim):
        super(IceCreamSimpleLoss, self).build(sim)

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 0.001 * (750 - cur_step_loss)
        loss = cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info
        
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
            