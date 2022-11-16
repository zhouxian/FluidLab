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
from .shapematching_loss import ShapeMatchingLoss

@ti.data_oriented
class LatteArtLoss(ShapeMatchingLoss):
    def __init__(self, type, **kwargs):
        super(LatteArtLoss, self).__init__(
            matching_mat=MILK,
            temporal_range_type='all',
            **kwargs
        )

    def build(self, sim):
        super(LatteArtLoss, self).build(sim)

    def get_step_loss(self):
        cur_step_loss = self.step_loss[self.sim.cur_step_global-1]
        reward = 0.1 * (50 - cur_step_loss)
        loss = 0.1 * cur_step_loss

        loss_info = {}
        loss_info['reward'] = reward
        loss_info['loss'] = loss
        return loss_info