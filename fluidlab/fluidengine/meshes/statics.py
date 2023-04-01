import os
import copy
import pickle as pkl
import trimesh
import taichi as ti
import numpy as np
from .static import Static

class Statics:
    # Static objects, where each one is a Mesh
    def __init__(self):
        self.statics = []

    def add_static(self, **kwargs):
        self.statics.append(Static(**kwargs))

    def __getitem__(self, index):
        return self.statics[index]

    def __len__(self):
        return len(self.statics)
