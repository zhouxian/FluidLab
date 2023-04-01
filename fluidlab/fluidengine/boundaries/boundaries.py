import copy
import numpy as np
import taichi as ti
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *
from fluidlab.utils.config import make_cls_config

@ti.data_oriented
class Boundary:
    def __init__(self, restitution=0.0, lock_dims=[]):
        self.restitution = restitution
        self.lock_dims = lock_dims

    @ti.func
    def impose_x_v(self, x, v):
        raise NotImplementedError

    @ti.func
    def impose_x(self, x):
        raise NotImplementedError

    @ti.func
    def is_out(self, x):
        raise NotImplementedError


@ti.data_oriented
class CylinderBoundary(Boundary):
    def __init__(self, y_range=(0.05, 0.95), xz_center=(0.5, 0.5), xz_radius=0.45, **kwargs):
        super(CylinderBoundary, self).__init__(**kwargs)

        y_range = np.array(eval_str(y_range), dtype=DTYPE_NP)
        xz_center = np.array(eval_str(xz_center), dtype=DTYPE_NP)
        self.y_lower = ti.Vector([0.0, y_range[0], 0.0], dt=DTYPE_TI)
        self.y_upper = ti.Vector([1.0, y_range[1], 1.0], dt=DTYPE_TI)
        self.xz_center = ti.Vector(xz_center, dt=DTYPE_TI)
        self.xz_radius = xz_radius

    @ti.func
    def impose_x_v(self, x, v):
        # y direction
        if x[1] > self.y_upper[1] and v[1] > 0.0:
            v[1] *= -self.restitution
        elif x[1] < self.y_lower[1] and v[1] < 0.0:
            v[1] *= -self.restitution

        x_new = ti.max(ti.min(x, self.y_upper), self.y_lower)

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            new_xz = r_vector / r_vector_norm * self.xz_radius + self.xz_center
            new_y = x_new[1]
            x_new = ti.Vector([new_xz[0], new_y, new_xz[1]])
            v[0] = 0.0
            v[2] = 0.0

        # enforce lock_dims
        for i in ti.static(self.lock_dims):
            v[i] = 0.0

        return x_new, v

    @ti.func
    def impose_x(self, x):
        # y direction
        x_new = ti.max(ti.min(x, self.y_upper), self.y_lower)

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            new_xz = r_vector / r_vector_norm * self.xz_radius + self.xz_center
            new_y = x_new[1]
            x_new = ti.Vector([new_xz[0], new_y, new_xz[1]])

        return x_new

    @ti.func
    def is_out(self, x):
        out = False

        # y direction
        if x[1] > self.y_upper[1] or x[1] < self.y_lower[1]:
            out = True

        # xz direction
        r_vector = ti.Vector([x[0], x[2]]) - self.xz_center
        r_vector_norm = r_vector.norm(EPS)
        if r_vector_norm > self.xz_radius:
            out = True
        return out

@ti.data_oriented
class CubeBoundary(Boundary):
    def __init__(self, lower=(0.05, 0.05, 0.05), upper=(0.95, 0.95, 0.95), **kwargs):
        super(CubeBoundary, self).__init__(**kwargs)
        upper = np.array(eval_str(upper), dtype=DTYPE_NP)
        lower = np.array(eval_str(lower), dtype=DTYPE_NP)
        assert (upper >= lower).all()

        self.upper = ti.Vector(upper, dt=DTYPE_TI)
        self.lower = ti.Vector(lower, dt=DTYPE_TI)

    @ti.func
    def impose_x_v(self, x, v):
        for i in ti.static(range(3)):
            if x[i] >= self.upper[i] and v[i] >=0:
                v[i] *= -self.restitution
            elif x[i] <= self.lower[i] and v[i] <=0:
                v[i] *= -self.restitution

        x_new = ti.max(ti.min(x, self.upper), self.lower)

        # enforce lock_dims
        for i in ti.static(self.lock_dims):
            v[i] = 0.0

        return x_new, v

    @ti.func
    def impose_x(self, x):
        x_new = ti.max(ti.min(x, self.upper), self.lower)
        return x_new

    @ti.func
    def is_out(self, x):
        out = False

        if any(x > self.upper) or any(x < self.lower):
            out = True
            
        return out

def create_boundary(type='cube', **kwargs):
    if type == 'cylinder':
        return CylinderBoundary(**kwargs)
    if type == 'cube':
        return CubeBoundary(**kwargs)
    else:
        assert False