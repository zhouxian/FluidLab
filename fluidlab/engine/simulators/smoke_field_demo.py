import colorsys
import taichi as ti
import numpy as np
import pickle as pkl
import uuid
import os
import torch
from time import time
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
import fluidlab.utils.geom as geom_utils

@ti.data_oriented
class SmokeFieldDemo:
    def __init__(self, dim, ckpt_dest, res=128, dt=0.03, solver_iters=500, q_dim=3, decay=0.99):
        self.dim          = dim
        self.ckpt_dest    = ckpt_dest
        self.n_grid       = res
        self.dx           = 1 / self.n_grid
        self.res          = (res,) * self.dim
        self.dt           = dt
        self.solver_iters = solver_iters
        self.q_dim        = q_dim
        self.decay        = decay
        self.high_T       = 1.0
        self.low_T        = 0.0
        self.lower_y      = -1
        self.higher_y     = 200
        self.lower_y_vis  = -1
        self.higher_y_vis = 200
        self.new_color    = np.array([1.0, 0.3, 0.2])
        self.prev_color   = np.array(self.new_color)
        self.color = ti.Vector.field(3, DTYPE_TI, shape=())
        self.color.from_numpy(self.prev_color)

        print(f'===>  Smoke field of {self.res} initialized.')

    def build(self, mpm_sim, agent):
        self.mpm_sim         = mpm_sim
        self.max_steps_local = mpm_sim.max_steps_local
        self.mpm_grid_ratio  = self.n_grid / mpm_sim.n_grid
        self.mpm_v_coeff     = 1.0
        self.agent           = mpm_sim.agent

        self.setup_fields()
        self.init_fields()
        self.init_ckpt()

    def init_ckpt(self):
        if self.ckpt_dest == 'disk':
            # placeholder np array from checkpointing
            self.v_np     = np.zeros((*self.res, self.dim), dtype=DTYPE_NP)
            self.v_tmp_np = np.zeros((*self.res, self.dim), dtype=DTYPE_NP)
            self.div_np   = np.zeros((*self.res,), dtype=DTYPE_NP)
            self.p_np     = np.zeros((*self.res,), dtype=DTYPE_NP)
            self.q_np     = np.zeros((*self.res, self.q_dim), dtype=DTYPE_NP)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = dict()

    def setup_fields(self):
        '''
        Taichi fields for smoke simulation.
        '''
        cell_state = ti.types.struct(
            v       = ti.types.vector(self.dim, DTYPE_TI),
            v_tmp   = ti.types.vector(self.dim, DTYPE_TI),
            div     = DTYPE_TI,
            p       = DTYPE_TI,
            q       = ti.types.vector(self.q_dim, DTYPE_TI),
        )

        cell_state_ng = ti.types.struct(
            is_free = ti.i32,
        )

        self.grid    = cell_state.field(shape=(self.max_steps_local+1, *self.res), needs_grad=True, layout=ti.Layout.SOA)
        self.grid_ng = cell_state_ng.field(shape=(self.max_steps_local+1, *self.res), needs_grad=False, layout=ti.Layout.SOA)

        # swap area for pressure projection solver
        self.p_swap = TexPair(
            cur = ti.field(dtype=DTYPE_TI, shape=self.res, needs_grad=True),
            nxt = ti.field(dtype=DTYPE_TI, shape=self.res, needs_grad=True),
        )

        self.vis_particles   = ti.Vector.field(3, float, shape=np.prod(self.res))
        self.vis_particles_c = ti.Vector.field(4, float, shape=np.prod(self.res))


    @ti.kernel
    def init_fields(self):
        for i, j, k in ti.ndrange(*self.res):
            ind = j * self.n_grid * self.n_grid + i * self.n_grid + k
            self.vis_particles[ind] = (ti.Vector([i, j, k], dt=ti.f32) + 0.5) * self.dx

    def step(self, s, f):
        update_interval = 50
        if self.mpm_sim.cur_step_global % update_interval == 0: # update color
            self.prev_color = self.new_color
            self.new_color = np.array(colorsys.hsv_to_rgb(np.random.rand(),1,1))
            self.color_diff = self.new_color - self.prev_color
        self.prev_color += self.color_diff / update_interval
        self.color.from_numpy(self.prev_color)

        self.compute_free_space(s, f)
        self.advect_and_impulse(s, f)
        self.divergence(s)

        # projection
        self.reset_swap_and_grad()
        self.pressure_to_swap(s)
        for i in range(self.solver_iters):
            self.pressure_jacobi(self.p_swap.cur, self.p_swap.nxt, s)
            self.p_swap.swap()
        self.pressure_from_swap(s)
        self.reset_swap_and_grad()

        self.subtract_gradient(s)
        self.colorize(s)

    def step_grad(self, s, f):
        self.compute_free_space(s, f)

        self.subtract_gradient.grad(s)

        self.reset_swap_and_grad()
        self.pressure_from_swap.grad(s)
        for i in range(self.solver_iters-1, -1, -1):
            self.p_swap.swap()
            self.p_swap.cur.grad.fill(0)
            self.pressure_jacobi.grad(self.p_swap.cur, self.p_swap.nxt, s)
        self.pressure_to_swap.grad(s)
        self.reset_swap_and_grad()

        self.divergence.grad(s)
        self.advect_and_impulse.grad(s, f)

    def reset_swap_and_grad(self):
        self.p_swap.cur.fill(0)
        self.p_swap.nxt.fill(0)
        self.p_swap.cur.grad.fill(0)
        self.p_swap.nxt.grad.fill(0)

    @ti.kernel
    def pressure_jacobi(self, pf: ti.template(), new_pf: ti.template(), s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                pl = pf[self.compute_location(s, i, j, k, -1, 0, 0)]
                pr = pf[self.compute_location(s, i, j, k, 1, 0, 0)]
                pb = pf[self.compute_location(s, i, j, k, 0, -1, 0)]
                pt = pf[self.compute_location(s, i, j, k, 0, 1, 0)]
                pp = pf[self.compute_location(s, i, j, k, 0, 0, -1)]
                pq = pf[self.compute_location(s, i, j, k, 0, 0, 1)]

                new_pf[i, j, k] = (pl + pr + pb + pt + pp + pq - self.grid[s, i, j, k].div) / 6.0

    @ti.kernel
    def pressure_jacobi_grad(self, pf: ti.template(), new_pf: ti.template(), s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:

                self.grid.grad[s, i, j, k].div += - 1.0 / 6.0 * new_pf.grad[i, j, k]

                pf.grad[self.compute_location(s, i, j, k, -1, 0, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 1, 0, 0)]  += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, -1, 0)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 1, 0)]  += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 0, -1)] += 1.0 / 6.0 * new_pf.grad[i, j, k]
                pf.grad[self.compute_location(s, i, j, k, 0, 0, 1)]  += 1.0 / 6.0 * new_pf.grad[i, j, k]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid[target, i, j, k].v     = self.grid[source, i, j, k].v
            self.grid[target, i, j, k].v_tmp = self.grid[source, i, j, k].v_tmp
            self.grid[target, i, j, k].div   = self.grid[source, i, j, k].div
            self.grid[target, i, j, k].p     = self.grid[source, i, j, k].p
            self.grid[target, i, j, k].q     = self.grid[source, i, j, k].q

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid.grad[target, i, j, k].v     = self.grid.grad[source, i, j, k].v
            self.grid.grad[target, i, j, k].v_tmp = self.grid.grad[source, i, j, k].v_tmp
            self.grid.grad[target, i, j, k].div   = self.grid.grad[source, i, j, k].div
            self.grid.grad[target, i, j, k].p     = self.grid.grad[source, i, j, k].p
            self.grid.grad[target, i, j, k].q     = self.grid.grad[source, i, j, k].q

    def reset_grad(self):
        self.grid.grad.fill(0)
        self.p_swap.cur.grad.fill(0)
        self.p_swap.nxt.grad.fill(0)

    @ti.kernel
    def reset_grad_till_frame(self, s: ti.i32):
        for n, i, j, k in ti.ndrange(s, *self.res):
            self.grid.grad[n, i, j, k].fill(0)

    @ti.kernel
    def compute_free_space(self, s: ti.i32, f: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            self.grid_ng[s, i, j, k].is_free = 0
            if self.lower_y < j < self.higher_y:
                self.grid_ng[s, i, j, k].is_free = 1

            p = ti.Vector([i, j, k], dt=DTYPE_TI) + 0.5
            if ti.static(self.mpm_sim.n_statics>0):
                for static_i in ti.static(range(self.mpm_sim.n_statics)):
                    if self.mpm_sim.statics[static_i].is_collide(p*self.dx):
                        self.grid_ng[s, i, j, k].is_free = 0

    @ti.kernel
    def advect_and_impulse(self, s: ti.i32, f: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                p = ti.Vector([i, j, k], dt=DTYPE_TI) + 0.5

                p = self.backtrace(s, self.grid.v, p, self.dt)
                v_f = self.trilerp(s, self.grid.v, p) * self.decay
                q_f = self.trilerp(s, self.grid.q, p) * self.decay

                # apply agent impulse
                imp_pos = self.agent.aircon.pos[f] / self.dx
                imp_dir = geom_utils.transform_by_quat_ti(self.agent.aircon.inject_v, self.agent.aircon.quat[f])
                dist = (ti.Vector([i, j, k]) - imp_pos).norm(EPS)

                factor = ti.exp(- dist**2 / 32)

                momentum = (imp_dir * self.agent.aircon.s[f] * factor) * self.dt

                # compute impulse from mpm particles
                momentum_mpm = ti.Vector([0.0, 0.0, 0.0], dt=DTYPE_TI)
                # I_mpm = ti.floor(ti.Vector([i, j, k], dt=DTYPE_TI) / self.mpm_grid_ratio, ti.i32)
                # if self.mpm_sim.grid[f, I_mpm].mass > EPS:
                #     momentum_mpm = self.mpm_sim.grid[f, I_mpm].v_out * self.mpm_v_coeff

                v_tmp = v_f + momentum + momentum_mpm

                self.grid.v_tmp[s, i, j, k] = v_tmp
                self.grid.q[s+1, i, j, k] = (1 - factor) * q_f + factor * self.color[None]

            else:
                self.grid.v_tmp[s, i, j, k] = ti.Vector([0.0, 0.0, 0.0])
                self.grid.q[s+1, i, j, k] = self.grid.q[s, i, j, k]

    @ti.kernel
    def divergence(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                vl = self.grid.v_tmp[s, self.compute_location(s, i, j, k, -1, 0, 0)]
                vr = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 1, 0, 0)]
                vb = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, -1, 0)]
                vt = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 1, 0)]
                vp = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, -1)]
                vq = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, 1)]
                vc = self.grid.v_tmp[s, self.compute_location(s, i, j, k, 0, 0, 0)]

                if not self.is_free(s, i, j, k, -1, 0, 0):
                    vl.x = -vc.x
                if not self.is_free(s, i, j, k, 1, 0, 0):
                    vr.x = -vc.x
                if not self.is_free(s, i, j, k, 0, -1, 0):
                    vb.y = -vc.y
                if not self.is_free(s, i, j, k, 0, 1, 0):
                    vt.y = -vc.y
                if not self.is_free(s, i, j, k, 0, 0, -1):
                    vp.z = -vc.z
                if not self.is_free(s, i, j, k, 0, 0, 1):
                    vq.z = -vc.z

                self.grid.div[s, i, j, k] = (vr.x - vl.x + vt.y - vb.y + vq.z - vp.z) * 0.5


    @ti.kernel
    def pressure_to_swap(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                self.p_swap.cur[i, j, k] = self.grid.p[s, i, j, k]

    @ti.kernel
    def pressure_from_swap(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:
                self.grid.p[s+1, i, j, k] = self.p_swap.cur[i, j, k]

    @ti.kernel
    def subtract_gradient(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            if self.grid_ng[s, i, j, k].is_free:

                pl = self.grid.p[s+1, self.compute_location(s, i, j, k, -1, 0, 0)]
                pr = self.grid.p[s+1, self.compute_location(s, i, j, k, 1, 0, 0)]
                pb = self.grid.p[s+1, self.compute_location(s, i, j, k, 0, -1, 0)]
                pt = self.grid.p[s+1, self.compute_location(s, i, j, k, 0, 1, 0)]
                pp = self.grid.p[s+1, self.compute_location(s, i, j, k, 0, 0, -1)]
                pq = self.grid.p[s+1, self.compute_location(s, i, j, k, 0, 0, 1)]

                self.grid.v[s+1, i, j, k] = self.grid.v_tmp[s, i, j, k] - 0.5 * ti.Vector([pr - pl, pt - pb, pq - pp])
            else:
                self.grid.v[s+1, i, j, k] = self.grid.v_tmp[s, i, j, k]

    @ti.kernel
    def colorize(self, s: ti.i32):
        for i, j, k in ti.ndrange(*self.res):
            ind = j * self.n_grid * self.n_grid + i * self.n_grid + k
            if self.lower_y_vis < j < self.higher_y_vis:
                color = self.grid.q[s+1, i, j, k]
                self.vis_particles_c[ind][:3] = ti.pow(color / color.max(), 0.75)
                self.vis_particles_c[ind][3] = color.norm(EPS)*0.85
            else:
                self.vis_particles_c[ind] = ti.Vector([0.0, 0.0, 0.0, 0.0])

    @ti.func
    def compute_location(self, s, u, v, w, du, dv, dw):
        I = ti.Vector([int(u+du), int(v+dv), int(w+dw)])
        I = max(0, min(self.n_grid - 1, I))

        if not self.grid_ng[s, I].is_free:
            I = ti.Vector([int(u), int(v), int(w)])

        return I


    @ti.func
    def is_free(self, s, u, v, w, du, dv, dw):
        flag = 1

        I = ti.Vector([int(u+du), int(v+dv), int(w+dw)])
        if (I < 0).any() or (I > self.n_grid - 1).any():
            flag = 0

        elif not self.grid_ng[s, I].is_free:
            flag = 0

        return flag

    @ti.func
    def trilerp(self, f, qf, p):
        '''
        p: position, within (0, 1).
        qf: field for interpolation
        '''
        # convert position to grid index
        base_I = ti.floor(p - 0.5, ti.i32)
        p_I = p - 0.5

        q = ti.Vector.zero(DTYPE_TI, qf.n)
        w_total = 0.0
        for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
            grid_I = base_I + offset
            w_xyz = 1 - ti.abs(p_I - grid_I)
            w = w_xyz[0] * w_xyz[1] * w_xyz[2]
            grid_I_ = self.compute_location(f, grid_I[0], grid_I[1], grid_I[2], 0, 0, 0)
            q += w * qf[f, grid_I_]
            w_total += w
        # w_total is less then one when at boundary
        q /= w_total
        return q

    # RK3
    @ti.func
    def backtrace(self, f, vf, p, dt):
        '''
        vf: velocity field
        '''
        v1 = self.trilerp(f, vf, p)
        p1 = p - 0.5 * dt * v1
        v2 = self.trilerp(f, vf, p1)
        p2 = p - 0.75 * dt * v2
        v3 = self.trilerp(f, vf, p2)
        p -= dt * ((2 / 9) * v1 + (1 / 3) * v2 + (4 / 9) * v3)
        return p

    @ti.kernel
    def readframe(self, s: ti.i32, v_np: ti.types.ndarray(), v_tmp_np: ti.types.ndarray(), div_np: ti.types.ndarray(), p_np: ti.types.ndarray(), q_np: ti.types.ndarray()):
        for i, j, k in ti.ndrange(*self.res):
            for n in ti.static(range(self.dim)):
                v_np[i, j, k, n]     = self.grid[s, i, j, k].v[n]
                v_tmp_np[i, j, k, n] = self.grid[s, i, j, k].v_tmp[n]
            for n in ti.static(range(self.q_dim)):
                q_np[i, j, k, n] = self.grid[s, i, j, k].q[n]
            div_np[i, j, k] = self.grid[s, i, j, k].div
            p_np[i, j, k]   = self.grid[s, i, j, k].p

    @ti.kernel
    def setframe(self, s: ti.i32, v_np: ti.types.ndarray(), v_tmp_np: ti.types.ndarray(), div_np: ti.types.ndarray(), p_np: ti.types.ndarray(), q_np: ti.types.ndarray()):
        for i, j, k in ti.ndrange(*self.res):
            for n in ti.static(range(self.dim)):
                self.grid[s, i, j, k].v[n] = v_np[i, j, k, n]
                self.grid[s, i, j, k].v_tmp[n] = v_tmp_np[i, j, k, n]
            for n in ti.static(range(self.q_dim)):
                self.grid[s, i, j, k].q[n] = q_np[i, j, k, n]
            self.grid[s, i, j, k].div = div_np[i, j, k]
            self.grid[s, i, j, k].p = p_np[i, j, k]

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            ckpt = {
                'v'     : self.v_np,
                'v_tmp' : self.v_tmp_np,
                'div'   : self.div_np,
                'p'     : self.p_np,
                'q'     : self.q_np,
            }
            self.readframe(0, self.v_np, self.v_tmp_np, self.div_np, self.p_np, self.q_np)
            return ckpt

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if not ckpt_name in self.ckpt_ram:
                if self.ckpt_dest == 'cpu':
                    device = 'cpu'
                elif self.ckpt_dest == 'gpu':
                    device = 'cuda'
                self.ckpt_ram[ckpt_name] = {
                    'v'     : torch.zeros((*self.res, self.dim), dtype=DTYPE_TC, device=device),
                    'v_tmp' : torch.zeros((*self.res, self.dim), dtype=DTYPE_TC, device=device),
                    'div'   : torch.zeros((*self.res,), dtype=DTYPE_TC, device=device),
                    'p'     : torch.zeros((*self.res,), dtype=DTYPE_TC, device=device),
                    'q'     : torch.zeros((*self.res, self.q_dim), dtype=DTYPE_TC, device=device),
                }
            self.readframe(
                0,
                self.ckpt_ram[ckpt_name]['v'],
                self.ckpt_ram[ckpt_name]['v_tmp'],
                self.ckpt_ram[ckpt_name]['div'],
                self.ckpt_ram[ckpt_name]['p'],
                self.ckpt_ram[ckpt_name]['q'],
            )

    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            assert ckpt is not None

        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]

        self.setframe(0, ckpt['v'], ckpt['v_tmp'], ckpt['div'], ckpt['p'], ckpt['q'])

    def get_state(self, s):
        state = {
            'v'     : np.zeros((*self.res, self.dim), dtype=DTYPE_NP),
            'v_tmp' : np.zeros((*self.res, self.dim), dtype=DTYPE_NP),
            'div'   : np.zeros((*self.res,), dtype=DTYPE_NP),
            'p'     : np.zeros((*self.res,), dtype=DTYPE_NP),
            'q'     : np.zeros((*self.res, self.q_dim), dtype=DTYPE_NP),
        }
        self.readframe(s, state['v'], state['v_tmp'], state['div'], state['p'], state['q'])
        return state

    def set_state(self, s, state):
        self.setframe(s, state['v'], state['v_tmp'], state['div'], state['p'], state['q'])


class TexPair:
    def __init__(self, cur, nxt):
        self.cur = cur
        self.nxt = nxt

    def swap(self):
        self.cur, self.nxt = self.nxt, self.cur

