import taichi as ti
import numpy as np
import pickle as pkl
import uuid
import os
import torch
from time import time
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *
from fluidlab.engine.boundaries import create_boundary

@ti.data_oriented
class MPMSimulator:
    def __init__(self, dim, quality, gravity, horizon, max_substeps_local, max_substeps_global, ckpt_dest):

        self.dim       = dim
        self.ckpt_dest = ckpt_dest
        self.sim_id    = str(uuid.uuid4())
        self.gravity   = ti.Vector(gravity)

        self.n_grid              = int(64 * quality)
        self.dx                  = 1 / self.n_grid
        self.inv_dx              = float(self.n_grid)
        self.dt                  = 2e-4
        self.p_vol               = (self.dx * 0.5) ** 2
        self.res                 = (self.n_grid,) * self.dim
        self.max_substeps_local  = max_substeps_local
        self.max_substeps_global = max_substeps_global
        self.horizon             = horizon
        self.n_substeps          = int(2e-3 / self.dt)
        self.max_steps_local     = int(self.max_substeps_local / self.n_substeps)

        assert self.n_substeps * self.horizon < self.max_substeps_global
        assert self.max_substeps_local % self.n_substeps == 0

        self.boundary      = None
        self.has_particles = False

    def setup_boundary(self, **kwargs):
        self.boundary = create_boundary(**kwargs)

    def build(self, agent, smoke_field, statics, particles):
        # default boundary
        if self.boundary is None:
            self.boundary = create_boundary()

        # statics
        self.n_statics = len(statics)
        self.statics = statics

        # particles and bodies
        if particles is not None:
            self.has_particles = True
            self.n_particles = len(particles['x'])
            self.setup_particle_fields()
            self.setup_grid_fields()
            self.setup_ckpt_vars()
            self.init_particles_and_bodies(particles)
        else:
            self.has_particles = False
            self.n_particles = 0

        # agent
        self.agent = agent

        # smoke
        self.smoke_field = smoke_field

        # misc
        self.cur_substep_global = 0
        self.disable_grad() # grad disabled by default

    def setup_particle_fields(self):
        # particle state
        particle_state = ti.types.struct(
            x     = ti.types.vector(self.dim, DTYPE_TI),  # position
            v     = ti.types.vector(self.dim, DTYPE_TI),  # velocity
            C     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # affine velocity field
            F     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # deformation gradient
            F_tmp = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # temp deformation gradient
            U     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            V     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
            S     = ti.types.matrix(self.dim, self.dim, DTYPE_TI),  # SVD
        )
        # particle state without gradient
        particle_state_ng = ti.types.struct(
            used = ti.i32,
        )
        # single frame particle state for rendering
        particle_state_render = ti.types.struct(
            x    = ti.types.vector(self.dim, ti.f32),
            used = ti.i32,
        )

        # particle info
        particle_info = ti.types.struct(
            mu      = DTYPE_TI,
            lam     = DTYPE_TI,
            mat     = ti.i32,
            mat_cls = ti.i32,
            body_id = ti.i32,
            mass    = DTYPE_TI,
        )

        # construct fields
        self.particles    = particle_state.field(shape=(self.max_substeps_local+1, self.n_particles), needs_grad=True, layout=ti.Layout.SOA)
        self.particles_ng = particle_state_ng.field(shape=(self.max_substeps_local+1, self.n_particles), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_render  = particle_state_render.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)
        self.particles_i  = particle_info.field(shape=(self.n_particles,), needs_grad=False, layout=ti.Layout.SOA)

    def setup_grid_fields(self):
        grid_cell_state = ti.types.struct(
            v_in  = ti.types.vector(self.dim, DTYPE_TI), # input momentum/velocity
            mass  = DTYPE_TI,                            # mass
            v_out = ti.types.vector(self.dim, DTYPE_TI), # output momentum/velocity
        )
        self.grid = grid_cell_state.field(shape=(self.max_substeps_local+1, *self.res), needs_grad=True, layout=ti.Layout.SOA)

    def setup_ckpt_vars(self):
        if self.ckpt_dest == 'disk':
            # placeholder np array from checkpointing
            self.x_np    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.v_np    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            self.C_np    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.F_np    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            self.used_np = np.zeros((self.n_particles,), dtype=np.int32)
        elif self.ckpt_dest == 'cpu' or 'gpu':
            self.ckpt_ram = dict()
        self.actions_buffer = []
        self.setup_ckpt_dir()

    def setup_ckpt_dir(self):
        self.ckpt_dir = os.path.join('/tmp', 'fluidlab', self.sim_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

    def init_particles_and_bodies(self, particles):
        x          = particles['x'].astype(DTYPE_NP)
        used       = particles['used'].astype(np.int32)
        mat        = particles['mat'].astype(np.int32)
        p_rho      = particles['rho'].astype(DTYPE_NP)
        body_id    = particles['body_id'].astype(np.int32)
        
        mu         = np.array([MU[mat_i] for mat_i in mat]).astype(DTYPE_NP)
        lam        = np.array([LAMDA[mat_i] for mat_i in mat]).astype(DTYPE_NP)
        mat_cls    = np.array([MAT_CLASS[mat_i] for mat_i in mat]).astype(np.int32)

        self.init_particles_kernel(x, mat, mat_cls, used, mu, lam, p_rho, body_id)
        self.init_bodies(mat_cls, body_id, particles['bodies'])

    @ti.kernel
    def init_particles_kernel(
            self,
            x       : ti.types.ndarray(),
            mat     : ti.types.ndarray(),
            mat_cls : ti.types.ndarray(),
            used    : ti.types.ndarray(),
            mu      : ti.types.ndarray(),
            lam     : ti.types.ndarray(),
            p_rho   : ti.types.ndarray(),
            body_id : ti.types.ndarray()
        ):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[0, i].x[j] = x[i, j]
            self.particles[0, i].v       = ti.Vector.zero(DTYPE_TI, self.dim)
            self.particles[0, i].F       = ti.Matrix.identity(DTYPE_TI, self.dim)
            self.particles[0, i].C       = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
            self.particles_ng[0, i].used = used[i]

            self.particles_i[i].mat     = mat[i]
            self.particles_i[i].mat_cls = mat_cls[i]
            self.particles_i[i].mu      = mu[i]
            self.particles_i[i].lam     = lam[i]
            self.particles_i[i].mass    = self.p_vol * p_rho[i]
            self.particles_i[i].body_id = body_id[i]

    def init_bodies(self, mat_cls, body_id, bodies):
        self.n_bodies = bodies['n']
        assert self.n_bodies == np.max(body_id) + 1

        # body state, for rigidity enforcement
        body_state = ti.types.struct(
            COM_t0 = ti.types.vector(self.dim, DTYPE_TI),
            COM_t1 = ti.types.vector(self.dim, DTYPE_TI),
            H      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            R      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            U      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            S      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
            V      = ti.types.matrix(self.dim, self.dim, DTYPE_TI),
        )
        # body info
        body_info = ti.types.struct(
            n_particles = ti.i32,
            mat_cls     = ti.i32,
        )
        self.bodies   = body_state.field(shape=(self.n_bodies,), needs_grad=True, layout=ti.Layout.SOA)
        self.bodies_i = body_info.field(shape=(self.n_bodies,), needs_grad=False, layout=ti.Layout.SOA)

        for i in range(self.n_bodies):
            self.bodies_i[i].n_particles = np.sum(body_id == i)
            self.bodies_i[i].mat_cls = mat_cls[body_id == i][0]

    def reset_grad(self):
        self.particles.grad.fill(0)
        self.grid.grad.fill(0)

    def enable_grad(self):
        '''
        If grad_enable == True, we do checkpointing when gpu memory is not enough for storing the whole episode.
        '''
        self.grad_enabled       = True
        self.cur_substep_global = 0

    def disable_grad(self):
        self.grad_enabled       = False
        self.cur_substep_global = 0

    # --------------------------------- MPM part -----------------------------------
    @ti.kernel
    def reset_grid_and_grad(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            self.grid[f, I].fill(0)
            self.grid.grad[f, I].fill(0)

    def f_global_to_f_local(self, f_global):
        f_local = f_global % self.max_substeps_local
        return f_local

    def f_local_to_s_local(self, f_local):
        f_local = f_local // self.n_substeps
        return f_local

    def f_global_to_s_local(self, f_global):
        f_local = self.f_global_to_f_local(f_global)
        s_local = self.f_local_to_s_local(f_local)
        return s_local

    def f_global_to_s_global(self, f_global):
        s_global = f_global // self.n_substeps
        return s_global

    @property
    def cur_substep_local(self):
        return self.f_global_to_f_local(self.cur_substep_global)

    @property
    def cur_step_local(self):
        return self.f_global_to_s_local(self.cur_substep_global)

    @property
    def cur_step_global(self):
        return self.f_global_to_s_global(self.cur_substep_global)

    @ti.kernel
    def compute_F_tmp(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles[f, p].F_tmp = (ti.Matrix.identity(DTYPE_TI, self.dim) + self.dt * self.particles[f, p].C) @ self.particles[f, p].F

    @ti.kernel
    def svd(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V = ti.svd(self.particles[f, p].F_tmp, DTYPE_TI)

    @ti.kernel
    def svd_grad(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                self.particles.grad[f, p].F_tmp += self.backward_svd(self.particles.grad[f, p].U, self.particles.grad[f, p].S, self.particles.grad[f, p].V, self.particles[f, p].U, self.particles[f, p].S, self.particles[f, p].V)

    @ti.func
    def backward_svd(self, grad_U, grad_S, grad_V, U, S, V):
        # https://github.com/pytorch/pytorch/blob/ab0a04dc9c8b84d4a03412f1c21a6c4a2cefd36c/tools/autograd/templates/Functions.cpp
        vt = V.transpose()
        ut = U.transpose()
        S_term = U @ grad_S @ vt

        s = ti.Vector.zero(DTYPE_TI, self.dim)
        if ti.static(self.dim==2):
            s = ti.Vector([S[0, 0], S[1, 1]]) ** 2
        else:
            s = ti.Vector([S[0, 0], S[1, 1], S[2, 2]]) ** 2
        F = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
        for i, j in ti.static(ti.ndrange(self.dim, self.dim)):
            if i == j:
                F[i, j] = 0
            else:
                F[i, j] = 1.0 / self.clamp(s[j] - s[i])
        u_term = U @ ((F * (ut @ grad_U - grad_U.transpose() @ U)) @ S) @ vt
        v_term = U @ (S @ ((F * (vt @ grad_V - grad_V.transpose() @ V)) @ vt))
        return u_term + v_term + S_term

    @ti.func
    def clamp(self, a):
        # remember that we don't support if return in taichi
        # stop the gradient ...
        if a>=0:
            a = ti.max(a, 1e-8)
        else:
            a = ti.min(a, -1e-8)
        return a

    @ti.kernel
    def advect_used(self, f: ti.i32):
        for p in range(self.n_particles):
            self.particles_ng[f+1, p].used = self.particles_ng[f, p].used

    @ti.kernel
    def process_unused_particles(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used == 0:
                self.particles[f+1, p].v = self.particles[f, p].v
                self.particles[f+1, p].x = self.particles[f, p].x
                self.particles[f+1, p].C = self.particles[f, p].C
                self.particles[f+1, p].F = self.particles[f, p].F

    def agent_act(self, f, is_none_action):
        if not is_none_action:
            self.agent.act(f, self.cur_substep_global)

    def agent_act_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.act_grad(f, self.cur_substep_global)


    @ti.func
    def stencil_range(self):
        return ti.ndrange(*((3, ) * self.dim))

    @ti.kernel
    def p2g(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
                fx   = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w    = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]

                J = self.particles[f, p].S.determinant()

                r = self.particles[f, p].U @ self.particles[f, p].V.transpose()
                stress = 2 * self.particles_i[p].mu * (self.particles[f, p].F_tmp - r) @ self.particles[f, p].F_tmp.transpose() + ti.Matrix.identity(DTYPE_TI, self.dim) * self.particles_i[p].lam * J * (J - 1)
                stress = (-self.dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * stress
                affine = stress + self.particles_i[p].mass * self.particles[f, p].C

                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = (offset.cast(DTYPE_TI) - fx) * self.dx
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]

                    self.grid[f, base + offset].v_in += weight * (self.particles_i[p].mass * self.particles[f, p].v + affine @ dpos)
                    self.grid[f, base + offset].mass += weight * self.particles_i[p].mass

                # update deformation gradient based on material class
                F_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)

                if self.particles_i[p].mat_cls == MAT_LIQUID:
                    F_new = ti.Matrix.identity(DTYPE_TI, self.dim) * ti.pow(J, 1.0/self.dim)

                elif self.particles_i[p].mat_cls == MAT_ELASTIC:
                    F_new = self.particles[f, p].F_tmp

                elif self.particles_i[p].mat_cls == MAT_RIGID:
                    F_new = self.particles[f, p].F_tmp

                elif self.particles_i[p].mat_cls == MAT_PLASTO_ELASTIC:
                    S_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                    for d in ti.static(range(self.dim)):
                        S_new[d, d] = min(max(self.particles[f, p].S[d, d], 1 - 2e-3), 1 + 3e-3)
                    F_new = self.particles[f, p].U @ S_new @ self.particles[f, p].V.transpose()
                elif self.particles_i[p].mat_cls == MAT_PLASTO_ELASTIC_DEMO:
                    S_new = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                    for d in ti.static(range(self.dim)):
                        S_new[d, d] = min(max(self.particles[f, p].S[d, d], 1 - 2e-3), 1 + 3e-3)
                    F_new = self.particles[f, p].U @ S_new @ self.particles[f, p].V.transpose()

                self.particles[f+1, p].F = F_new

    @ti.kernel
    def grid_op(self, f: ti.i32):
        for I in ti.grouped(ti.ndrange(*self.res)):
            if self.grid[f, I].mass > EPS:
                v_out = (1 / self.grid[f, I].mass) * self.grid[f, I].v_in  # Momentum to velocity
                v_out += self.dt * self.gravity # gravity

                # collide with statics
                if ti.static(self.n_statics>0):
                    for i in ti.static(range(self.n_statics)):
                        v_out = self.statics[i].collide(I*self.dx, v_out)

                # collide with agent
                if ti.static(self.agent is not None):
                    if ti.static(self.agent.collide_type in ['grid', 'both']):
                        v_out = self.agent.collide(f, I*self.dx, v_out, self.dt)

                # impose boundary
                _, self.grid[f, I].v_out = self.boundary.impose_x_v(I*self.dx, v_out)

    @ti.kernel
    def g2p(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                base = (self.particles[f, p].x * self.inv_dx - 0.5).cast(int)
                fx = self.particles[f, p].x * self.inv_dx - base.cast(DTYPE_TI)
                w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
                new_v = ti.Vector.zero(DTYPE_TI, self.dim)
                new_C = ti.Matrix.zero(DTYPE_TI, self.dim, self.dim)
                for offset in ti.static(ti.grouped(self.stencil_range())):
                    dpos = offset.cast(DTYPE_TI) - fx
                    g_v = self.grid[f, base + offset].v_out
                    weight = ti.cast(1.0, DTYPE_TI)
                    for d in ti.static(range(self.dim)):
                        weight *= w[offset[d]][d]
                    new_v += weight * g_v
                    new_C += 4 * self.inv_dx * weight * g_v.outer_product(dpos)

                # collide with agent
                if ti.static(self.agent is not None):
                    if ti.static(self.agent.collide_type in ['particle', 'both']):
                        new_x_tmp = self.particles[f, p].x + self.dt * new_v
                        new_v = self.agent.collide(f, new_x_tmp, new_v, self.dt)

                # advect to next frame    
                self.particles[f+1, p].v = new_v
                self.particles[f+1, p].C = new_C

    def advect(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)
        self.advect_kernel(f)

    def advect_grad(self, f):
        self.reset_bodies_and_grad()
        self.compute_COM(f)
        self.compute_H(f)
        self.compute_H_svd(f)
        self.compute_R(f)

        self.advect_kernel.grad(f)
        self.compute_R.grad(f)
        self.compute_H_svd_grad(f)
        self.compute_H.grad(f)
        self.compute_COM.grad(f)

    @ti.kernel
    def reset_bodies_and_grad(self):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].fill(0)
                self.bodies.grad[body_id].fill(0)

    @ti.kernel
    def compute_COM(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].COM_t0 += self.particles[f, p].x / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)
                self.bodies[body_id].COM_t1 += (self.particles[f, p].x + self.dt * self.particles[f+1, p].v) / ti.cast(self.bodies_i[body_id].n_particles, DTYPE_TI)

    @ti.kernel
    def compute_H(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used and self.particles_i[p].mat_cls == MAT_RIGID:
                body_id = self.particles_i[p].body_id
                self.bodies[body_id].H[0, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[0, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[0, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[0] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[1, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[1, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[1, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[1] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]
                self.bodies[body_id].H[2, 0] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[0]
                self.bodies[body_id].H[2, 1] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[1]
                self.bodies[body_id].H[2, 2] += (self.particles[f, p].x - self.bodies[body_id].COM_t0)[2] * (self.particles[f, p].x + self.dt * self.particles[f+1, p].v - self.bodies[body_id].COM_t1)[2]

    @ti.kernel
    def compute_H_svd(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].U, self.bodies[body_id].S, self.bodies[body_id].V = ti.svd(self.bodies[body_id].H, DTYPE_TI)

    @ti.kernel
    def compute_H_svd_grad(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies.grad[body_id].H = self.backward_svd(self.bodies.grad[body_id].U, self.bodies.grad[body_id].S, self.bodies.grad[body_id].V, self.bodies[body_id].U, self.bodies[body_id].S, self.bodies[body_id].V)

    @ti.kernel
    def compute_R(self, f: ti.i32):
        for body_id in range(self.n_bodies):
            if self.bodies_i[body_id].mat_cls == MAT_RIGID:
                self.bodies[body_id].R = self.bodies[body_id].V @ self.bodies[body_id].U.transpose()

    @ti.kernel
    def advect_kernel(self, f: ti.i32):
        for p in range(self.n_particles):
            if self.particles_ng[f, p].used:
                if self.particles_i[p].mat_cls == MAT_RIGID: # rigid objects
                    body_id = self.particles_i[p].body_id
                    self.particles[f+1, p].x = self.bodies[body_id].R @ (self.particles[f, p].x - self.bodies[body_id].COM_t0) + self.bodies[body_id].COM_t1
                else: # other particles
                    self.particles[f+1, p].x = self.particles[f, p].x + self.dt * self.particles[f+1, p].v

    def agent_move(self, f, is_none_action):
        if not is_none_action:
            self.agent.move(f)

    def agent_move_grad(self, f, is_none_action):
        if not is_none_action:
            self.agent.move_grad(f)

    def substep(self, f, is_none_action):
        if self.has_particles:
            self.reset_grid_and_grad(f)
            self.advect_used(f)
            self.process_unused_particles(f)

        self.agent_act(f, is_none_action)

        if self.has_particles:
            self.compute_F_tmp(f)
            self.svd(f)
            self.p2g(f)

        self.agent_move(f, is_none_action)

        if self.has_particles:
            self.grid_op(f)
            self.g2p(f)
            self.advect(f)

    def substep_grad(self, f, is_none_action):
        if self.has_particles:
            self.advect_grad(f)
            self.g2p.grad(f)
            self.grid_op.grad(f)

        self.agent_move_grad(f, is_none_action)

        if self.has_particles:
            self.p2g.grad(f)
            self.svd_grad(f)
            self.compute_F_tmp.grad(f)

        self.agent_act_grad(f, is_none_action)

        if self.has_particles:
            self.process_unused_particles.grad(f)
            self.advect_used.grad(f)

    # ------------------------------------ io -------------------------------------#
    @ti.kernel
    def readframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]
                v[i, j] = self.particles[f, i].v[j]
                for k in ti.static(range(self.dim)):
                    C[i, j, k] = self.particles[f, i].C[j, k]
                    F[i, j, k] = self.particles[f, i].F[j, k]
            used[i] = self.particles_ng[f, i].used

    @ti.kernel
    def setframe(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), C: ti.types.ndarray(), F: ti.types.ndarray(), used: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[f, i].x[j] = x[i, j]
                self.particles[f, i].v[j] = v[i, j]
                for k in ti.static(range(self.dim)):
                    self.particles[f, i].C[j, k] = C[i, j, k]
                    self.particles[f, i].F[j, k] = F[i, j, k]
            self.particles_ng[f, i].used = used[i]

    @ti.kernel
    def set_x(self, f:ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles[f, i].x[j] = x[i, j]

    @ti.kernel
    def set_used(self, f:ti.i32, used: ti.types.ndarray()):
        for i in range(self.n_particles):
            self.particles_ng[f, i].used = used[i]

    @ti.kernel
    def copy_frame(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles[target, i].x = self.particles[source, i].x
            self.particles[target, i].v = self.particles[source, i].v
            self.particles[target, i].F = self.particles[source, i].F
            self.particles[target, i].C = self.particles[source, i].C
            self.particles_ng[target, i].used = self.particles_ng[source, i].used

    @ti.kernel
    def copy_grad(self, source: ti.i32, target: ti.i32):
        for i in range(self.n_particles):
            self.particles.grad[target, i].x = self.particles.grad[source, i].x
            self.particles.grad[target, i].v = self.particles.grad[source, i].v
            self.particles.grad[target, i].F = self.particles.grad[source, i].F
            self.particles.grad[target, i].C = self.particles.grad[source, i].C
            self.particles_ng[target, i].used = self.particles_ng[source, i].used

    @ti.kernel
    def reset_grad_till_frame(self, f: ti.i32):
        for i, j in ti.ndrange(f, self.n_particles):
            self.particles.grad[i, j].fill(0)

    def get_state(self):
        f = self.cur_substep_local
        s = self.cur_step_local

        state = {}

        if self.has_particles:
            state['x']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['C']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['F']    = np.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.n_particles,), dtype=np.int32)
            self.readframe(f, state['x'], state['v'], state['C'], state['F'], state['used'])

        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)

        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)

        return state

    def set_state(self, f_global, state):
        f = self.f_global_to_f_local(f_global)
        s = self.f_global_to_s_local(f_global)

        if self.has_particles:
            self.setframe(f, state['x'], state['v'], state['C'], state['F'], state['used'])

        if self.agent is not None:
            self.agent.set_state(f, state['agent'])

        if self.smoke_field is not None:
            self.smoke_field.set_state(s, state['smoke_field'])

    @ti.kernel
    def get_x_kernel(self, f: ti.i32, x: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]

    @ti.kernel
    def get_used_kernel(self, f: ti.i32, used: ti.types.ndarray()):
        for i in range(self.n_particles):
            used[i] = self.particles_ng[f, i].used

    def get_x(self, f=None):
        if f is None:
            f = self.cur_substep_local

        x = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_x_kernel(f, x)
        return x

    def get_used(self, f=None):
        if f is None:
            f = self.cur_substep_local

        used = np.zeros((self.n_particles), dtype=np.int32)
        if self.has_particles:
            self.get_used_kernel(f, used)
        return used

    @ti.kernel
    def get_state_RL_kernel(self, f:ti.i32, x: ti.types.ndarray(), v: ti.types.ndarray(), used: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                x[i, j] = self.particles[f, i].x[j]
                v[i, j] = self.particles[f, i].v[j]
            used[i] = self.particles_ng[f, i].used

    def get_state_RL(self):
        f = self.cur_substep_local
        s = self.cur_step_local
        state = {}
        if self.has_particles:
            state['x']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['v']    = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
            state['used'] = np.zeros((self.n_particles,), dtype=np.int32)
            self.get_state_RL_kernel(f, state['x'], state['v'], state['used'])
        if self.agent is not None:
            state['agent'] = self.agent.get_state(f)
        if self.smoke_field is not None:
            state['smoke_field'] = self.smoke_field.get_state(s)
        return state

    @ti.kernel
    def get_state_render_kernel(self, f: ti.i32):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                self.particles_render[i].x[j] = ti.cast(self.particles[f, i].x[j], ti.f32)
            self.particles_render[i].used = ti.cast(self.particles_ng[f, i].used, ti.i32)

    def get_state_render(self, f):
        self.get_state_render_kernel(f)
        return self.particles_render

    @ti.kernel
    def get_v_kernel(self, f: ti.i32, v: ti.types.ndarray()):
        for i in range(self.n_particles):
            for j in ti.static(range(self.dim)):
                v[i, j] = self.particles[f, i].v[j]

    def get_v(self, f):
        v = np.zeros((self.n_particles, self.dim), dtype=DTYPE_NP)
        if self.has_particles:
            self.get_v_kernel(f, v)
        return v

    def step(self, action=None):
        if self.grad_enabled:
            if self.cur_substep_local == 0:
                self.actions_buffer = []

        self.step_(action)

        if self.grad_enabled:
            self.actions_buffer.append(action)

        if self.cur_substep_local == 0:
            self.memory_to_cache()


    def step_(self, action=None):
        is_none_action = action is None
        if not is_none_action:
            self.agent.set_action(
                s=self.cur_step_local,
                s_global=self.cur_step_global,
                n_substeps=self.n_substeps,
                action=action
            )

        # smoke simulates at step level, not substep
        if self.smoke_field is not None:
            self.smoke_field.step(s=self.cur_step_local, f=self.cur_substep_local)

        for i in range(0, self.n_substeps):
            self.substep(self.cur_substep_local, is_none_action)
            self.cur_substep_global += 1

        assert self.cur_substep_global <= self.max_substeps_global

    def step_grad(self, action=None):     
        if self.cur_substep_local == 0:
            self.memory_from_cache()

        is_none_action = action is None

        for i in range(self.n_substeps-1, -1, -1):
            self.cur_substep_global -= 1
            self.substep_grad(self.cur_substep_local, is_none_action)

        # smoke simulates at step level, not substep
        if self.smoke_field is not None:
            self.smoke_field.step_grad(s=self.cur_step_local, f=self.cur_substep_local)

        if not is_none_action:
            self.agent.set_action_grad(
                s=self.cur_substep_local//self.n_substeps,
                s_global=self.cur_substep_global//self.n_substeps, 
                n_substeps=self.n_substeps,
                action=action
            )

    def memory_to_cache(self):
        if self.grad_enabled:
            ckpt_start_step = self.cur_substep_global - self.max_substeps_local
            ckpt_end_step = self.cur_substep_global - 1
            ckpt_name = f'{ckpt_start_step:06d}'

            if self.ckpt_dest == 'disk':
                ckpt = {}
                if self.has_particles:
                    self.readframe(0, self.x_np, self.v_np, self.C_np, self.F_np, self.used_np)
                    ckpt['x']       = self.x_np
                    ckpt['v']       = self.v_np
                    ckpt['C']       = self.C_np
                    ckpt['F']       = self.F_np
                    ckpt['used']    = self.used_np
                    ckpt['actions'] = self.actions_buffer

                if self.smoke_field is not None:
                    ckpt['smoke_field'] = self.smoke_field.get_ckpt()

                if self.agent is not None:
                    ckpt['agent'] = self.agent.get_ckpt()

                # save to /tmp
                ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_name}.pkl')
                if os.path.exists(ckpt_file):
                    os.remove(ckpt_file)
                pkl.dump(ckpt, open(ckpt_file, 'wb'))

            elif self.ckpt_dest in ['cpu', 'gpu']:
                if not ckpt_name in self.ckpt_ram:
                    self.ckpt_ram[ckpt_name] = {}
                    
                    if self.ckpt_dest == 'cpu':
                        device = 'cpu'
                    elif self.ckpt_dest == 'gpu':
                        device = 'cuda'
                    if self.has_particles:
                        self.ckpt_ram[ckpt_name]['x']    = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['v']    = torch.zeros((self.n_particles, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['C']    = torch.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['F']    = torch.zeros((self.n_particles, self.dim, self.dim), dtype=DTYPE_TC, device=device)
                        self.ckpt_ram[ckpt_name]['used'] = torch.zeros((self.n_particles,), dtype=torch.int32, device=device)

                if self.has_particles:
                    self.readframe(
                        0,
                        self.ckpt_ram[ckpt_name]['x'],
                        self.ckpt_ram[ckpt_name]['v'],
                        self.ckpt_ram[ckpt_name]['C'],
                        self.ckpt_ram[ckpt_name]['F'],
                        self.ckpt_ram[ckpt_name]['used'],
                    )

                self.ckpt_ram[ckpt_name]['actions'] = list(self.actions_buffer)

                if self.smoke_field is not None:
                    self.smoke_field.get_ckpt(ckpt_name)

                if self.agent is not None:
                    self.agent.get_ckpt(ckpt_name)

            else:
                assert False

            # print(f'[Forward] Cached step {ckpt_start_step} to {ckpt_end_step}. {t2-t1:.2f}s {t3-t2:.2f}s')

        # restart from frame 0 in memory
        if self.has_particles:
            self.copy_frame(self.max_substeps_local, 0)

        if self.smoke_field is not None:
            self.smoke_field.copy_frame(self.max_steps_local, 0)

        if self.agent is not None:
            self.agent.copy_frame(self.max_substeps_local, 0)

        # print(f'[Forward] Memory refreshed. Now starts from global step {self.cur_substep_global}.')

    def memory_from_cache(self):
        assert self.grad_enabled
        if self.has_particles:
            self.copy_frame(0, self.max_substeps_local)
            self.copy_grad(0, self.max_substeps_local)
            self.reset_grad_till_frame(self.max_substeps_local)
            
        if self.smoke_field is not None:
            self.smoke_field.copy_frame(0, self.max_steps_local)
            self.smoke_field.copy_grad(0, self.max_steps_local)
            self.smoke_field.reset_grad_till_frame(self.max_steps_local)

        if self.agent is not None:
            self.agent.copy_frame(0, self.max_substeps_local)
            self.agent.copy_grad(0, self.max_substeps_local)
            self.agent.reset_grad_till_frame(self.max_substeps_local)

        ckpt_start_step = self.cur_substep_global - self.max_substeps_local
        ckpt_end_step = self.cur_substep_global - 1
        ckpt_name = f'{ckpt_start_step:06d}'

        if self.ckpt_dest == 'disk':
            ckpt_file = os.path.join(self.ckpt_dir, f'{ckpt_start_step:06d}.pkl')
            assert os.path.exists(ckpt_file)
            ckpt = pkl.load(open(ckpt_file, 'rb'))

            if self.has_particles:
                self.setframe(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'], ckpt['used'])


            if self.smoke_field is not None:
                self.smoke_field.set_ckpt(ckpt=ckpt['smoke_field'])

            if self.agent is not None:
                self.agent.set_ckpt(ckpt=ckpt['agent'])

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if self.has_particles:
                ckpt = self.ckpt_ram[ckpt_name]
                self.setframe(0, ckpt['x'], ckpt['v'], ckpt['C'], ckpt['F'], ckpt['used'])

            if self.smoke_field is not None:
                self.smoke_field.set_ckpt(ckpt_name=ckpt_name)

            if self.agent is not None:
                self.agent.set_ckpt(ckpt_name=ckpt_name)

        else:
            assert False

        # now that we loaded the first frame, we do a forward pass to fill up the rest 
        self.cur_substep_global = ckpt_start_step
        for action in ckpt['actions']:
            self.step_(action)

        # print(f'[Backward] Loading step {ckpt_start_step} to {ckpt_end_step} from cache. {t2-t1:.2f}s {t3-t2:.2f}s {t4-t3:.2f}s')
        # print(f'[Backward] Memory reloaded. Now starts from global step {ckpt_start_step}.')
