import taichi as ti
import numpy as np
import yaml
import torch
from .effector import Effector
from fluidlab.fluidengine.meshes import Dynamic
import fluidlab.utils.geom as geom_utils
from fluidlab.utils.misc import *
from fluidlab.configs.macros import *

@ti.data_oriented
class Injector(Effector):
    def __init__(
        self,
        radius=1.0,
        flux=1,
        inject_v=(0.0, 0.0, 0.0),
        inject_p=(0.0, 0.0, 0.0),
        randomize_inject_v=False,
        locally_random=False,
        **kwargs
    ):
        super(Injector, self).__init__(**kwargs)

        self.radius = radius
        self.n_particles = flux
        self.locally_random = locally_random
        self.randomize_inject_v = randomize_inject_v
        self.act_id = ti.field(dtype=ti.i32, shape=(self.max_substeps_local+1,), needs_grad=False)
        self.inject_v = ti.Vector(eval_str(inject_v))
        self.inject_p = ti.Vector(eval_str(inject_p))
        self.has_dynamics = False
        self.mesh = None

        self.init_random_vector()

    def setup_mesh(self, **kwargs):
        self.mesh = Dynamic(
            container=self,
            has_dynamics=self.has_dynamics,
            **kwargs
        )

    def init_ckpt(self):
        if self.ckpt_dest == 'disk':
            self.pos_np = np.zeros((3), dtype=DTYPE_NP)
            self.quat_np = np.zeros((4), dtype=DTYPE_NP)
            self.v_np = np.zeros((3), dtype=DTYPE_NP)
            self.w_np = np.zeros((3), dtype=DTYPE_NP)
            self.act_id_np = np.zeros((), dtype=np.int32)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = dict()

    def init_random_vector(self):
        if self.locally_random:
            random_length = self.max_substeps_local
        else:
            random_length = self.max_substeps_global
        self.random_vector = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=(random_length, self.n_particles))
        self.random_vector.from_numpy(np.random.uniform(size=(random_length, self.n_particles, self.dim)).astype(DTYPE_NP))

    def set_act_range(self, used):
        act_range = np.where(used==False)[0]

        self.act_range = ti.field(dtype=ti.i32, shape=(len(act_range),))
        self.act_range.from_numpy(act_range.astype(np.int32))

        self.act_id[0] = act_range[0]

    def move(self, f):
        self.move_kernel(f)
        self.update_latest_pos(f)
        self.update_mesh_pose(f)
        
    def update_mesh_pose(self, f):
        # For visualization only. No need to compute grad.
        if self.mesh is not None:
            self.mesh.update_vertices(f)

    @ti.func
    def act(self, f, f_global, used, x, v):
        for i in ti.static(range(self.n_particles)):
            particle_id = self.act_range[self.act_id[f]+i]

            # compute inject position
            random_vector = ti.Vector([0.0, 0.0, 0.0], dt=DTYPE_TI)
            if ti.static(self.locally_random):
                random_vector = self.random_vector[f, i]
            else:
                random_vector = self.random_vector[f_global, i]
            offset = (random_vector * 2 - 1) * ti.Vector([self.radius for _ in range(self.dim)])
            inject_p = geom_utils.transform_by_quat_ti(self.inject_p, self.quat[f])
            x[f+1, particle_id] = offset + self.pos[f] + inject_p

            # compute inject velocity
            inject_v = geom_utils.transform_by_quat_ti(self.inject_v, self.quat[f])
            if ti.static(self.randomize_inject_v):
                v[f+1, particle_id] = inject_v + (random_vector * 2 - 1) * self.inject_v.norm() * 2.0
            else:
                v[f+1, particle_id] = inject_v
                
            # update used
            used[f+1, particle_id] = 1

        self.act_id[f+1] = self.act_id[f] + self.n_particles

    @ti.kernel
    def get_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(), w_np: ti.types.ndarray(), act_id_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            pos_np[i] = self.pos[0][i]
            v_np[i] = self.v[0][i]
            w_np[i] = self.w[0][i]

        for i in ti.static(range(4)):
            quat_np[i] = self.quat[0][i]

        act_id_np[()] = self.act_id[0]

    @ti.kernel
    def set_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(), w_np: ti.types.ndarray(), act_id_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.pos[0][i] = pos_np[i]
            self.v[0][i] = v_np[i]
            self.w[0][i] = w_np[i]

        for i in ti.static(range(4)):
            self.quat[0][i] = quat_np[i]

        self.act_id[0] = act_id_np[()]

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            ckpt = {
                'pos': self.pos_np,
                'quat': self.quat_np,
                'v': self.v_np,
                'w': self.w_np,
                'act_id': self.act_id_np,
            }
            self.get_ckpt_kernel(self.pos_np, self.quat_np, self.v_np, self.w_np, self.act_id_np)
            return ckpt

        elif self.ckpt_dest in ['cpu', 'gpu']:
            if not ckpt_name in self.ckpt_ram:
                if self.ckpt_dest == 'cpu':
                    device = 'cpu'
                elif self.ckpt_dest == 'gpu':
                    device = 'cuda'
                self.ckpt_ram[ckpt_name] = {
                    'pos': torch.zeros((3), dtype=DTYPE_TC, device=device),
                    'quat': torch.zeros((4), dtype=DTYPE_TC, device=device),
                    'v': torch.zeros((3), dtype=DTYPE_TC, device=device),
                    'w': torch.zeros((3), dtype=DTYPE_TC, device=device),
                    'act_id': torch.zeros((), dtype=torch.int32, device=device),
                }
            self.get_ckpt_kernel(
                self.ckpt_ram[ckpt_name]['pos'],
                self.ckpt_ram[ckpt_name]['quat'],
                self.ckpt_ram[ckpt_name]['v'],
                self.ckpt_ram[ckpt_name]['w'],
                self.ckpt_ram[ckpt_name]['act_id']
            )

    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            assert ckpt is not None

        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]

        self.set_ckpt_kernel(ckpt['pos'], ckpt['quat'], ckpt['v'], ckpt['w'], ckpt['act_id'])

    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.pos[target] = self.pos[source]
        self.quat[target] = self.quat[source]
        self.v[target] = self.v[source]
        self.w[target] = self.w[source]
        self.act_id[target] = self.act_id[source]

    @ti.func
    def copy_grad(self, source, target):
        self.pos.grad[target] = self.pos.grad[source]
        self.quat.grad[target] = self.quat.grad[source]
        self.v.grad[target] = self.v.grad[source]
        self.w.grad[target] = self.w.grad[source]

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            controller[j] = self.pos[f][j]
        for j in ti.static(range(4)):
            controller[j+self.dim] = self.quat[f][j]
        controller[7] = ti.cast(self.act_id[f], DTYPE_TI)

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            self.pos[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.quat[f][j] = controller[j+self.dim]
        self.act_id[f] = ti.cast(controller[7], ti.i32)

    def get_state(self, f):
        out = np.zeros((8), dtype=DTYPE_NP)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)


class BallInjector(Injector):
    def __init__(self, **kwargs):
        super(BallInjector, self).__init__(**kwargs)

    def init_random_vector(self):
        # generate random vector
        if self.locally_random:
            random_length = self.max_substeps_local
        else:
            random_length = self.max_substeps_global
        self.random_vector_np = []
        n_generated = 0
        while True:
            rand_pos = np.random.uniform(high=self.radius, low=-self.radius, size=(self.n_particles*random_length, 3))
            in_ball = np.linalg.norm(rand_pos, axis=1) <= self.radius
            rand_pos = rand_pos[in_ball]
            n_generated += rand_pos.shape[0]
            self.random_vector_np.append(rand_pos)
            if n_generated >= self.n_particles*random_length:
                break
        self.random_vector_np = np.concatenate(self.random_vector_np)[:self.n_particles*random_length].reshape([random_length, self.n_particles, 3])
        self.random_vector = ti.Vector.field(self.dim, dtype=DTYPE_TI, shape=(random_length, self.n_particles))
        self.random_vector.from_numpy(self.random_vector_np.astype(DTYPE_NP))

    @ti.func
    def act(self, f, f_global, used, x, v):
        for i in ti.static(range(self.n_particles)):
            particle_id = self.act_range[self.act_id[f]+i]

            if ti.static(self.locally_random):
                offset = self.random_vector[f, i]
                x[f+1, particle_id] = offset + self.pos[f]
            else:
                offset = self.random_vector[f_global, i]
                x[f+1, particle_id] = offset + self.pos[f]

            v[f+1, particle_id] = self.inject_v
            used[f+1, particle_id] = 1


        self.act_id[f+1] = self.act_id[f] + self.n_particles
