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
class AirCon(Effector):
    def __init__(
        self,
        inject_v=(-0.3, 0.0, 1.0),
        **kwargs
    ):
        super(AirCon, self).__init__(**kwargs)

        # strength and radius
        self.s = ti.field(dtype=DTYPE_TI, shape=(self.max_substeps_local+1,), needs_grad=True)
        self.r = ti.field(dtype=DTYPE_TI, shape=(self.max_substeps_local+1,), needs_grad=True)

        self.has_dynamics = False
        self.mesh = None
        self.inject_v = ti.Vector(eval_str(inject_v))

    def setup_mesh(self, **kwargs):
        self.mesh = Dynamic(
            container=self,
            has_dynamics=self.has_dynamics,
            **kwargs
        )

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
            self.s_np = np.zeros((), dtype=DTYPE_NP)
            self.r_np = np.zeros((), dtype=DTYPE_NP)
        elif self.ckpt_dest in ['cpu', 'gpu']:
            self.ckpt_ram = dict()

    def reset_grad(self):
        self.pos.grad.fill(0)
        self.quat.grad.fill(0)
        self.v.grad.fill(0)
        self.w.grad.fill(0)
        self.s.grad.fill(0)
        self.r.grad.fill(0)
        self.action_buffer.grad.fill(0)
        self.action_buffer_p.grad.fill(0)

    def move(self, f):
        self.move_kernel(f)
        self.update_latest_pos(f)
        self.update_mesh_pose(f)
        
    def update_mesh_pose(self, f):
        # For visualization only. No need to compute grad.
        if self.mesh is not None:
            self.mesh.update_vertices(f)

    @ti.kernel
    def get_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(), w_np: ti.types.ndarray(), s_np: ti.types.ndarray(), r_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            pos_np[i] = self.pos[0][i]
            v_np[i] = self.v[0][i]
            w_np[i] = self.w[0][i]

        for i in ti.static(range(4)):
            quat_np[i] = self.quat[0][i]

        s_np[()] = self.s[0]
        r_np[()] = self.r[0]

    @ti.kernel
    def set_ckpt_kernel(self, pos_np: ti.types.ndarray(), quat_np: ti.types.ndarray(), v_np: ti.types.ndarray(), w_np: ti.types.ndarray(), s_np: ti.types.ndarray(), r_np: ti.types.ndarray()):
        for i in ti.static(range(3)):
            self.pos[0][i] = pos_np[i]
            self.v[0][i] = v_np[i]
            self.w[0][i] = w_np[i]

        for i in ti.static(range(4)):
            self.quat[0][i] = quat_np[i]

        self.s[0] = s_np[()]
        self.r[0] = r_np[()]

    def get_ckpt(self, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            ckpt = {
                'pos': self.pos_np,
                'quat': self.quat_np,
                'v': self.v_np,
                'w': self.w_np,
                's': self.s_np,
                'r': self.r_np,
            }
            self.get_ckpt_kernel(self.pos_np, self.quat_np, self.v_np, self.w_np, self.s_np, self.r_np)
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
                    's': torch.zeros((), dtype=DTYPE_TC, device=device),
                    'r': torch.zeros((), dtype=DTYPE_TC, device=device),
                }
            self.get_ckpt_kernel(
                self.ckpt_ram[ckpt_name]['pos'],
                self.ckpt_ram[ckpt_name]['quat'],
                self.ckpt_ram[ckpt_name]['v'],
                self.ckpt_ram[ckpt_name]['w'],
                self.ckpt_ram[ckpt_name]['s'],
                self.ckpt_ram[ckpt_name]['r'],
            )


    def set_ckpt(self, ckpt=None, ckpt_name=None):
        if self.ckpt_dest == 'disk':
            assert ckpt is not None

        elif self.ckpt_dest in ['cpu', 'gpu']:
            ckpt = self.ckpt_ram[ckpt_name]

        self.set_ckpt_kernel(ckpt['pos'], ckpt['quat'], ckpt['v'], ckpt['w'], ckpt['s'], ckpt['r'])

    # state set and copy ...
    @ti.func
    def copy_frame(self, source, target):
        self.pos[target] = self.pos[source]
        self.quat[target] = self.quat[source]
        self.v[target] = self.v[source]
        self.w[target] = self.w[source]
        self.s[target] = self.s[source]
        self.r[target] = self.r[source]

    @ti.func
    def copy_grad(self, source, target):
        self.pos.grad[target] = self.pos.grad[source]
        self.quat.grad[target] = self.quat.grad[source]
        self.v.grad[target] = self.v.grad[source]
        self.w.grad[target] = self.w.grad[source]
        self.s.grad[target] = self.s.grad[source]
        self.r.grad[target] = self.r.grad[source]

    @ti.func
    def reset_grad_till_frame(self, f):
        for i in range(f):
            self.pos.grad[i].fill(0)
            self.quat.grad[i].fill(0)
            self.v.grad[i].fill(0)
            self.w.grad[i].fill(0)
            self.s.grad[i] = 0
            self.r.grad[i] = 0

    @ti.kernel
    def get_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            controller[j] = self.pos[f][j]
        for j in ti.static(range(4)):
            controller[j+3] = self.quat[f][j]
        controller[7] = self.s[f]
        controller[8] = self.r[f]

    @ti.kernel
    def set_state_kernel(self, f: ti.i32, controller: ti.types.ndarray()):
        for j in ti.static(range(3)):
            self.pos[f][j] = controller[j]
        for j in ti.static(range(4)):
            self.quat[f][j] = controller[j+3]
        self.s[f] = controller[7]
        self.r[f] = controller[8]

    def get_state(self, f):
        out = np.zeros((9), dtype=DTYPE_NP)
        self.get_state_kernel(f, out)
        return out

    def set_state(self, f, state):
        ss = self.get_state(f)
        ss[:len(state)] = state
        self.set_state_kernel(f, ss)

    @ti.kernel
    def set_velocity(self, s: ti.i32, s_global: ti.i32, n_substeps: ti.i32):
        for j in range(s*n_substeps, (s+1)*n_substeps):
            n_substeps_f = ti.cast(n_substeps, DTYPE_TI)
            for k in ti.static(range(3)):
                self.v[j][k] = self.action_buffer[s_global][k] * self.action_scale[None][k]/n_substeps_f

            if ti.static(self.action_dim>3):
                for k in ti.static(range(3)):
                    self.w[j][k] = self.action_buffer[s_global][k+3] * self.action_scale[None][k+3]/n_substeps_f

            if ti.static(self.action_dim>6):
                self.s[j] = self.action_buffer[s_global][6] * self.action_scale[None][6]
                self.r[j] = self.action_buffer[s_global][7] * self.action_scale[None][7]
