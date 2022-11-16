import numpy as np
import taichi as ti
from scipy.spatial.transform import Rotation
from fluidlab.configs.macros import *


@ti.func
def qmul(q, r):
    terms = r.outer_product(q)
    w = terms[0, 0] - terms[1, 1] - terms[2, 2] - terms[3, 3]
    x = terms[0, 1] + terms[1, 0] - terms[2, 3] + terms[3, 2]
    y = terms[0, 2] + terms[1, 3] + terms[2, 0] - terms[3, 1]
    z = terms[0, 3] - terms[1, 2] + terms[2, 1] + terms[3, 0]
    out = ti.Vector([w, x, y, z])
    return out / ti.sqrt(out.dot(out)) # normalize it to prevent some unknown NaN problems.

@ti.func
def w2quat(axis_angle, dtype):
    w = axis_angle.norm(EPS)
    out = ti.Vector.zero(dt=dtype, n=4)

    v = (axis_angle/w) * ti.sin(w/2)
    out[0] = ti.cos(w/2)
    out[1] = v[0]
    out[2] = v[1]
    out[3] = v[2]

    return out

@ti.func
def inv_quat(quat):
    return ti.Vector([quat[0], -quat[1], -quat[2], -quat[3]]).normalized()

@ti.func
def inv_trans_ti(pos_A, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ (pos_A - trans_B_to_A)

@ti.func
def trans_ti(pos_B, trans_B_to_A, rot_B_to_A):
    return rot_B_to_A.inverse() @ pos_B + trans_B_to_A

def scale_to_T(scale):
    T_scale = np.eye(4, dtype=scale.dtype)
    T_scale[[0, 1, 2], [0, 1, 2]] = scale
    return T_scale

def trans_quat_to_T(trans=None, quat=None):
    if trans is not None:
        dtype = trans.dtype
    else:
        dtype = quat.dtype

    T = np.eye(4, dtype=dtype)
    if trans is not None:
        T[:3, 3] = trans
    if quat is not None:
        T[:3, :3] = Rotation.from_quat(xyzw_from_wxyz(quat)).as_matrix()
        
    return T

def transform_by_T_np(pos, T):
    if len(pos.shape) == 2:
        assert pos.shape[1] == 3
        new_pos = np.hstack([pos, np.ones_like(pos[:, :1])]).T
        new_pos = (T @ new_pos).T
        new_pos = new_pos[:, :3]

    elif len(pos.shape) == 1:
        assert pos.shape[0] == 3
        new_pos = np.append(pos, 1)
        new_pos = T @ new_pos
        new_pos = new_pos[:3]

    else:
        assert False

    return new_pos

def transform_by_trans_quat_np(pos, trans=None, quat=None):
    return transform_by_quat_np(pos, quat) + trans

def inv_transform_np(pos, T):
    T_inv = np.linalg.inv(T)
    return transform_np(pos, T_inv)

def transform_by_quat_np(v, quat):
    qvec = quat[1:]
    uv = np.cross(qvec, v)
    uuv = np.cross(qvec, uv)
    return v + 2 * (quat[0] * uv + uuv)

@ti.func
def normalize(v, eps=EPS):
    return v / (v.norm(eps))

@ti.func
def transform_by_quat_ti(v, quat):
    qvec = ti.Vector([quat[1], quat[2], quat[3]])
    uv = qvec.cross(v)
    uuv = qvec.cross(uv)
    return v + 2 * (quat[0] * uv + uuv)

@ti.func
def inv_transform_by_quat_ti(v, quat):
    return transform_by_quat_ti(v, inv_quat(quat))

@ti.func
def transform_by_T_ti(pos, T, dtype):
    new_pos = ti.Vector([pos[0], pos[1], pos[2], 1.0], dt=dtype)
    new_pos = T @ new_pos
    return new_pos[:3]

@ti.func
def inv_transform_by_T_ti(pos, T, dtype):
    T_inv = T.inverse()
    return transform_ti(pos, T_inv, dtype)

@ti.func
def transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos, quat) + trans

@ti.func
def inv_transform_by_trans_quat_ti(pos, trans, quat):
    return transform_by_quat_ti(pos - trans, inv_quat(quat))

def xyzw_to_wxyz(xyzw):
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]])

def xyzw_from_wxyz(wxyz):
    return np.array([wxyz[1], wxyz[2], wxyz[3], wxyz[0]])

def compute_camera_angle(camera_pos, camera_lookat):
    camera_dir = np.array(camera_lookat) - np.array(camera_pos)

    # rotation around vertical (y) axis
    angle_x = np.arctan2(-camera_dir[0], -camera_dir[2])

    # rotation w.r.t horizontal plane
    angle_y = np.arctan2(camera_dir[1], np.linalg.norm([camera_dir[0], camera_dir[2]]))
    
    angle_z = 0.0

    return np.array([angle_x, angle_y, angle_z])