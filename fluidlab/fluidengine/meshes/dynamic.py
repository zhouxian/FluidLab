import os
import copy
import pickle as pkl
import trimesh
import taichi as ti
import numpy as np
from .mesh import Mesh
import fluidlab.utils.geom as geom_utils
from fluidlab.configs.macros import *

@ti.data_oriented
class Dynamic(Mesh):
    # Dynamic mesh-based object
    def __init__(self, container, **kwargs):
        self.container = container
        super(Dynamic, self).__init__(**kwargs)

    def init_transform(self):
        super(Dynamic, self).init_transform()
        self.vertices = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))

    @ti.kernel
    def update_vertices(self, f: ti.i32):
        for i in self.vertices:
            self.vertices[i] = ti.cast(geom_utils.transform_by_trans_quat_ti(self.init_vertices[i], self.container.pos[f], self.container.quat[f]), self.vertices.dtype)
            self.vertex_normals[i] = ti.cast(geom_utils.transform_by_quat_ti(self.init_vertex_normals[i], self.container.quat[f]), self.vertices.dtype)

    @ti.func
    def sdf(self, f, pos_world):
        # sdf value from world coordinate
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)

        return self.sdf_(pos_voxels)

    @ti.func
    def sdf_(self, pos_voxels):
        # sdf value from voxels coordinate
        base = ti.floor(pos_voxels, ti.i32)
        signed_dist = ti.cast(0.0, DTYPE_TI)
        if (base >= self.sdf_voxels_res - 1).any() or (base < 0).any():
            signed_dist = 1.0
        else:
            signed_dist = 0.0
            for offset in ti.static(ti.grouped(ti.ndrange(2, 2, 2))):
                voxel_pos = base + offset
                w_xyz = 1 - ti.abs(pos_voxels - voxel_pos)
                w = w_xyz[0] * w_xyz[1] * w_xyz[2]
                signed_dist += w * self.sdf_voxels[voxel_pos]

        return signed_dist

    @ti.func
    def normal(self, f, pos_world):
        # compute normal with finite difference
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_voxels = geom_utils.transform_by_T_ti(pos_mesh, self.T_mesh_to_voxels[None], DTYPE_TI)
        normal_vec_voxels = self.normal_(pos_voxels)

        R_voxels_to_mesh = self.T_mesh_to_voxels[None][:3, :3].inverse()
        normal_vec_mesh = R_voxels_to_mesh @ normal_vec_voxels

        normal_vec_world = geom_utils.transform_by_quat_ti(normal_vec_mesh, self.container.quat[f])
        normal_vec_world = geom_utils.normalize(normal_vec_world)

        return normal_vec_world

    @ti.func
    def normal_(self, pos_voxels):
        # since we are in voxels frame, delta can be a relatively big value
        delta = ti.cast(1e-2, DTYPE_TI)
        normal_vec = ti.Vector([0, 0, 0], dt=DTYPE_TI)

        for i in ti.static(range(3)):
            inc = pos_voxels
            dec = pos_voxels
            inc[i] += delta
            dec[i] -= delta
            normal_vec[i] = (self.sdf_(inc) - self.sdf_(dec)) / (2 * delta)

        normal_vec = geom_utils.normalize(normal_vec)

        return normal_vec

    @ti.func
    def collider_v(self, f, pos_world, dt):
        pos_mesh = geom_utils.inv_transform_by_trans_quat_ti(pos_world, self.container.pos[f], self.container.quat[f])
        pos_world_new = geom_utils.transform_by_trans_quat_ti(pos_mesh, self.container.pos[f+1], self.container.quat[f+1])
        collider_v = (pos_world_new - pos_world) / dt
        return collider_v

    @ti.func
    def collide(self, f, pos_world, mat_v, dt):
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f, pos_world)
            influence = min(ti.exp(-signed_dist * self.softness), 1)
            if signed_dist <= 0 or (self.softness > 0 and influence > 0.1):
                collider_v = self.collider_v(f, pos_world, dt)

                if ti.static(self.friction > 10.0):
                    mat_v = collider_v
                else:
                    # v w.r.t collider
                    rel_v = mat_v - collider_v
                    normal_vec = self.normal(f, pos_world)
                    normal_component = rel_v.dot(normal_vec)

                    # remove inward velocity, if any
                    rel_v_t = rel_v - min(normal_component, 0) * normal_vec
                    rel_v_t_norm = rel_v_t.norm()

                    # tangential component after friction (if friction exists)
                    rel_v_t_friction = rel_v_t / rel_v_t_norm * max(0, rel_v_t_norm + normal_component * self.friction)

                    # tangential component after friction
                    flag = ti.cast(normal_component < 0 and rel_v_t_norm > EPS, DTYPE_TI)
                    rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
                    mat_v = collider_v + rel_v_t * influence + rel_v * (1-influence)

        return mat_v

    # @ti.func
    # def collide(self, f, pos_world, mat_v, dt):
    #     if ti.static(self.has_dynamics):
    #         signed_dist = self.sdf(f, pos_world)
    #         if signed_dist <= 0:
    #             collider_v = self.collider_v(f, pos_world, dt)

    #             if ti.static(self.friction > 10.0):
    #                 mat_v = collider_v
    #             else:
    #                 # v w.r.t collider
    #                 rel_v = mat_v - collider_v
    #                 normal_vec = self.normal(f, pos_world)
    #                 normal_component = rel_v.dot(normal_vec)

    #                 # remove inward velocity, if any
    #                 rel_v_t = rel_v - min(normal_component, 0) * normal_vec
    #                 rel_v_t_norm = rel_v_t.norm()

    #                 # tangential component after friction (if friction exists)
    #                 rel_v_t_friction = rel_v_t / rel_v_t_norm * max(0, rel_v_t_norm + normal_component * self.friction)

    #                 # tangential component after friction
    #                 flag = ti.cast(normal_component < 0 and rel_v_t_norm > EPS, DTYPE_TI)
    #                 rel_v_t = rel_v_t_friction * flag + rel_v_t * (1 - flag)
    #                 mat_v = collider_v + rel_v_t

    #     return mat_v


    @ti.func
    def is_collide(self, f, pos_world):
        flag = 0
        if ti.static(self.has_dynamics):
            signed_dist = self.sdf(f, pos_world)
            if signed_dist <= 0:
                flag = 1

        return flag