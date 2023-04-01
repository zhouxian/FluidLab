import os
import copy
import trimesh
import numpy as np
import taichi as ti
import pickle as pkl
from fluidlab.configs.macros import *
from scipy.spatial.transform import Rotation
from fluidlab.utils.misc import *
from fluidlab.utils.mesh import *
import fluidlab.utils.geom as geom_utils

@ti.data_oriented
class Mesh:
    def __init__(
            self,
            file,
            material,
            file_vis     = None,
            sdf_res      = 128,
            pos          = (0.0, 0.0, 0.0),
            euler        = (0.0, 0.0, 0.0),
            scale        = (1.0, 1.0, 1.0),
            softness     = 0,
            has_dynamics = False
        ):
        self.pos            = eval_str(pos)
        self.euler          = eval_str(euler)
        self.scale          = eval_str(scale)
        self.raw_file       = file
        self.sdf_res        = sdf_res
        self.raw_file_vis   = file if file_vis is None else file_vis
        self.material       = eval_str(material)
        self.has_dynamics   = has_dynamics
        self.softness       = softness
        self.gl_renderer_id = None

        self.load_file()
        self.init_transform()

    def load_file(self):
        # mesh
        self.process_mesh()
        self.mesh = trimesh.load(self.processed_file_path)
        self.raw_vertices = np.array(self.mesh.vertices, dtype=np.float32)
        self.raw_vertex_normals_np = np.array(self.mesh.vertex_normals, dtype=np.float32)
        self.faces_np = np.array(self.mesh.faces, dtype=np.int32).flatten()

        self.n_vertices = len(self.raw_vertices)
        self.n_faces = len(self.faces_np)

        # load color
        vcolor_path = self.raw_file_path.replace('obj', 'vcolor')
        if os.path.exists(vcolor_path):
            self.colors_np = pkl.load(open(vcolor_path, 'rb')).astype(np.float32)
        else:
            # if vcolor file does not exist, get color based on material
            self.colors_np = np.tile([COLOR[self.material]], [self.n_vertices, 1]).astype(np.float32)

        if self.has_dynamics:
            # sdf
            self.friction = FRICTION[self.material]
            sdf_data = pkl.load(open(self.processed_sdf_path, 'rb'))
            self.sdf_voxels_np = sdf_data['voxels'].astype(DTYPE_NP)
            self.sdf_voxels_res = self.sdf_voxels_np.shape[0]
            self.T_mesh_to_voxels_np = sdf_data['T_mesh_to_voxels'].astype(DTYPE_NP)

    def process_mesh(self):
        self.raw_file_path       = get_raw_mesh_path(self.raw_file)
        self.raw_file_vis_path   = get_raw_mesh_path(self.raw_file_vis)
        self.processed_file_path = get_processed_mesh_path(self.raw_file, self.raw_file_vis)
        if self.has_dynamics:
            self.processed_sdf_path = get_processed_sdf_path(self.raw_file, self.sdf_res)

        # clean up mesh
        if not os.path.exists(self.processed_file_path):
            print(f'===> Processing mesh(es) {self.raw_file_path} and vis {self.raw_file_vis_path}.')
            raw_mesh     = load_mesh(self.raw_file_path)
            raw_mesh_vis = load_mesh(self.raw_file_vis_path)

            # process and save
            processed_mesh = cleanup_mesh(normalize_mesh(raw_mesh_vis, raw_mesh))
            processed_mesh.export(self.processed_file_path)
            print(f'===> Processed mesh saved as {self.processed_file_path}.')

        # generate sdf
        if self.has_dynamics and not os.path.exists(self.processed_sdf_path):
            print(f'===> Computing sdf for {self.raw_file_path}. This might take minutes...')
            raw_mesh           = load_mesh(self.raw_file_path)
            processed_mesh_sdf = cleanup_mesh(normalize_mesh(raw_mesh))
            sdf_data           = compute_sdf_data(processed_mesh_sdf, self.sdf_res)

            pkl.dump(sdf_data, open(self.processed_sdf_path, 'wb'))
            print(f'===> sdf saved as {self.processed_sdf_path}.')

    def init_transform(self):
        scale = np.array(self.scale, dtype=DTYPE_NP)
        pos = np.array(self.pos, dtype=DTYPE_NP)
        quat = geom_utils.xyzw_to_wxyz(Rotation.from_euler('zyx', self.euler[::-1], degrees=True).as_quat().astype(DTYPE_NP))

        # apply initial transforms (scale then quat then pos)
        T_init = geom_utils.trans_quat_to_T(pos, quat) @ geom_utils.scale_to_T(scale)
        self.init_vertices_np = geom_utils.transform_by_T_np(self.raw_vertices, T_init).astype(np.float32)
        self.init_vertices_np_flattened = self.init_vertices_np.flatten()

        R_init = geom_utils.trans_quat_to_T(None, quat)
        self.init_vertex_normals_np = geom_utils.transform_by_T_np(self.raw_vertex_normals_np, R_init).astype(np.float32)
        self.init_vertex_normals_np_flattened = self.init_vertex_normals_np.flatten()

        # init ti fields
        self.init_vertices       = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.init_vertex_normals = ti.Vector.field(3, dtype=ti.f32, shape=(self.n_vertices))
        self.faces               = ti.field(dtype=ti.i32, shape=(self.n_faces))
        self.colors              = ti.Vector.field(self.colors_np.shape[1], dtype=ti.f32, shape=(self.n_vertices))

        self.init_vertices.from_numpy(self.init_vertices_np)
        self.init_vertex_normals.from_numpy(self.init_vertex_normals_np)
        self.faces.from_numpy(self.faces_np)
        self.colors.from_numpy(self.colors_np)

        if self.has_dynamics:
            self.T_mesh_to_voxels_np = self.T_mesh_to_voxels_np @ np.linalg.inv(T_init)
            self.sdf_voxels          = ti.field(dtype=DTYPE_TI, shape=self.sdf_voxels_np.shape)
            self.T_mesh_to_voxels    = ti.Matrix.field(4, 4, dtype=DTYPE_TI, shape=())

            self.sdf_voxels.from_numpy(self.sdf_voxels_np)
            self.T_mesh_to_voxels.from_numpy(self.T_mesh_to_voxels_np)

