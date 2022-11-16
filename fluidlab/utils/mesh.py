import os
import trimesh
import numpy as np
from .misc import *
from mesh_to_sdf import mesh_to_sdf

def get_raw_mesh_path(file):
    assert file.endswith('.obj')
    return os.path.join(get_src_dir(), 'assets', 'meshes', 'raw', file)

def get_processed_mesh_path(file, file_vis):
    assert file.endswith('.obj') and file_vis.endswith('.obj')
    processed_file      = f"{file.replace('.obj', '')}-{file_vis.replace('.obj', '')}.obj"
    processed_file_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'processed', processed_file)
    return processed_file_path

def get_processed_sdf_path(file, sdf_res):
    assert file.endswith('.obj')
    processed_sdf = f"{file.replace('.obj', '')}-{sdf_res}.sdf"
    processed_sdf_path = os.path.join(get_src_dir(), 'assets', 'meshes', 'processed', processed_sdf)
    return processed_sdf_path

def get_voxelized_mesh_path(file, voxelize_res):
    assert file.endswith('.obj')
    return os.path.join(get_src_dir(), 'assets', 'meshes', 'voxelized', f"{file.replace('.obj', '')}-{voxelize_res}.vox")

def load_mesh(file):
    return trimesh.load(file, force='mesh', skip_texture=True)

def normalize_mesh(mesh, mesh_actual=None):
    '''
    Normalize mesh_dict to [-0.5, 0.5] using size of mesh_dict_actual.
    '''
    if mesh_actual is None:
        mesh_actual = mesh

    scale  = (mesh_actual.vertices.max(0) - mesh_actual.vertices.min(0)).max()
    center = (mesh_actual.vertices.max(0) + mesh_actual.vertices.min(0))/2

    normalized_mesh = mesh.copy()
    normalized_mesh.vertices -= center
    normalized_mesh.vertices /= scale
    return normalized_mesh

def scale_mesh(mesh, scale):
    scale = np.array(scale)
    return trimesh.Trimesh(
        vertices = mesh.vertices * scale,
        faces    = mesh.faces,
    )

def cleanup_mesh(mesh):
    '''
    Retain only mesh's vertices, faces, and normals.
    '''
    return trimesh.Trimesh(
        vertices       = mesh.vertices,
        faces          = mesh.faces,
        vertex_normals = mesh.vertex_normals,
        face_normals   = mesh.face_normals,
    )

def compute_sdf_data(mesh, res):
    '''
    Convert mesh to sdf voxels and a transformation matrix from mesh frame to voxel frame.
    '''
    scan_count = int(res / 64 * 100)
    scan_resolution = 400
    voxels_radius = 0.6
    x = np.linspace(-voxels_radius, voxels_radius, res)
    y = np.linspace(-voxels_radius, voxels_radius, res)
    z = np.linspace(-voxels_radius, voxels_radius, res)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

    voxels = mesh_to_sdf(mesh, query_points, scan_count=scan_count, scan_resolution=scan_resolution, normal_sample_count=11)
    voxels = voxels.reshape([res, res, res])

    T_mesh_to_voxels = np.eye(4)
    T_mesh_to_voxels[:3, :3] *= (res - 1) / (voxels_radius * 2)
    T_mesh_to_voxels[:3, 3] = (res - 1) / 2

    sdf_data = {
        'voxels': voxels,
        'T_mesh_to_voxels': T_mesh_to_voxels,
    }
    return sdf_data

def voxelize_mesh(file, res):
    '''
    Normalize mesh to [-0.5, 0.5] and then voxelize.
    '''
    raw_mesh        = load_mesh(file)
    normalized_mesh = cleanup_mesh(normalize_mesh(raw_mesh))
    voxelized_mesh  = normalized_mesh.voxelized(pitch=1.0/res).fill()
    return voxelized_mesh
