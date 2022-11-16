import trimesh
import argparse
import numpy as np
from mesh_to_sdf import mesh_to_sdf
import pickle as pkl

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--in_file", type=str)
    # mesh for rendering purpose
    parser.add_argument("--in_file_vis", type=str, default=None)
    parser.add_argument("--out_file", type=str)
    parser.add_argument("--res", type=int, default=128)
    parser.add_argument('--no-sdf', dest='sdf', action='store_false')
    parser.set_defaults(sdf=True)
    args = parser.parse_args()

    return args

def main():
    args = get_args()

    mesh = trimesh.load(args.in_file)
    if args.in_file_vis == None:
        args.in_file_vis = args.in_file
    mesh_vis = trimesh.load(args.in_file_vis)

    if type(mesh) is trimesh.Scene:
        mesh = trimesh.load(args.in_file, force='mesh', skip_texture=True)
        mesh_vis = trimesh.load(args.in_file_vis, force='mesh', skip_texture=True)

    vertices = np.array(mesh.vertices)
    faces = np.array(mesh.faces)
    vertex_normals = np.array(mesh.vertex_normals)
    face_normals = np.array(mesh.face_normals)

    vertices_vis = np.array(mesh_vis.vertices)
    faces_vis = np.array(mesh_vis.faces)
    vertex_normals_vis = np.array(mesh_vis.vertex_normals)
    face_normals_vis = np.array(mesh_vis.face_normals)

    # normalize to [-0.5, 0.5]
    scale = (vertices.max(0) - vertices.min(0)).max()
    center = (vertices.max(0) + vertices.min(0))/2
    vertices -= center
    vertices /= scale
    vertices_vis -= center
    vertices_vis /= scale

    # retain only vertices and faces
    new_mesh_vis = trimesh.Trimesh(
        vertices=vertices_vis,
        faces=faces_vis,
        vertex_normals=vertex_normals_vis,
        face_normals=face_normals_vis,
    )
    # save new mesh
    new_mesh_vis.export(args.out_file)
    print('==> Mesh cleaned up and exported.')

    new_mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        vertex_normals=vertex_normals,
        face_normals=face_normals,
    )

    # generate sdf
    if args.sdf:
        res = args.res
        scan_count = int(res / 64 * 100)
        scan_resolution = 400
        voxels_radius = 0.6
        x = np.linspace(-voxels_radius, voxels_radius, res)
        y = np.linspace(-voxels_radius, voxels_radius, res)
        z = np.linspace(-voxels_radius, voxels_radius, res)
        X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
        query_points = np.stack([X, Y, Z], axis=-1).reshape((-1, 3))

        print('==> Computing sdf...')
        voxels = mesh_to_sdf(new_mesh, query_points, scan_count=scan_count, scan_resolution=scan_resolution, normal_sample_count=11)
        voxels = voxels.reshape([res, res, res])
        T_mesh_to_voxels = np.eye(4)
        T_mesh_to_voxels[:3, :3] *= (res - 1) / (voxels_radius * 2)
        T_mesh_to_voxels[:3, 3] = (res - 1) / 2

        sdf_data = {
            'voxels': voxels,
            'T_mesh_to_voxels': T_mesh_to_voxels,
        }
        sdf_file = args.out_file.replace('obj', 'sdf')
        pkl.dump(sdf_data, open(sdf_file, 'wb'))
        print('==> sdf saved.')


if __name__ == '__main__':
    main()
