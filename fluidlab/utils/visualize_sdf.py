import trimesh
import skimage
import argparse
import numpy as np
import pickle as pkl
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str)
    args = parser.parse_args()

    return args

def main():
    args = get_args()
    file = args.file
    voxels = pkl.load(open(file, 'rb'))['voxels']

    vertices, faces, normals, _ = skimage.measure.marching_cubes(voxels, level=0)
    new_mesh = trimesh.Trimesh(vertices=vertices, faces=faces, vertex_normals=normals)
    new_mesh.show()

if __name__ == '__main__':
    main()


