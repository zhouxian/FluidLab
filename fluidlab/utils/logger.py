import os
import cv2
import h5py 
import numpy as np
import pickle as pkl
from time import time
from torch.utils.tensorboard import SummaryWriter as TorchWriter
from fluidlab.utils.misc import get_src_dir

class SummaryWriter:
    def __init__(self, exp_name):
        self.dir = os.path.join(get_src_dir(), '..', 'logs', 'logs', exp_name)
        os.makedirs(self.dir, exist_ok=True)
        self.writer = TorchWriter(log_dir=self.dir)

    def write(self, iteration, info):
        for key in info:
            self.writer.add_scalar(key, info[key], iteration)

class ImageWriter:
    def __init__(self, exp_name):
        self.dir = os.path.join(get_src_dir(), '..', 'logs', 'imgs', exp_name)

    def write(self, img, iteration, step):
        img_dir = os.path.join(self.dir, f'{iteration}')
        img_path = os.path.join(self.dir, f'{iteration}/{step:04d}.png')
        os.makedirs(img_dir, exist_ok=True)
        cv2.imwrite(img_path, img[:, :, ::-1])

class TrajectoryWriter:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.dir = os.path.join(get_src_dir(), '..', 'trajs')
    def write(self, sim_state, img_obs, iteration: int):
        with h5py.File(f"{self.exp_name}.hdf5", "a") as f:
            if self.exp_name not in f.keys():
                f.create_group(self.exp_name)
            g = f[self.exp_name]
            print(g.keys())
class Logger:
    def __init__(self, exp_name):
        self.exp_name = exp_name
        self.summary_writer = SummaryWriter(exp_name)
        self.image_writer = ImageWriter(exp_name)
        self.last_step_t = time()

    def write_img(self, img, iteration, step):
        self.image_writer.write(img, iteration, step)

    def save_policy(self, policy, iteration):
        policy_dir = os.path.join(get_src_dir(), '..', 'logs', 'policies', self.exp_name)
        os.makedirs(policy_dir, exist_ok=True)
        pkl.dump(policy, open(os.path.join(policy_dir, f'{iteration:04d}.pkl'), 'wb'))

    def log(self, iteration, info):
        cur_t = time()
        print_msg = f'Iteration: {iteration}, '
        tb_info = dict()
        for key in info:
            val = info[key]
            if type(val) is int:
                print_msg += f'{key}: {info[key]}, '
                tb_info[key] = info[key]
            elif type(val) is float or type(val) is np.float32:
                print_msg += f'{key}: {info[key]:.3f}, '
                tb_info[key] = info[key]
            else:
                pass

        print_msg += f'Step time: {cur_t-self.last_step_t:.2f}s'
        print(print_msg)

        self.summary_writer.write(iteration, tb_info)
        self.last_step_t = cur_t

if __name__ == "__main__":
    # Create a test TrajectoryWriter
    writer = TrajectoryWriter("latteart")
    writer.write([], [], 1)
