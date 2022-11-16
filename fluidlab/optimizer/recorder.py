import os
import cv2
import numpy as np
import taichi as ti
import pickle as pkl
from fluidlab.utils.misc import is_on_server

class Recorder:
    def __init__(self, env):
        self.env = env
        self.target_file = env.target_file
        if self.target_file is not None:
            os.makedirs(os.path.dirname(self.target_file), exist_ok=True)

    def record(self):
        policy = self.env.demo_policy()
        taichi_env = self.env.taichi_env

        # initialize ...
        taichi_env_state = taichi_env.get_state()

        # start recording
        target = {
            'x'    : [],
            'used' : [],
            'mat'  : None
        }
        taichi_env.set_state(**taichi_env_state)
        action_p = policy.get_actions_p()
        if action_p is not None:
            taichi_env.apply_agent_action_p(action_p)
        
        save = False
        save = True
        if save:
            os.makedirs(f'tmp/recorder', exist_ok=True)
            
        for i in range(self.env.horizon):
            if i < self.env.horizon_action:
                action = policy.get_action_v(i)
            else:
                action = None
            # print(action)
            taichi_env.step(action)

            # get state
            if self.target_file is not None:
                cur_state = taichi_env.get_state()
                if taichi_env.has_particles:
                    target['x'].append(cur_state['state']['x'])
                    target['used'].append(cur_state['state']['used'])

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/recorder/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    taichi_env.render('human')

        # while True:
        #     taichi_env.render()

        if self.target_file is not None:
            target['mat'] = taichi_env.simulator.particles_i.mat.to_numpy()
            if os.path.exists(self.target_file):
                os.remove(self.target_file)
            pkl.dump(target, open(self.target_file, 'wb'))
            print(f'==========> New target generated and dumped to {self.target_file}.')

    def replay_traj(self):
        taichi_env = self.env.taichi_env
        target = pkl.load(open(self.target_file, 'rb'))

        save = False
        save = True
        if save:
            os.makedirs(f'tmp/replay', exist_ok=True)
            
        for i in range(self.env.horizon):
            taichi_env.simulator.set_x(0, target['x'][i])
            taichi_env.simulator.set_used(0, target['used'][i])

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/replay/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    taichi_env.render('human')

    def replay_policy(self, policy_path):
        taichi_env = self.env.taichi_env

        policy = pkl.load(open(policy_path, 'rb'))

        taichi_env.apply_agent_action_p(policy.get_actions_p())

        save = False
        save = True
        if save:
            os.makedirs(f'tmp/replay', exist_ok=True)
            
        for i in range(self.env.horizon):
            if i < self.env.horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            taichi_env.step(action)

            if save:
                img = taichi_env.render('rgb_array')
                cv2.imwrite(f'tmp/replay/{i:04d}.png', img[:, :, ::-1])
            else:
                if not is_on_server():
                    taichi_env.render('human')


def record_target(env, record, replay, path=None):
    env.reset()

    recorder = Recorder(env)
    if record:
        recorder.record()
    elif replay:
        recorder.replay_policy(path)
        # recorder.replay_traj()
