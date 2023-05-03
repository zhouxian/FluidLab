import os
import cv2
impoet numpy as np
import taichi as ti
from fluidlab.utils.misc import is_on_server

from fluidlab.fluidengine.taichi_env import TaichiEnv


class Solver:
    def __init__(self, env, logger=None, cfg=None):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.logger = logger
    
    def create_trajs(self, taichi_env, init_state, policy, horizon, horizon_action, iteration):
        taichi_env = self.env.taichi_env

        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_actions_p())

        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            taichi_env.step(action)

            img = taichi_env.render('rgb_array')
            self.logger.write_img(img, iteration, i)

    def solve(self):
        taichi_env = self.env.taichi_env
        policy = self.env.trainable_policy(self.cfg.optim, self.cfg.init_range)

        taichi_env_state = taichi_env.get_state()

        def forward_backward(sim_state, policy, horizon, horizon_action):

            taichi_env.set_state(sim_state, grad_enabled=True)

            # forward pass
            from time import time
            t1 = time()
            taichi_env.apply_agent_action_p(policy.get_actions_p())
            cur_horizon = taichi_env.loss.temporal_range[1]
            for i in range(cur_horizon):
                if i < horizon_action:
                    action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
                else:
                    action = None
                taichi_env.step(action)

                # print(i, taichi_env.get_step_loss())
                # self.env._get_obs()

            loss_info = taichi_env.get_final_loss()
            t2 = time()

            # backward pass
            taichi_env.reset_grad()
            taichi_env.get_final_loss_grad()

            for i in range(cur_horizon-1, policy.freeze_till-1, -1):
                if i < horizon_action:
                    action = policy.get_action_v(i)
                else:
                    action = None
                taichi_env.step_grad(action)

            taichi_en.apply_agent_action_p_grad(policy.get_actions_p())
            t3 = time()
            print(f'=======> forward: {t2-t1:.2f}s backward: {t3-t2:.2f}s')
            return loss_info, taichi_env.agent.get_grad(horizon_action)

        for iteration in range(self.cfg.n_iters):
            self.logger.save_policy(policy, iteration)
            if iteration % 50 == 0:
                self.render_policy(taichi_env, taichi_env_state, policy, self.env.horizon, self.env.horizon_action, iteration)
            loss_info, grad = forward_backward(taichi_env_state['state'], policy, self.env.horizon, self.env.horizon_action)
            loss = loss_info['loss']
            loss_info['iteration'] = iteration
            policy.optimize(grad, loss_info)

            if self.logger is not None:
                loss_info['lr'] = policy.optim.lr
                self.logger.log(iteration, loss_info)


    def render_policy(self, taichi_env, init_state, policy, horizon, horizon_action, iteration):
        if is_on_server():
            return

        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(policy.get_actions_p())

        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action_v(i, agent=taichi_env.agent, update=True)
            else:
                action = None
            taichi_env.step(action)
            # print(i, taichi_env.get_step_loss())

            save = True
            save = False
            if save:
                img = taichi_env.render('rgb_array')
                self.logger.write_img(img, iteration, i)
            else:
                taichi_env.render('human')

def solve_policy(env, logger, cfg):
    env.reset()
    solver = Solver(env, logger, cfg)
    solver.solve()
