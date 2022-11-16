import pickle
import numpy as np
import os
from fluidlab.utils.misc import is_on_server
from tqdm import tqdm, trange
import shutil
import wandb

from moviepy.editor import ImageSequenceClip


def animate(imgs, filename='animation.mp4', _return=True, fps=10):
    if isinstance(imgs, dict):
        imgs = imgs['image']
    imgs = ImageSequenceClip(imgs, fps=fps)
    imgs.write_videofile(filename, fps=fps, logger=None)

def mkdir(path, clean=False):
    """Make directory.
    
    Args:
        path: path of the target directory
        clean: If there exist such directory, remove the original one or not
    """
    if clean and os.path.exists(path):
        shutil.rmtree(path)
    if not os.path.exists(path):
        os.makedirs(path)

class CMAESSolver:
    def __init__(self, env, cfg=None, exp_name='exp'):
        self.cfg = cfg
        self.env = env
        self.target_file = env.target_file
        self.exp_name = exp_name

    def solve(self):
        wandb.init(
            project='fliudlab',
            name=self.exp_name
        )
        exp_dir = os.path.join('logs', self.exp_name)
        mkdir(exp_dir, clean=False)
        
        taichi_env = self.env.taichi_env
        policy = self.env.cmaes_policy(self.cfg.init_range)

        taichi_env_state = taichi_env.get_state()

        def forward(sim_state, actions, horizon, horizon_action):

            taichi_env.set_state(sim_state, grad_enabled=False)

            # forward pass
            taichi_env.apply_agent_action_p(actions[-1])
            # cur_horizon = taichi_env.loss.temporal_range[1]
            cur_horizon = horizon
            step_loss_history = list()
            for i in range(cur_horizon):
                if i < horizon_action:
                    action = policy.get_action(actions, i, taichi_env.agent)
                else:
                    action = None
                taichi_env.step(action)
                step_loss = taichi_env.get_step_loss()
                step_loss_history.append(step_loss)
            return step_loss_history

        for iteration in range(self.cfg.n_iters):
            # sample action
            action_samples = policy.sample_actions()
            losses = list()
            rewards = list()
            step_loss_history_list = list()
            for actions in tqdm(action_samples):
                step_loss_history = forward(taichi_env_state['state'], actions, self.env.horizon, self.env.horizon_action)
                losses.append(np.sum([x['loss'] for x in step_loss_history]))
                rewards.append(np.sum([x['reward'] for x in step_loss_history]))
                step_loss_history_list.append(step_loss_history)
            policy.optimize(losses)
            print(f'==> iter {iteration}, min_loss = {np.min(losses)}, avg_loss = {np.mean(losses)}', )

            wandb_info = {
                'min_loss': np.min(losses),
                'avg_loss': np.mean(losses),
                'max_reward': np.max(rewards)
            }

            if iteration == 0 or (iteration + 1) % 5 == 0:
                best_idx = np.argmin(losses)
                actions = action_samples[best_idx]
                images, loss = self.render_policy(taichi_env, taichi_env_state, actions, self.env.horizon, self.env.horizon_action, policy)

                wandb_info['best_vis'] = wandb.Video(np.stack(images).transpose([0, 3, 1, 2]), fps=10)
            
            wandb.log(wandb_info)
        
            log_info = {
                'action_samples': action_samples,
                'losses': losses,
                'step_loss_history_list': step_loss_history_list
            }
            pickle.dump(log_info, open(os.path.join(exp_dir, f'iter-{iteration}.pkl'), 'wb'))


    def render_policy(self, taichi_env, init_state, actions, horizon, horizon_action, policy):
        if is_on_server():
            return

        save = True
        images = list()

        taichi_env.set_state(**init_state)
        taichi_env.apply_agent_action_p(actions[-1])
        sum_loss = 0
        for i in range(horizon):
            if i < horizon_action:
                action = policy.get_action(actions, i, taichi_env.agent)
            else:
                action = None
            taichi_env.step(action)
            loss = taichi_env.get_step_loss()['loss']
            sum_loss += loss

            if save:
                img = taichi_env.render('rgb_array')
                images.append(img)
            else:
                taichi_env.render('human')

        
        if save:
            return images, sum_loss

def cmaes_policy(env, cfg, exp_name):
    env.reset()
    solver = CMAESSolver(env, cfg, exp_name)
    solver.solve()
