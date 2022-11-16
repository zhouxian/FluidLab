import os
import gym
import torch
import random
import argparse
import numpy as np

import fluidlab.envs
from fluidlab.utils.logger import Logger
from fluidlab.optimizer.solver import solve_policy
# from fluidlab.optimizer.cmaes import cmaes_policy
from fluidlab.optimizer.recorder import record_target
from fluidlab.utils.config import load_config

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp_name", type=str, default='test')
    parser.add_argument("--env_name", type=str, default='')
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg_file", type=str, default=None)
    parser.add_argument("--record", action='store_true')
    parser.add_argument("--replay", action='store_true')
    parser.add_argument("--cmaes", action='store_true')
    parser.add_argument("--path", type=str, default=None)


    args = parser.parse_args()

    return args

def main():
    args = get_args()
    if args.cfg_file is not None:
        cfg = load_config(args.cfg_file)
    else:
        cfg = None

    if args.record or args.replay:
        if cfg is not None:
            env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=False, loss_type='diff')
        else:
            env = gym.make(args.env_name, seed=args.seed, loss=False, loss_type='diff')
        record_target(env, record=args.record, replay=args.replay, path=args.path)
    elif args.cmaes:
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='default')
        cmaes_policy(env, cfg.SOLVER, args.exp_name)
    else:
        logger = Logger(args.exp_name)
        # env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='default')
        env = gym.make(cfg.EXP.env_name, seed=cfg.EXP.seed, loss=True, loss_type='diff')
        solve_policy(env, logger, cfg.SOLVER)

if __name__ == '__main__':
    main()
