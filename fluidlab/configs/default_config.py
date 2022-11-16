from yacs.config import CfgNode as CN

_C = CN()

####################################################################
########################## EXP CONFIG #############################
####################################################################

_C.EXP = CN()
_C.EXP.seed = 0
_C.EXP.env_name = ''


####################################################################
######################## SOLVER CONFIG #############################
####################################################################

_C.SOLVER = CN()
_C.SOLVER.n_iters = 100
_C.SOLVER.init_range = CN()
_C.SOLVER.init_range.v = ()
_C.SOLVER.init_range.p = ()
_C.SOLVER.init_sampler = 'uniform'
_C.SOLVER.optim = CN()
_C.SOLVER.optim.lr = 0.1
_C.SOLVER.optim.bounds = (-1.0, 1.0)
_C.SOLVER.optim.type = '' # ['Adam', 'Momentum' ]
# Adam specific
_C.SOLVER.optim.beta_1 = 0.9
_C.SOLVER.optim.beta_2 = 0.9999
_C.SOLVER.optim.epsilon = 1e-8
# Momentum specific
_C.SOLVER.optim.momentum = 0.9
_C.SOLVER.optim.trainable = None
_C.SOLVER.optim.fix_dim = None


def get_default_cfg():
    return _C.clone()