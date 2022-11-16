############ material type #############
WATER          = 0
MILK           = 1
COFFEE         = 2
ELASTIC        = 3
ICECREAM       = 4
RIGID          = 5
RIGID_HEAVY    = 6
RIGID_LIGHT    = 7
MILK_VIS       = 8
COFFEE_VIS     = 9
ELASTIC_DEMO   = 10
PLASTIC_DEMO   = 11
INVISCID_DEMO  = 12
VISCOUS_DEMO   = 13
INVISCID_DEMO2 = 14
INVISCID_DEMO3 = 15
ICECREAM1      = 16

CUP       = 50
TANK      = 51
LADDLE    = 52
POURER    = 53
DISPENSER = 54
CONE      = 55
ROBOT     = 56
BOTTLE    = 57
PILLAR    = 58
STIRRER   = 59
PLATE     = 60
BOWL      = 61

FRAME    = 100
TARGET   = 101
EFFECTOR = 102

MAT_LIQUID              = 200
MAT_PLASTO_ELASTIC      = 201
MAT_ELASTIC             = 202
MAT_RIGID               = 203
MAT_PLASTO_ELASTIC_DEMO = 204

############ material name #############
MAT_NAME = {
    WATER          : 'water',
    INVISCID_DEMO  : 'inviscid-demo',
    INVISCID_DEMO2 : 'inviscid-demo2',
    INVISCID_DEMO3 : 'inviscid-demo3',
    VISCOUS_DEMO   : 'viscous-demo',
    MILK           : 'milk',
    COFFEE         : 'coffee',
    ELASTIC        : 'elastic',
    ELASTIC_DEMO   : 'elastic-demo',
    PLASTIC_DEMO   : 'plastic-demo',
    RIGID          : 'rigid',
    RIGID_HEAVY    : 'rigid-heavy',
    RIGID_LIGHT    : 'rigid-light',
    ICECREAM       : 'ice-cream',
    ICECREAM1       : 'ice-cream1',
    MILK_VIS       : 'milk-viscous',
    COFFEE_VIS     : 'coffee-viscous',
}

############ material class #############
MAT_CLASS = {
    WATER          : MAT_LIQUID,
    INVISCID_DEMO  : MAT_LIQUID,
    INVISCID_DEMO2 : MAT_LIQUID,
    INVISCID_DEMO3 : MAT_LIQUID,
    VISCOUS_DEMO   : MAT_LIQUID,
    MILK           : MAT_LIQUID,
    COFFEE         : MAT_LIQUID,
    ELASTIC        : MAT_ELASTIC,
    ELASTIC_DEMO   : MAT_ELASTIC,
    PLASTIC_DEMO   : MAT_PLASTO_ELASTIC_DEMO,
    RIGID          : MAT_RIGID,
    RIGID_HEAVY    : MAT_RIGID,
    RIGID_LIGHT    : MAT_RIGID,
    ICECREAM       : MAT_PLASTO_ELASTIC,
    ICECREAM1       : MAT_PLASTO_ELASTIC,
    MILK_VIS       : MAT_LIQUID,
    COFFEE_VIS     : MAT_LIQUID,
}

############ default color #############
COLOR = {
    WATER          : (0.3, 0.8, 1.0, 0.0),
    # WATER          : (0.3, 0.8, 1.0, 1.0),
    INVISCID_DEMO  : (0.3, 0.8, 1.0, 1.0),
    INVISCID_DEMO2 : (1.0, 0.2, 0.1, 1.0),
    INVISCID_DEMO3 : (1.0, 0.2, 0.1, 1.0),
    VISCOUS_DEMO   : (1.0, 0.2, 0.1, 1.0),
    # INVISCID_DEMO  : (0.3, 0.8, 1.0, 0.0),
    # INVISCID_DEMO2 : (0.3, 0.8, 1.0, 0.0),
    # INVISCID_DEMO3 : (1.0, 0.2, 0.1, 0.2),
    # VISCOUS_DEMO   : (1.0, 0.2, 0.1, 0.2),
    MILK           : (0.9, 0.9, 0.9, 1.0),
    COFFEE         : (0.58, 0.42, 0.22, 1.0),
    # COFFEE       : (0.48, 0.32, 0.12, 0.8),
    ELASTIC        : (1.0, 1.0, 1.0, 1.0),
    ELASTIC_DEMO   : (1.0, 1.0, 1.0, 1.0),
    PLASTIC_DEMO   : (1.0, 1.0, 1.0, 1.0),
    ICECREAM       : (1.0, 1.0, 1.0, 1.0),
    ICECREAM1       : (1.0, 1.0, 1.0, 1.0),
    RIGID          : (1.0, 0.5, 0.5, 1.0),
    RIGID_HEAVY    : (1.0, 0.5, 0.5, 1.0),
    RIGID_LIGHT    : (1.0, 0.5, 0.5, 1.0),
    MILK_VIS       : (0.9, 0.9, 0.9, 1.0),
    COFFEE_VIS     : (0.58, 0.42, 0.22, 1.0),

    CUP       : (0.9, 0.9, 0.9, 1.0),
    TANK      : (0.70, 0.95, 0.96, 0.6),
    BOWL      : (0.78, 0.56, 0.12, 1.0),
    LADDLE    : (1.0, 1.0, 1.0, 1.0),
    POURER    : (1.0, 1.0, 1.0, 1.0),
    DISPENSER : (1.0, 1.0, 1.0, 1.0),
    CONE      : (0.645, 0.474, 0.303, 1.0),
    ROBOT     : (1.0, 1.0, 1.0, 1.0),
    BOTTLE    : (0.70, 0.95, 0.96, 0.5),
    PILLAR    : (1.0, 1.0, 1.0, 1.0),
    STIRRER   : (1.0, 1.0, 1.0, 1.0),
    PLATE     : (1.0, 1.0, 1.0, 1.0),

    FRAME    : (1.0, 0.2, 0.2, 1.0),
    TARGET   : (0.2, 0.9, 0.2, 0.4),
    EFFECTOR : (1.0, 0.0, 0.0, 1.0),
}


############ properties #############
FRICTION = {
    CUP     : 0.5,
    TANK    : 0.5,
    BOWL    : 0.0,
    LADDLE  : 0.1,
    CONE    : 8.0,
    BOTTLE  : 0.1,
    PILLAR  : 0.0,
    STIRRER : 8.0,
    PLATE   : 0.1,
}

MU = {
    WATER          : 0.0,
    INVISCID_DEMO  : 0.0,
    INVISCID_DEMO2 : 0.0,
    INVISCID_DEMO3 : 0.0,
    VISCOUS_DEMO   : 800.0,
    MILK           : 0.0,
    COFFEE         : 0.0,
    MILK_VIS       : 200.0,
    COFFEE_VIS     : 200.0,
    ELASTIC        : 416.67,
    ELASTIC_DEMO   : 10.0,
    PLASTIC_DEMO   : 160.0,
    ICECREAM       : 416.67,
    ICECREAM1       : 216.67,
    RIGID          : 416.67,
    RIGID_HEAVY    : 416.67,
    RIGID_LIGHT    : 416.67,
}

LAMDA = {
    WATER          : 277.78,
    INVISCID_DEMO  : 277.78,
    INVISCID_DEMO2 : 277.78,
    INVISCID_DEMO3 : 277.78,
    VISCOUS_DEMO   : 277.78,
    MILK           : 277.78,
    COFFEE         : 277.78,
    MILK_VIS       : 277.78,
    COFFEE_VIS     : 277.78,
    ELASTIC        : 277.78,
    ELASTIC_DEMO   : 100.0,
    PLASTIC_DEMO   : 277.78,
    ICECREAM       : 277.78,
    ICECREAM1       : 277.78,
    RIGID          : 277.78,
    RIGID_HEAVY    : 277.78,
    RIGID_LIGHT    : 277.78,
}

RHO = {
    WATER          : 1.0,
    INVISCID_DEMO  : 5.0,
    INVISCID_DEMO2 : 1.0,
    INVISCID_DEMO3 : 3.0,
    VISCOUS_DEMO   : 5.0,
    MILK           : 0.5,
    COFFEE         : 1.0,
    MILK_VIS       : 1.0,
    COFFEE_VIS     : 1.0,
    ELASTIC        : 1.0,
    ELASTIC_DEMO   : 1.0,
    PLASTIC_DEMO   : 1.0,
    ICECREAM       : 0.5,
    ICECREAM1       : 0.5,
    RIGID          : 1.0,
    RIGID_HEAVY    : 10.0,
    RIGID_LIGHT    : 0.5,
}

############ dtype #############
import numpy as np
import torch
import taichi as ti
dprecision = 32
# dprecision = 64
DTYPE_TI = eval(f'ti.f{dprecision}')
DTYPE_NP = eval(f'np.float{dprecision}')
DTYPE_TC = eval(f'torch.float{dprecision}')

EPS = 1e-12

############ misc #############
NOWHERE = [-100.0, -100.0, -100.0]


