import numpy as np
import taichi as ti
from yacs.config import CfgNode
from fluidlab.fluidengine.simulators import MPMSimulator, SmokeField
from fluidlab.fluidengine.agents import *
from fluidlab.fluidengine.meshes import Statics
from fluidlab.fluidengine.renderers import *
from fluidlab.fluidengine.bodies import Bodies
from fluidlab.configs.macros import *
from fluidlab.utils.misc import *

ti.init(arch=ti.gpu, device_memory_GB=8, packed=True)
# ti.init(arch=ti.gpu, device_memory_GB=10, packed=True)
# ti.init(arch=ti.gpu, device_memory_GB=10, packed=True, debug=True)

@ti.data_oriented
class TaichiEnv:
    '''
    TaichiEnv wraps all components in a simulation environment.
    '''    
    def __init__(
            self,
            dim=3,
            quality=1,
            particle_density=1e6,
            max_substeps_local=50,
            max_substeps_global=100000,
            horizon=100,
            ckpt_dest='disk',
            gravity=(0.0, -10.0, 0.0),
        ):
        self.particle_density    = particle_density
        self.dim                 = dim
        self.max_substeps_local  = max_substeps_local
        self.max_substeps_global = max_substeps_global
        self.horizon             = horizon
        self.ckpt_dest           = ckpt_dest
        self.t                   = 0

        # env components
        self.simulator = MPMSimulator(
            dim                 = self.dim,
            quality             = quality,
            horizon             = self.horizon,
            max_substeps_local  = self.max_substeps_local,
            max_substeps_global = self.max_substeps_global,
            gravity             = gravity,
            ckpt_dest           = ckpt_dest,
        )
        self.agent           = None
        self.statics         = Statics()
        self.particle_bodies = Bodies(dim=self.dim, particle_density=self.particle_density)
        self.renderer        = None
        self.loss            = None
        self.smoke_field     = None

        print('===>  TaichiEnv created.')

    def setup_agent(self, agent_cfg):
        self.agent = eval(agent_cfg.type)(
            max_substeps_local=self.max_substeps_local,
            max_substeps_global=self.max_substeps_global,
            max_action_steps_global=self.horizon,
            ckpt_dest=self.ckpt_dest,
            **agent_cfg.get('params', {}),
        )
        for effector_cfg_dict in agent_cfg.effectors:
            effector_cfg = CfgNode(effector_cfg_dict)
            self.agent.add_effector(
                type=effector_cfg.type,
                params=effector_cfg.params,
                mesh_cfg=effector_cfg.get('mesh', None),
                boundary_cfg=effector_cfg.boundary,
            )

    def setup_renderer(self, type='GGUI', **kwargs):
        if type == 'GGUI':
            Renderer = GGUIRenderer
        elif type == 'GL':
            Renderer = GLRenderer
        else:
            raise NotImplementedError

        self.renderer = Renderer(**kwargs)

    def setup_boundary(self, **kwargs):
        self.simulator.setup_boundary(**kwargs)

    def add_static(self, **kwargs):
        self.statics.add_static(**kwargs)

    def add_body(self, **kwargs):
        self.particle_bodies.add_body(**kwargs)

    def setup_smoke_field(self, **kwargs):
        self.smoke_field = SmokeField(
            dim=self.dim,
            ckpt_dest=self.ckpt_dest,
             **kwargs
        )

    def setup_loss(self, loss_cls, **kwargs):
        self.loss = loss_cls(
            max_loss_steps=self.horizon,
            **kwargs
        )

    def build(self):
        # particles
        self.particles = self.particle_bodies.get()

        if self.particles is not None:
            self.n_particles = len(self.particles['x'])
            self.has_particles = True
        else:
            self.n_particles = 0
            self.has_particles = False

        # build and initialize states of all environment components
        self.simulator.build(self.agent, self.smoke_field, self.statics, self.particles)

        if self.agent is not None:
            self.agent.build(self.simulator)

        if self.smoke_field is not None:
            self.smoke_field.build(self.simulator, self.agent)

        if self.renderer is not None:
            self.renderer.build(self.simulator, self.particles)

        if self.loss is not None:
            self.loss.build(self.simulator)

        self.t = 0

    def reset_grad(self):
        self.simulator.reset_grad()
        
        if self.agent is not None:
            self.agent.reset_grad()

        if self.smoke_field is not None:
            self.smoke_field.reset_grad()

        if self.loss is not None:
            self.loss.reset_grad()

    def enable_grad(self):
        self.simulator.enable_grad()

    def disable_grad(self):
        self.simulator.disable_grad()

    @property
    def grad_enabled(self):
        return self.simulator.grad_enabled

    def render(self, mode='human', tgt_particles=None):
        assert self.renderer is not None, 'No renderer available.'
        return self.renderer.render_frame(mode, tgt_particles, self.t)

    def get_state_RL(self):
        return self.simulator.get_state_RL()

    def step(self, action=None):
        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step(action=action)

        if self.loss:
            self.loss.step()

        self.t += 1

    def step_grad(self, action=None):
        if self.loss:
            self.loss.step_grad()

        if action is not None:
            assert self.agent is not None, 'Environment has no agent to execute action.'
            action = np.array(action).astype(DTYPE_NP)
        self.simulator.step_grad(action=action)

    def get_step_loss(self):
        assert self.loss is not None
        return self.loss.get_step_loss()

    def get_final_loss(self):
        assert self.loss is not None
        return self.loss.get_final_loss()

    def get_final_loss_grad(self):
        assert self.loss is not None
        self.loss.get_final_loss_grad()

    def get_state(self):
        return {
            'state': self.simulator.get_state(),
            'grad_enabled': self.grad_enabled
        }

    def set_state(self, state, grad_enabled=False):
        self.t = 0

        self.simulator.cur_substep_global = 0
        self.simulator.set_state(0, state)

        if grad_enabled:
            self.enable_grad()
        else:
            self.disable_grad()

        if self.loss:
            self.loss.reset()

    def apply_agent_action_p(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p(action_p)

    def apply_agent_action_p_grad(self, action_p):
        assert self.agent is not None, 'Environment has no agent to execute action.'
        self.agent.apply_action_p_grad(action_p)