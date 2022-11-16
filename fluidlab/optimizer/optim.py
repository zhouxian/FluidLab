import numpy as np

class Optimizer:
    def __init__(self, parameters_shape, cfg):
        self.cfg = cfg
        self.lr = self.cfg.lr
        print('====> lr: ', self.lr)
        self.init_lr = self.cfg.lr
        self.parameters_shape = parameters_shape

        self.initialize()

    def initialize(self):
        raise NotImplementedError

    def _step(self, grads):
        raise NotImplementedError

    def step(self, parameters, grads):
        return self._step(parameters, grads)

class Adam(Optimizer):
    def initialize(self):
        self.momentum_buffer = np.zeros(self.parameters_shape).astype(np.float64)
        self.v_buffer = np.zeros_like(self.momentum_buffer).astype(np.float64)
        self.iter = 0

    def _step(self, parameters, grads):
        beta_1 = self.cfg.beta_1
        beta_2 = self.cfg.beta_2
        epsilon = self.cfg.epsilon
        m_t = beta_1 * self.momentum_buffer + (1 - beta_1) * grads  # updates the moving averages of the gradient
        v_t = beta_2 * self.v_buffer + (1 - beta_2) * (grads * grads)  # updates the moving averages of the squared gradient
        self.momentum_buffer[:] = m_t
        self.v_buffer[:] = v_t

        m_cap = m_t / (1 - (beta_1 ** (self.iter + 1)))  # calculates the bias-corrected estimates
        v_cap = v_t / (1 - (beta_2 ** (self.iter + 1)))  # calculates the bias-corrected estimates

        self.iter += 1
        return parameters - (self.lr * m_cap) / (np.sqrt(v_cap) + epsilon)
