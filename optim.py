import torch


class BatchedAdam:
    def __init__(self, n_models, params, lr, betas = (0.9, 0.999), eps = 1e-8, weight_decays = 0.0):
        self.n_models = n_models
        self.params = list(params)
        self.m_params = [torch.zeros_like(p) for p in self.params]
        self.v_params = [torch.zeros_like(p) for p in self.params]

        self.learning_rates = lr
        self.beta1 = betas[0]
        self.beta2 = betas[1]
        self.eps = eps
        self.t = 0

        self.weight_decays = weight_decays
        
        if isinstance(weight_decays, torch.Tensor):
            if weight_decays.shape[0] == self.n_models:
                if torch.any(weight_decays > 0):
                    self.weight_decay_fn = self.add_weight_decay_tensor
                else:
                    self.weight_decay_fn = self.add_weight_decay_none
            else:
                if weight_decays.item() > 0:
                    self.weight_decay_fn = self.add_weight_decay_float
                else:
                    self.weight_decay_fn = self.add_weight_decay_none
        else:
            if weight_decays > 0:
                self.weight_decay_fn = self.add_weight_decay_float
            else:
                self.weight_decay_fn = self.add_weight_decay_none
        
    def add_weight_decay_none(self, grad: torch.Tensor, param: torch.Tensor):
        pass

    def add_weight_decay_float(self, grad: torch.Tensor, param: torch.Tensor):
        grad.add_(param, alpha=self.weight_decays)

    def add_weight_decay_tensor(self, grad: torch.Tensor, param: torch.Tensor):
        wd_shape = [self.n_models] + [1] * (param.dim() - 1)
        wd_tensor = self.weight_decays.view(wd_shape)
        grad.add_(param * wd_tensor)

    def step(self):
        self.t += 1
        with torch.no_grad():
            one_minus_beta1_to_t = 1 - self.beta1 ** self.t
            one_minus_beta2_to_t = 1 - self.beta2 ** self.t

            for p, m, v in zip(self.params, self.m_params, self.v_params):
                if p.grad is not None:
                    g = p.grad.data

                    self.weight_decay_fn(g, p)

                    m.mul_(self.beta1).add_(g, alpha=1 - self.beta1)
                    v.mul_(self.beta2).addcmul_(g, g, value=1 - self.beta2)

                    m_hat = m / one_minus_beta1_to_t
                    v_hat = v / one_minus_beta2_to_t
                    update = m_hat / (v_hat.sqrt() + self.eps)

                    # Reshape LR for broadcasting and apply update
                    lr_shape = [self.n_models] + [1] * (p.dim() - 1)
                    lr_tensor = self.learning_rates.view(lr_shape)
                    p.sub_(update * lr_tensor)

    def zero_grad(self):
        for p in self.params:
            p.grad = None
