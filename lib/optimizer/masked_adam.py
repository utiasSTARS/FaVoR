import torch
import lib.cuda

''' Extend Adam optimizer
1. support per-voxel learning rate
2. masked update (ignore zero grad) which speeduping training
'''


''' Extend Adam optimizer
1. support per-voxel learning rate
2. masked update (ignore zero grad) which speeduping training
'''
class MaskedAdam(torch.optim.Optimizer):

    def __init__(self, params, lr=1e-3, betas=(0.9, 0.99), eps=1e-8):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        defaults = dict(lr=lr, betas=betas, eps=eps)
        self.per_lr = None
        super(MaskedAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(MaskedAdam, self).__setstate__(state)

    def set_pervoxel_lr(self, count):
        assert self.param_groups[0]['params'][0].shape == count.shape
        self.per_lr = count.float() / count.max()

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            skip_zero_grad = group['skip_zero_grad']

            for param in group['params']:
                if param.grad is not None:
                    state = self.state[param]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        # Exponential moving average of gradient values
                        state['exp_avg'] = torch.zeros_like(param, memory_format=torch.preserve_format)
                        # Exponential moving average of squared gradient values
                        state['exp_avg_sq'] = torch.zeros_like(param, memory_format=torch.preserve_format)

                    state['step'] += 1

                    if self.per_lr is not None and param.shape == self.per_lr.shape:
                        torch.ops.adam_upd.adam_upd_with_perlr(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'], self.per_lr,
                                state['step'], beta1, beta2, lr, eps)
                    elif skip_zero_grad:
                        torch.ops.adam_upd.masked_adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)
                    else:
                        torch.ops.adam_upd.adam_upd(
                                param, param.grad, state['exp_avg'], state['exp_avg_sq'],
                                state['step'], beta1, beta2, lr, eps)
