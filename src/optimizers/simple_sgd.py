import torch
from torch.optim import Optimizer


class SimpleSGD(Optimizer):
    def __init__(
        self,
        params,
        lr: float=1e-3,
    ) -> None:

        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))

        defaults = dict(lr=lr)
        super(SimpleSGD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SimpleSGD, self).__setstate__(state)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            # groupはdict型
            lr = group['lr']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                p.data.add_(d_p, alpha=-lr)

        return loss
