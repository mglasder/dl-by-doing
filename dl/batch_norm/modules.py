import torch
from torch import nn


class BatchNorm(nn.Module):
    """
    Implementation based on:

    Ioffe and Szegedy (2015) -
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    https://arxiv.org/abs/1502.03167
    """

    def __init__(self, dim, momentum=0.1, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.cum_mu = torch.zeros(dim)
        self.cum_var = torch.ones(dim)
        self.n_batches = 0

    def forward(self, x):
        if self.training:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True, unbiased=False)
            x_hat = (x - mu) / torch.sqrt(var + self.eps)
            y = self.gamma * x_hat + self.beta

            self._update_cum_mu(mu)
            self._update_cum_var(var)
            self.n_batches += 1

        else:
            x_hat = (x - self.cum_mu) / torch.sqrt(self.cum_var + self.eps)
            y = self.gamma * x_hat + self.beta
            # sigma = torch.sqrt(self.cum_var + self.eps)
            # y = self.gamma / sigma * x + (self.beta - self.gamma * self.cum_mu / sigma)

        return y

    def _update_cum_mu(self, mu):
        if self.momentum:
            self.cum_mu = self.momentum * mu + (1 - self.momentum) * self.cum_mu
        else:
            self.cum_mu = self.cum_mu + (mu - self.cum_mu) / (self.n_batches + 1)

    def _update_cum_var(self, var):
        if self.momentum:
            self.cum_var = self.momentum * var + (1 - self.momentum) * self.cum_var
        else:
            self.cum_var = self.cum_var + (var - self.cum_var) / (self.n_batches + 1)
