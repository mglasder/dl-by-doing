import torch
from torch import nn


class BatchNorm(nn.Module):
    """
    Implementation based on:

    Ioffe and Szegedy (2015) -
    Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift

    https://arxiv.org/abs/1502.03167
    """

    def __init__(self, dim, eps=1e-05):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(dim))
        self.beta = nn.Parameter(torch.zeros(dim))

        self.cum_mu = torch.tensor(0)
        self.cum_var = torch.tensor(0)
        self.m = 0

    def forward(self, x):
        if self.train:
            mu = x.mean(axis=0, keepdims=True)
            var = x.var(axis=0, keepdims=True, unbiased=False)
            x_hat = (x - mu) / torch.sqrt(var + self.eps)
            y = self.gamma * x_hat + self.beta

        else:
            self._update_cum_mu(x)
            self._update_cum_var(x)
            self.m += 1
            sigma = torch.sqrt(self.cum_var + self.eps)
            y = self.gamma / sigma * x + (self.beta - self.gamma * self.cum_mu / sigma)

        return y

    def _update_cum_mu(self, x):
        self.cum_mu = (x.mean(axis=0, keepdims=True) + self.m * self.cum_mu) / (
            self.m + 1
        )

    def _update_cum_var(self, x):
        self.cum_var = (
            self.m
            / (self.m - 1)
            * (x.var(axis=0, keepdims=True, unbiased=False) + self.m * self.cum_var)
            / (self.m + 1)
        )
