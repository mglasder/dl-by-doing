import torch
from torch import nn, Generator


class NPLM(nn.Module):
    """
    Neural Probabilistic Language Model

    Implementation based on:
    Bengio et al. 2003 https://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf
    """

    def __init__(
        self,
        batch_size: int,
        hidden_units: int,
        context: int,
        features: int,
        vocab_size: int,
        g: Generator = None,
    ):
        super().__init__()
        self.h = hidden_units
        self.n = context
        self.m = features
        self.v = vocab_size

        self._batch_size = batch_size

        self.C = nn.Parameter(torch.randn((self.v, self.m), generator=g))
        self.H = nn.Parameter(torch.randn((self.n * self.m, self.h), generator=g))
        self.d = nn.Parameter(torch.randn((1, self.h), generator=g))
        self.b = nn.Parameter(torch.randn((1, self.v), generator=g))
        self.U = nn.Parameter(torch.randn((self.h, self.v), generator=g))

    def forward(self, x):
        emb = self.C[x]
        X = emb.view(-1, self.n * self.m)
        h = torch.tanh(self.d + X @ self.H)
        logits = self.b + h @ self.U
        return logits


class NPLMref(nn.Module):
    """
    Reference implementation of the above taken from Andrey Kaparthy's github repo:
    https://github.com/karpathy/makemore/blob/master/makemore.py

    Adapted to input data representation of the above implementation, otherwise kept the same.
    """

    def __init__(self, block_size, vocab_size, n_embd, n_embd2):
        super().__init__()

        self.n_embd = n_embd
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.wte = nn.Embedding(vocab_size, n_embd)
        # removed the +1, since this case does not apply in my implementation
        self.mlp = nn.Sequential(
            nn.Linear(self.block_size * n_embd, n_embd2),
            nn.Tanh(),
            nn.Linear(n_embd2, self.vocab_size),
        )

    def get_block_size(self):
        return self.block_size

    def forward(self, x):
        # adapted to match input data representation as in NPLM above
        embs = self.wte(x)
        x = embs.view(-1, self.block_size * self.n_embd)
        logits = self.mlp(x)
        return logits
