"""Shared nn.Module utilities."""

import torch.nn as nn


class Transpose(nn.Module):
    """Swap two dimensions of a tensor in a Sequential pipeline."""

    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self._dim0 = dim0
        self._dim1 = dim1

    def forward(self, x):
        return x.transpose(self._dim0, self._dim1)
