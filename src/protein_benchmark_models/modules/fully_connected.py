"""Fully connected pytorch modules."""

from __future__ import annotations

from typing import List, Literal

import torch.nn as nn


class FullyConnected(nn.Module):
    def __init__(
        self,
        layer_dims: List[int],
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
    ):
        super().__init__()

        layers = []
        for i in range(len(layer_dims) - 1):
            layers.append(nn.Linear(layer_dims[i], layer_dims[i + 1], bias=use_bias))

            if i == len(layer_dims) - 2:
                layers.append(getattr(nn, output_activation)())
            else:
                if norm == "batch":
                    layers.append(nn.BatchNorm1d(layer_dims[i + 1]))
                elif norm == "layer":
                    layers.append(nn.LayerNorm(layer_dims[i + 1]))
                layers.append(getattr(nn, hidden_activation)())

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)
