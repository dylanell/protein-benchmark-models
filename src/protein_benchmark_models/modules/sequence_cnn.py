"""1D CNN for processing sequences.

Each conv layer slides a k-mer kernel along the sequence dimension; the
out_channels of each layer become the token embedding dimension fed into
the next layer.

Tensor shapes through the network:
  embedding : [B, seq_len, embed_dim]
  transpose  : [B, embed_dim, seq_len]   ← Conv1d expects (B, C, L)
  conv_1     : [B, out_channels_1, seq_len_1]
  conv_2     : [B, out_channels_2, seq_len_2]
  ...
  flatten    : [B, out_channels_last * seq_len_last]
  linear     : [B, output_dim]
"""

from __future__ import annotations

from typing import Literal

import torch.nn as nn

from .utils import Transpose


class SequenceCNN(nn.Module):
    def __init__(
        self,
        embed_dims: list[int],
        kernel_spec: list[list[int]],
        seq_length: int,
        output_dim: int,
        padding_idx: int = 0,
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
    ):
        """Sequence regressor using stacked 1D convolutions.

        Args:
            embed_dims: [vocab_size, embed_dim]. The embedding table maps each
                token index to a vector of size embed_dim.
            kernel_spec: List of [kernel_size, out_channels, stride] for each
                conv layer. kernel_size is the k-mer width (number of sequence
                positions); out_channels becomes the token embedding dimension
                for the next layer.
            seq_length: Input sequence length. Used to compute the flattened
                size of the CNN output.
            output_dim: Number of output units in the linear head.
            padding_idx: Token index treated as padding (embedding zeroed out).
            hidden_activation: nn activation applied after each hidden layer.
            output_activation: nn activation name applied after the final conv.
            use_bias: Whether to include bias in conv and linear layers.
            norm: "batch" for BatchNorm1d, "layer" for LayerNorm, or None.
        """
        super().__init__()

        self.embedding = nn.Embedding(
            embed_dims[0], embed_dims[1], padding_idx=padding_idx
        )

        cnn_layers = []
        in_channels = embed_dims[1]
        current_seq_len = seq_length

        for i, (kernel_size, out_channels, stride) in enumerate(kernel_spec):
            cnn_layers.append(
                nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    stride=stride,
                    bias=use_bias,
                )
            )

            if i == len(kernel_spec) - 1:
                cnn_layers.append(getattr(nn, output_activation)())
            else:
                # BatchNorm1d operates directly on [B, out_channels, seq_len]
                if norm == "batch":
                    cnn_layers.append(nn.BatchNorm1d(out_channels))
                # LayerNorm normalises over channels at each position, so we
                # need [B, seq_len, out_channels] — transpose, norm, transpose.
                elif norm == "layer":
                    cnn_layers.append(Transpose(1, 2))
                    cnn_layers.append(nn.LayerNorm(out_channels))
                    cnn_layers.append(Transpose(1, 2))
                cnn_layers.append(getattr(nn, hidden_activation)())

            in_channels = out_channels
            current_seq_len = (current_seq_len - kernel_size) // stride + 1

        self.cnn = nn.Sequential(*cnn_layers)

        cnn_output_dim = in_channels * current_seq_len
        self.linear = nn.Linear(cnn_output_dim, output_dim, bias=use_bias)

    def forward(self, x):
        x = self.embedding(x)  # [B, seq_len, embed_dim]
        x = x.transpose(
            1, 2
        )  # [B, embed_dim, seq_len] — Conv1d expects (B, C, L)
        x = self.cnn(x)  # [B, out_channels_last, final_seq_len]
        x = x.flatten(start_dim=1)  # [B, cnn_output_dim]
        x = self.linear(x)  # [B, output_dim]
        return x
