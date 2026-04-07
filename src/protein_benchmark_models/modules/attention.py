"""Attention modules for sequence interaction tasks."""

from __future__ import annotations

import torch
import torch.nn as nn


class CrossAttention(nn.Module):
    """Multi-head cross-attention between two sequences.

    The query comes from one sequence while the keys and values come from
    another, allowing each sequence to attend over the other's representation.

    Wraps nn.MultiheadAttention with batch_first=True for consistency with
    (batch, sequence, feature) tensor layouts throughout this codebase.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
    ):
        """
        Args:
            embed_dim: Dimension of query, key, and value embeddings.
            num_heads: Number of parallel attention heads. embed_dim must be
                divisible by num_heads.
            dropout: Dropout probability applied to attention weights.
        """
        super().__init__()
        self.attn = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(
        self, query: torch.Tensor, context: torch.Tensor
    ) -> torch.Tensor:
        """Compute cross-attention from query sequence over context sequence.

        Args:
            query: Tensor of shape (B, L_q, embed_dim) providing queries.
            context: Tensor of shape (B, L_k, embed_dim) providing keys and
                values.

        Returns:
            Tensor of shape (B, L_q, embed_dim) — query updated by attending
            over context.
        """
        out, _ = self.attn(query, context, context)
        return out
