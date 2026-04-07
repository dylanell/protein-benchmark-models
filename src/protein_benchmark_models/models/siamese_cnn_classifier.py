"""Siamese CNN classifier for paired-sequence tasks (e.g. PPI)."""

from __future__ import annotations

from typing import Literal, Any

import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ..data import PairedTokenizedSequenceDataset
from .base import BasePairedModel
from .registry import register
from ..modules.sequence_cnn import SequenceCNN
from ..modules.fully_connected import FullyConnected
from ..utils import evaluate_classification


class _SiameseCNNNet(nn.Module):
    """Combined shared CNN encoder + classification head.

    Wraps both modules as a single nn.Module for Lightning Fabric
    compatibility. The same SequenceCNN encoder is applied to each sequence
    independently; the two output embeddings are concatenated and passed
    through the classification head.
    """

    def __init__(self, encoder: nn.Module, head: nn.Module):
        super().__init__()
        self.encoder = encoder
        self.head = head

    def forward(
        self, tokens_a: torch.Tensor, tokens_b: torch.Tensor
    ) -> torch.Tensor:
        enc_a = self.encoder(tokens_a)
        enc_b = self.encoder(tokens_b)
        return self.head(torch.cat([enc_a, enc_b], dim=-1)).squeeze(-1)


@register
class SiameseCNNClassifier(BasePairedModel):
    """Siamese 1D-CNN binary classifier for paired-sequence inputs.

    Architecture:
        - A shared SequenceCNN encoder is applied independently to each
          tokenized sequence, producing a fixed-length embedding per sequence.
        - The two embeddings are concatenated and passed through a
          FullyConnected classification head that produces a single logit.
        - Training uses BCEWithLogitsLoss; predict() returns sigmoid
          probabilities.

    The encoder is identical in structure to CNNRegressor's backbone, reusing
    the same embed_dims / kernel_spec / seq_length / output_dim convention.
    output_dim is the encoder embedding dimension; head_dims[0] must equal
    2 * output_dim (the concatenated size).
    """

    model_name = "siamese_cnn_classifier"

    def __init__(
        self,
        embed_dims: list[int],
        kernel_spec: list[list[int]],
        seq_length: int,
        output_dim: int,
        head_dims: list[int],
        padding_idx: int = 0,
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: list[int] | str | int = "auto",
        precision: str | int = "32-true",
        plugins: str | Any | None = None,
        callbacks: list[Any] | Any | None = None,
        loggers: Logger | list[Logger] | None = None,
    ):
        """
        Args:
            embed_dims: [vocab_size, embed_dim] for the embedding table.
                embed_dims[0] must equal len(AA_VOCAB) = 22.
            kernel_spec: List of [kernel_size, out_channels, stride] for each
                conv layer.
            seq_length: Input sequence length (injected by train.py).
            output_dim: Output dimension of the CNN encoder (the final linear
                layer inside SequenceCNN). head_dims[0] must equal
                2 * output_dim.
            head_dims: Layer dimensions for the classification head.
                head_dims[0] must equal 2 * output_dim. head_dims[-1] must
                equal 1 (single logit output).
            padding_idx: Token index treated as padding (embedding zeroed).
            hidden_activation: Activation after each hidden layer.
            output_activation: Activation after the final head layer. Should
                be "Identity" when using BCEWithLogitsLoss.
            use_bias: Whether to include bias in linear and conv layers.
            norm: Optional normalization after each hidden layer.
            accelerator: Lightning Fabric accelerator.
            strategy: Lightning Fabric strategy.
            devices: Lightning Fabric devices.
            precision: Lightning Fabric precision.
            plugins: Lightning Fabric plugins.
            callbacks: Lightning Fabric callbacks.
            loggers: Lightning Fabric loggers.
        """
        super().__init__()

        if head_dims[0] != 2 * output_dim:
            raise ValueError(
                f"head_dims[0] must equal 2 * output_dim ({2 * output_dim}), "
                f"got {head_dims[0]}"
            )
        if head_dims[-1] != 1:
            raise ValueError(
                f"head_dims[-1] must equal 1 (single logit), "
                f"got {head_dims[-1]}"
            )

        encoder = SequenceCNN(
            embed_dims=embed_dims,
            kernel_spec=kernel_spec,
            seq_length=seq_length,
            output_dim=output_dim,
            padding_idx=padding_idx,
            hidden_activation=hidden_activation,
            output_activation=hidden_activation,  # encoder output is hidden
            use_bias=use_bias,
            norm=norm,
        )
        head = FullyConnected(
            layer_dims=head_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_bias=use_bias,
            norm=norm,
        )
        self.model = _SiameseCNNNet(encoder, head)

        self.loss_fcn = nn.BCEWithLogitsLoss()

        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers,
        )

    def _fit(
        self,
        train_data: PairedTokenizedSequenceDataset,
        val_data: PairedTokenizedSequenceDataset,
        *,
        model_path: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
    ) -> None:

        self.log_param("lr", lr)
        self.log_param("weight_decay", weight_decay)
        self.log_param("batch_size", batch_size)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)

        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        model, optimizer = self.fabric.setup(self.model, optimizer)

        train_dataloader = self.fabric.setup_dataloaders(
            DataLoader(train_data, batch_size=batch_size, shuffle=True)
        )
        val_dataloader = self.fabric.setup_dataloaders(
            DataLoader(val_data, batch_size=batch_size, shuffle=False)
        )

        epochs_without_improvement = 0
        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            cum_train_loss = 0
            model.train()
            for batch in train_dataloader:
                optimizer.zero_grad()
                logits = model(batch["tokens_a"], batch["tokens_b"])
                loss = self.loss_fcn(logits, batch["target"])
                self.fabric.backward(loss)
                optimizer.step()
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(train_dataloader)
            self.log_metric("train_loss", train_loss, step=epoch)

            if (epoch + 1) % val_frequency == 0:
                cum_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        logits = model(batch["tokens_a"], batch["tokens_b"])
                        loss = self.loss_fcn(logits, batch["target"])
                        cum_val_loss += loss.item()
                val_loss = cum_val_loss / len(val_dataloader)
                self.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    self.save(model_path + "_best")
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    print(
                        f"{patience} epochs without improvement."
                        " Early stopping."
                    )
                    break

            pbar.set_description(
                f"Epoch: {epoch + 1}/{max_epochs}"
                f" | train_loss: {train_loss:.4f}"
                f" | val_loss: {val_loss:.4f}"
                f" | best_val_loss: {best_val_loss:.4f}"
            )

        # Final validation metrics
        X_a = np.stack(
            [val_data[i]["tokens_a"].numpy() for i in range(len(val_data))]
        )
        X_b = np.stack(
            [val_data[i]["tokens_b"].numpy() for i in range(len(val_data))]
        )
        y = val_data.targets.numpy()
        metrics = evaluate_classification(self, X_a, X_b, y)
        for k, v in metrics.items():
            self.log_metric(f"val_{k}", v)
        print(
            f"[siamese_cnn_classifier] Valid AUC:  {metrics['auc']:.4f}"
        )
        print(
            f"[siamese_cnn_classifier] Valid AUPR: {metrics['aupr']:.4f}"
        )

    def predict(self, X_a: np.ndarray, X_b: np.ndarray) -> np.ndarray:
        """Run inference on paired token index arrays.

        Args:
            X_a: Int64 array of shape (N, seq_len) containing vocabulary
                indices for the first sequence of each pair.
            X_b: Int64 array of shape (N, seq_len) containing vocabulary
                indices for the second sequence of each pair.

        Returns:
            Sigmoid probabilities of shape (N,).
        """
        self.model.eval()
        device = self.fabric.device
        t_a = torch.from_numpy(X_a).long().to(device)
        t_b = torch.from_numpy(X_b).long().to(device)
        with torch.no_grad():
            logits = self.model(t_a, t_b)
        return torch.sigmoid(logits).cpu().numpy()

    def _save_weights(self, dir_path: str) -> None:
        self.fabric.save(
            os.path.join(dir_path, "model.pt"), {"model": self.model}
        )

    def _load_weights(self, dir_path: str) -> None:
        state = {"model": self.model}
        self.fabric.load(os.path.join(dir_path, "model.pt"), state)
