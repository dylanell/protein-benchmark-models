"""PyTorch CNN sequence regressor."""

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

from ..data import TokenizedSequenceDataset
from .base import BaseModel
from ..modules.sequence_cnn import SequenceCNN
from ..utils import evaluate_regression


class CNNRegressor(BaseModel):
    """Sequence regressor model using 1D CNNs."""

    model_name = "cnn_regressor"

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
        accelerator: str | Accelerator = "auto",
        strategy: str | Strategy = "auto",
        devices: list[int] | str | int = "auto",
        precision: str | int = "32-true",
        plugins: str | Any | None = None,
        callbacks: list[Any] | Any | None = None,
        loggers: Logger | list[Logger] | None = None,
    ):
        super().__init__()

        self.seq_length = seq_length

        self.model = SequenceCNN(
            embed_dims=embed_dims,
            kernel_spec=kernel_spec,
            seq_length=seq_length,
            output_dim=output_dim,
            padding_idx=padding_idx,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_bias=use_bias,
            norm=norm,
        )

        self.loss_fcn = nn.MSELoss()

        # Initialize fabric
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
        train_data: TokenizedSequenceDataset,
        val_data: TokenizedSequenceDataset,
        *,
        model_path: str,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
    ):
        # Log training parameters
        self.log_param("lr", lr)
        self.log_param("weight_decay", weight_decay)
        self.log_param("batch_size", batch_size)
        self.log_param("seq_length", self.seq_length)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)

        # Initialize optimizer and fabric
        optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=weight_decay
        )
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # Initialize dataloaders
        train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=True
        )
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        val_dataloader = DataLoader(
            val_data, batch_size=batch_size, shuffle=False
        )
        val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        epochs_without_improvement = 0
        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Train
            cum_train_loss = 0
            model.train()
            for batch in train_dataloader:
                X = batch["tokens"]
                y = batch["target"]
                optimizer.zero_grad()
                output = model(X).squeeze()
                loss = torch.sqrt(self.loss_fcn(output, y))
                self.fabric.backward(loss)
                optimizer.step()
                cum_train_loss += loss.item()
            train_loss = cum_train_loss / len(train_dataloader)

            self.log_metric("train_loss", train_loss, step=epoch)

            # Validate
            if (epoch + 1) % val_frequency == 0:
                cum_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        X = batch["tokens"]
                        y = batch["target"]
                        output = model(X).squeeze()
                        loss = torch.sqrt(self.loss_fcn(output, y))
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
                        f"{patience} epochs reached without improvement. Early stopping."
                    )
                    break

            status = (
                f"Epoch: {epoch + 1}/{max_epochs} | train_loss: {train_loss:.4f} | "
                f"val_loss: {val_loss:.4f} | best_val_loss: {best_val_loss:.4f}"
            )
            pbar.set_description(status)

        # Final validation metrics
        X = np.stack(
            [val_data[i]["tokens"].numpy() for i in range(len(val_data))]
        )
        y = val_data.targets.numpy()
        metrics = evaluate_regression(self, X, y)
        for k, v in metrics.items():
            self.log_metric(f"val_{k}", v)
        print(f"[cnn_regressor] Valid RMSE: {metrics['rmse']:.04f}")
        print(f"[cnn_regressor] Valid R2: {metrics['r2']:.04f}")
        print(f"[cnn_regressor] Valid SpearmanR: {metrics['spearmanr']:.04f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on an array of token indices.

        Args:
            X: Int64 array of shape (N, seq_len) containing vocabulary indices.
        """
        self.model.eval()
        X_tensor = torch.from_numpy(X).long().to(self.fabric.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.squeeze(-1).cpu().numpy()

    def _save_weights(self, dir_path: str) -> None:
        """Save model state dict to directory."""
        self.fabric.save(
            os.path.join(dir_path, "model.pt"), {"model": self.model}
        )

    def _load_weights(self, dir_path: str) -> None:
        """Load model state dict from directory."""
        state = {"model": self.model}
        self.fabric.load(os.path.join(dir_path, "model.pt"), state)
