"""PyTorch MLP regressor."""

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
from sklearn.metrics import r2_score
from scipy.stats import spearmanr

from protein_benchmark_models.data import OneHotSequenceDataset
from protein_benchmark_models.models.base import BaseModel
from protein_benchmark_models.modules.fully_connected import FullyConnected


class MLPRegressor(BaseModel):
    """MLP regressor backed by a FullyConnected module."""

    def __init__(
        self,
        layer_dims: list[int],
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
        loggers: Logger | list[Logger] | None = None
    ):
        super().__init__()

        # Initialize fabric
        self.fabric = L.Fabric(
            accelerator=accelerator,
            strategy=strategy,
            devices=devices,
            precision=precision,
            plugins=plugins,
            callbacks=callbacks,
            loggers=loggers
        )

        self.model = FullyConnected(
            layer_dims=layer_dims,
            hidden_activation=hidden_activation,
            output_activation=output_activation,
            use_bias=use_bias,
            norm=norm,
        )

        self.loss_fcn = nn.MSELoss()

    def _fit(
        self,
        train_data: OneHotSequenceDataset,
        val_data: OneHotSequenceDataset | None = None,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
        save_model: str | None = None,
        model_path: str | None = None,
    ) -> None:
        if patience > 0 and val_data is None:
            raise ValueError("Patience requires a validation dataset.")
        if save_model == "best" and val_data is None:
            raise ValueError("save_model='best' requires a validation dataset.")

        # Log training parameters before training starts
        self.log_param("lr", lr)
        self.log_param("weight_decay", weight_decay)
        self.log_param("batch_size", batch_size)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)

        # Initialize optimizer and fabric
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # Initialize dataloaders
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_data is not None:
            val_dataloader = DataLoader(val_data, batch_size=batch_size, shuffle=False)
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
                X = batch["one_hots"].flatten(start_dim=1)
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
            if (val_data is not None) and ((epoch + 1) % val_frequency == 0):
                cum_val_loss = 0
                model.eval()
                with torch.no_grad():
                    for batch in val_dataloader:
                        X = batch["one_hots"].flatten(start_dim=1)
                        y = batch["target"]
                        output = model(X).squeeze()
                        loss = torch.sqrt(self.loss_fcn(output, y))
                        cum_val_loss += loss.item()
                val_loss = cum_val_loss / len(val_dataloader)

                self.log_metric("val_loss", val_loss, step=epoch)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    epochs_without_improvement = 0
                    if save_model == "best":
                        self.save(model_path)
                else:
                    epochs_without_improvement += 1

                if epochs_without_improvement == patience:
                    print(f"{patience} epochs reached without improvement. Early stopping.")
                    break

            status = f"Epoch: {epoch+1}/{max_epochs} | train_loss: {train_loss:.4f} | "\
                f"val_loss: {val_loss:.4f} | best_val_loss: {best_val_loss:.4f}"
            pbar.set_description(status)

        # Final validation and metrics
        if val_data is not None:
            model.eval()

            y_pred = []
            y = []
            with torch.no_grad():
                for batch in val_dataloader:
                    X_batch = batch["one_hots"].flatten(start_dim=1)
                    y_batch = batch["target"]
                    y.append(y_batch)
                    y_pred_batch = model(X_batch).squeeze()
                    y_pred.append(y_pred_batch)

            y_pred = torch.concat(y_pred).detach().cpu().numpy()
            y = torch.concat(y).detach().cpu().numpy()
            
            val_rmse = np.sqrt(np.mean((y - y_pred)**2))
            val_r2 = r2_score(y, y_pred)
            val_spearmanr = spearmanr(y, y_pred).statistic
            self.log_metric("val_rmse", val_rmse)
            self.log_metric("val_r2", val_r2)
            self.log_metric("val_spearmanr", val_spearmanr)

        if save_model == "final":
            self.save(model_path)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference. Returns raw model output as numpy array."""
        self.model.eval()
        X_tensor = torch.from_numpy(X).float().to(self.fabric.device)
        with torch.no_grad():
            output = self.model(X_tensor)
        return output.cpu().numpy()

    def _save_weights(self, dir_path: str) -> None:
        """Save model state dict to directory."""
        self.fabric.save(os.path.join(dir_path, "model.pt"), {"model": self.model})

    def _load_weights(self, dir_path: str) -> None:
        """Load model state dict from directory."""
        state = {"model": self.model}
        self.fabric.load(os.path.join(dir_path, "model.pt"), state)
