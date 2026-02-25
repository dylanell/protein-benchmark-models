"""PyTorch MLP classifier."""

from __future__ import annotations

from typing import Union, Any, Optional, List, Literal

import os
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import lightning as L
from lightning.fabric.accelerators import Accelerator
from lightning.fabric.loggers import Logger
from lightning.fabric.strategies import Strategy

from ml_project_template.data import TabularDataset
from ml_project_template.models.base import BaseModel
from ml_project_template.modules.fully_connected import FullyConnected


class MLPClassifier(BaseModel):
    """Simple 2-layer MLP classifier."""

    def __init__(
        self,
        layer_dims: List[int],
        hidden_activation: str = "ReLU",
        output_activation: str = "Identity",
        use_bias: bool = True,
        norm: Literal["batch", "layer"] | None = None,
        accelerator: Union[str, Accelerator] = "auto",
        strategy: Union[str, Strategy] = "auto",
        devices: Union[list[int], str, int] = "auto",
        precision: Union[str, int] = "32-true",
        plugins: Optional[Union[str, Any]] = None,
        callbacks: Optional[Union[list[Any], Any]] = None,
        loggers: Optional[Union[Logger, list[Logger]]] = None
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

        self.loss_fcn = nn.CrossEntropyLoss()

    def _fit(
        self,
        train_data: TabularDataset,
        val_data: Optional[TabularDataset] = None,
        *,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        batch_size: int = 32,
        max_epochs: int = 100,
        val_frequency: int = 1,
        patience: int = -1,
        save_model: Optional[str] = None,
        model_path: Optional[str] = None,
    ) -> None:
        if patience > 0 and val_data is None:
            raise ValueError("Patience requires a validation dataset.")
        if save_model == "best" and val_data is None:
            raise ValueError("save_model='best' requires a validation dataset.")

        # Initialize optimizer and fabric
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        model, optimizer = self.fabric.setup(self.model, optimizer)

        # Initialize dataloaders
        train_dataloader = train_data.to_pytorch(batch_size=batch_size, shuffle=True)
        train_dataloader = self.fabric.setup_dataloaders(train_dataloader)
        if val_data is not None:
            val_dataloader = val_data.to_pytorch(batch_size=batch_size, shuffle=False)
            val_dataloader = self.fabric.setup_dataloaders(val_dataloader)

        epochs_without_improvement = 0
        val_loss = float("inf")
        best_val_loss = float("inf")

        pbar = tqdm(range(max_epochs))
        for epoch in pbar:
            # Train
            cum_train_loss = 0
            model.train()
            for X_batch, y_batch in train_dataloader:
                optimizer.zero_grad()
                output = model(X_batch)
                loss = self.loss_fcn(output, y_batch)
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
                    for X_batch, y_batch in val_dataloader:
                        output = model(X_batch)
                        loss = self.loss_fcn(output, y_batch)
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

        if save_model == "final":
            self.save(model_path)

        # Log training parameters
        self.log_param("lr", lr)
        self.log_param("weight_decay", weight_decay)
        self.log_param("batch_size", batch_size)
        self.log_param("max_epochs", max_epochs)
        self.log_param("val_frequency", val_frequency)
        self.log_param("patience", patience)

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
        