"""Ridge regression model using scikit-learn."""

from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

from protein_benchmark_models.data import OneHotSequenceDataset
from protein_benchmark_models.models.base import BaseModel


class RidgeRegressor(BaseModel):
    """Scikit-learn Ridge regression wrapper."""

    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)

    def _fit(
        self,
        train_data: OneHotSequenceDataset,
        val_data: OneHotSequenceDataset | None = None,
        **kwargs
    ) -> None:
        # Stack training data into one array for scikit models
        X = np.stack([train_data[i]["one_hots"].numpy().flatten() for i in range(len(train_data))])
        y = np.stack([train_data[i]["target"].numpy() for i in range(len(train_data))])

        print(f"[ridge_regressor] Train X: {X.shape}")
        print(f"[ridge_regressor] Train y: {y.shape}")

        print(f"[ridge_regressor] Training model")

        # Train model
        self.model.fit(X, y)

        print(f"[ridge_regressor] Evaluating model")

        # Evaluate model on trainig set
        y_pred = self.predict(X)
        train_rmse = np.sqrt(np.mean((y - y_pred)**2))
        train_r2 = r2_score(y, y_pred)
        self.log_metric("train_rmse", train_rmse)
        self.log_metric("train_r2", train_r2)

        print(f"[ridge_regressor] Train RMSE: {train_rmse:.04f}")
        print(f"[ridge_regressor] Train R2: {train_r2:.04f}")

        if val_data is not None:
            # Stack valid data into one array for scikit models
            X = np.stack([val_data[i]["one_hots"].numpy().flatten() for i in range(len(val_data))])
            y = np.stack([val_data[i]["target"].numpy() for i in range(len(val_data))])

            print(f"[ridge_regressor] Valid X: {X.shape}")
            print(f"[ridge_regressor] Valid y: {y.shape}")

            # Evaluate model on valid set
            y_pred = self.predict(X)
            val_rmse = np.sqrt(np.mean((y - y_pred)**2))
            val_r2 = r2_score(y, y_pred)
            self.log_metric("val_rmse", val_rmse)
            self.log_metric("val_r2", val_r2)

            print(f"[ridge_regressor] Valid RMSE: {val_rmse:.04f}")
            print(f"[ridge_regressor] Valid R2: {val_r2:.04f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_weights(self, dir_path: str) -> None:
        """Save model to directory."""
        joblib.dump(self.model, os.path.join(dir_path, "model.joblib"))

    def _load_weights(self, dir_path: str) -> None:
        """Load model from directory."""
        self.model = joblib.load(os.path.join(dir_path, "model.joblib"))

    # Override: BaseModel auto-captures __init__ args, but since this class
    # takes **kwargs, it would record {"kwargs": {...}} (a nested dict).
    # Sklearn's get_params() gives us a flat dict of all params with defaults.
    def get_params(self) -> dict:
        return self.model.get_params()
