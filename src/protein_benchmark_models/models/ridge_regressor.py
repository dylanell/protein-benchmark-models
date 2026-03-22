"""Ridge regression model using scikit-learn."""

from __future__ import annotations

import os
import joblib
import numpy as np
from sklearn.linear_model import Ridge

from ..data import OneHotSequenceDataset
from .base import BaseModel
from ..utils import evaluate_regression


class RidgeRegressor(BaseModel):
    """Scikit-learn Ridge regression wrapper."""

    model_name = "ridge_regressor"

    def __init__(self, **kwargs):
        self.model = Ridge(**kwargs)

    def _fit(
        self,
        train_data: OneHotSequenceDataset,
        val_data: OneHotSequenceDataset,
        **kwargs
    ) -> None:
        # Stack training data into one array for scikit models
        X = np.stack([train_data[i]["one_hots"].numpy().flatten() for i in range(len(train_data))])
        y = np.stack([train_data[i]["target"].numpy() for i in range(len(train_data))])

        print(f"[ridge_regressor] Train X: {X.shape}")
        print(f"[ridge_regressor] Train y: {y.shape}")
        print("[ridge_regressor] Training model")

        self.model.fit(X, y)

        # Final validation metrics
        X = np.stack([val_data[i]["one_hots"].numpy().flatten() for i in range(len(val_data))])
        y = val_data.targets.numpy()
        metrics = evaluate_regression(self, X, y)
        for k, v in metrics.items():
            self.log_metric(f"val_{k}", v)
        print(f"[ridge_regressor] Valid RMSE: {metrics['rmse']:.04f}")
        print(f"[ridge_regressor] Valid R2: {metrics['r2']:.04f}")
        print(f"[ridge_regressor] Valid SpearmanR: {metrics['spearmanr']:.04f}")

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.model.predict(X)

    def _save_weights(self, dir_path: str) -> None:
        """Save model to directory."""
        joblib.dump(self.model, os.path.join(dir_path, "model.joblib"))

    def _load_weights(self, dir_path: str) -> None:
        """Load model from directory."""
        self.model = joblib.load(os.path.join(dir_path, "model.joblib"))

