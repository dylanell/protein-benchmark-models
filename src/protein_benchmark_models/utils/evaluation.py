"""Evaluation utilities for benchmark tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score
from scipy.stats import spearmanr


def evaluate_regression(model, X: np.ndarray, y: np.ndarray) -> dict:
    """Compute regression metrics for a model on a pre-built input array.

    Args:
        model: A trained BaseModel instance.
        X: Input array of shape (N, ...) ready to pass to model.predict().
        y: Target array of shape (N,).

    Returns:
        Dict with keys "rmse", "r2", "spearmanr".
    """
    y_pred = model.predict(X)
    return {
        "rmse": float(np.sqrt(np.mean((y - y_pred) ** 2))),
        "r2": float(r2_score(y, y_pred)),
        "spearmanr": float(spearmanr(y, y_pred).statistic),
    }
