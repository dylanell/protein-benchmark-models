"""Evaluation utilities for benchmark tasks."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import r2_score, roc_auc_score, average_precision_score
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


def evaluate_classification(
    model, X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray
) -> dict:
    """Compute binary classification metrics for a paired-input model.

    Args:
        model: A trained BasePairedModel instance.
        X_a: Input array for the first sequence of each pair, shape (N, ...).
        X_b: Input array for the second sequence of each pair, shape (N, ...).
        y: Binary target array of shape (N,) with values 0 or 1.

    Returns:
        Dict with keys "auc" (ROC-AUC) and "aupr" (average precision).
    """
    y_pred = model.predict(X_a, X_b)
    return {
        "auc": float(roc_auc_score(y, y_pred)),
        "aupr": float(average_precision_score(y, y_pred)),
    }
