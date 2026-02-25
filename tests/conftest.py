"""Shared test fixtures."""

import numpy as np
import pytest

from ml_project_template.data import TabularDataset


@pytest.fixture
def iris_tiny():
    """Tiny 20-sample, 4-feature, 3-class dataset built from numpy arrays."""
    rng = np.random.default_rng(42)
    X = rng.standard_normal((20, 4)).astype(np.float32)
    y = np.array([0, 1, 2] * 6 + [0, 1], dtype=np.int64)
    return TabularDataset(
        X=X,
        y=y,
        feature_names=["f0", "f1", "f2", "f3"],
        class_names=["a", "b", "c"],
    )
