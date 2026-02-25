"""Tests for TabularDataset.

Tests cover dataset operations that don't require external services:
- split: Train/test splitting preserves metadata and sample counts.
- to_pytorch: Conversion to PyTorch DataLoader produces correct tensor shapes.
- from_csv: Loading from a local CSV works even with S3 env vars set
  (regression test for get_storage_options bug).
"""

import os
import tempfile

import torch

from ml_project_template.data import TabularDataset


def test_split(iris_tiny):
    """Splitting should partition samples while preserving feature/class metadata."""
    train, test = iris_tiny.split(test_size=0.3, random_state=0)
    assert len(train) == 14
    assert len(test) == 6
    assert train.feature_names == iris_tiny.feature_names
    assert test.class_names == iris_tiny.class_names


def test_to_pytorch(iris_tiny):
    """to_pytorch() should produce float32 feature tensors and int64 label tensors."""
    loader = iris_tiny.to_pytorch(batch_size=10, shuffle=False)
    X_batch, y_batch = next(iter(loader))
    assert isinstance(X_batch, torch.Tensor)
    assert X_batch.shape == (10, 4)
    assert y_batch.shape == (10,)


def test_from_csv_local_with_s3_env_set(iris_tiny, monkeypatch):
    """Loading a local CSV should work even when S3 env vars are set.

    Regression test: get_storage_options() previously checked whether
    S3_ENDPOINT_URL was set (not whether the path was S3), so local file
    loading would fail when S3 env vars were present.
    """
    # Simulate a local dev environment with MinIO credentials set
    monkeypatch.setenv("S3_ENDPOINT_URL", "http://localhost:7000")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "fake")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "fake")

    with tempfile.TemporaryDirectory() as tmp:
        # Write a small CSV from the fixture data
        import pandas as pd
        csv_path = os.path.join(tmp, "data.csv")
        df = pd.DataFrame(iris_tiny.X, columns=iris_tiny.feature_names)
        df["target"] = iris_tiny.y
        df.to_csv(csv_path, index=False)

        # This should succeed despite S3 env vars being set
        dataset = TabularDataset.from_csv(csv_path, target_column="target")
        assert len(dataset) == 20
        assert dataset.feature_names == iris_tiny.feature_names
