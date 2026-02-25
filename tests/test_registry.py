"""Tests for model registry.

Tests cover the ModelRegistry API:
- list/get: Discover and retrieve model classes by name.
- get_name: Reverse lookup — get the registry name for a model class.
- load: Full roundtrip — save a model to a directory, then reconstruct it
  from just the directory path (reads config.json, instantiates, loads weights).
"""

import tempfile

import numpy as np
import pytest

from ml_project_template.models import ModelRegistry
from ml_project_template.models.gb_classifier import GBClassifier
from ml_project_template.models.mlp_classifier import MLPClassifier


def test_list():
    """All registered model names should be discoverable."""
    names = ModelRegistry.list()
    assert "gb_classifier" in names
    assert "mlp_classifier" in names


def test_get():
    """get() should return the exact class registered under each name."""
    assert ModelRegistry.get("gb_classifier") is GBClassifier
    assert ModelRegistry.get("mlp_classifier") is MLPClassifier


def test_get_unknown():
    """get() should raise ValueError for unregistered names."""
    with pytest.raises(ValueError, match="Unknown model"):
        ModelRegistry.get("nonexistent")


def test_get_name():
    """get_name() should reverse-lookup the registry name from a model class."""
    assert ModelRegistry.get_name(MLPClassifier) == "mlp_classifier"
    assert ModelRegistry.get_name(GBClassifier) == "gb_classifier"


def test_get_name_unknown():
    """get_name() should raise ValueError for unregistered classes."""
    class FakeModel:
        pass
    with pytest.raises(ValueError, match="not registered"):
        ModelRegistry.get_name(FakeModel)


def test_load_mlp(iris_tiny):
    """ModelRegistry.load() should reconstruct an MLP from a saved directory."""
    model = MLPClassifier(layer_dims=[4, 8, 3])
    model.train(train_data=iris_tiny, tracking=False, max_epochs=5)
    preds = model.predict(iris_tiny.X)

    with tempfile.TemporaryDirectory() as tmp:
        # save() writes config.json + model.pt to the directory
        model.save(f"{tmp}/model")

        # load() reads config.json, instantiates MLPClassifier, loads weights
        loaded = ModelRegistry.load(f"{tmp}/model")

        assert isinstance(loaded, MLPClassifier)
        preds2 = loaded.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)


def test_load_gb(iris_tiny):
    """ModelRegistry.load() should reconstruct a GB classifier from a saved directory."""
    model = GBClassifier(n_estimators=10, max_depth=2)
    model.train(train_data=iris_tiny, tracking=False)
    preds = model.predict(iris_tiny.X)

    with tempfile.TemporaryDirectory() as tmp:
        # save() writes config.json + model.joblib to the directory
        model.save(f"{tmp}/model")

        # load() reads config.json, instantiates GBClassifier, loads weights
        loaded = ModelRegistry.load(f"{tmp}/model")

        assert isinstance(loaded, GBClassifier)
        preds2 = loaded.predict(iris_tiny.X)
        np.testing.assert_array_equal(preds, preds2)
