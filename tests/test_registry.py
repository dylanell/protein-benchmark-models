"""Tests for model registry.

Covers:
- list/get: Discover and retrieve model classes by name.
- get_name: Reverse lookup — get the registry name for a model class.
- load: Full roundtrip — train a model, then reconstruct it from just the
  directory path (reads config.json, instantiates, loads weights).
"""

import tempfile

import numpy as np
import pytest

from protein_benchmark_models.models import ModelRegistry
from protein_benchmark_models.models.ridge_regressor import RidgeRegressor
from protein_benchmark_models.models.mlp_regressor import MLPRegressor
from protein_benchmark_models.models.cnn_regressor import CNNRegressor
from tests.conftest import SEQ_LEN, VOCAB_SIZE, onehot_X, token_X


def test_list():
    names = ModelRegistry.list()
    assert "ridge_regressor" in names
    assert "mlp_regressor" in names
    assert "cnn_regressor" in names


def test_get():
    assert ModelRegistry.get("ridge_regressor") is RidgeRegressor
    assert ModelRegistry.get("mlp_regressor") is MLPRegressor
    assert ModelRegistry.get("cnn_regressor") is CNNRegressor


def test_get_unknown():
    with pytest.raises(ValueError, match="Unknown model"):
        ModelRegistry.get("nonexistent")


def test_get_name():
    assert ModelRegistry.get_name(RidgeRegressor) == "ridge_regressor"
    assert ModelRegistry.get_name(MLPRegressor) == "mlp_regressor"
    assert ModelRegistry.get_name(CNNRegressor) == "cnn_regressor"


def test_get_name_unknown():
    class FakeModel:
        pass
    with pytest.raises(ValueError, match="not registered"):
        ModelRegistry.get_name(FakeModel)


def test_load_ridge(onehot_data):
    with tempfile.TemporaryDirectory() as tmp:
        model = RidgeRegressor(alpha=1.0)
        model.train(train_data=onehot_data, val_data=onehot_data, tracking=False, model_path=f"{tmp}/model")
        X = onehot_X(onehot_data)
        preds = model.predict(X)

        loaded = ModelRegistry.load(f"{tmp}/model_final")
        assert isinstance(loaded, RidgeRegressor)
        np.testing.assert_array_equal(preds, loaded.predict(X))


def test_load_mlp(onehot_data):
    with tempfile.TemporaryDirectory() as tmp:
        model = MLPRegressor(layer_dims=[SEQ_LEN * VOCAB_SIZE, 16, 1])
        model.train(train_data=onehot_data, val_data=onehot_data, tracking=False, model_path=f"{tmp}/model", max_epochs=5)
        X = onehot_X(onehot_data)
        preds = model.predict(X)

        loaded = ModelRegistry.load(f"{tmp}/model_final")
        assert isinstance(loaded, MLPRegressor)
        np.testing.assert_array_equal(preds, loaded.predict(X))


def test_load_cnn(tokenized_data):
    with tempfile.TemporaryDirectory() as tmp:
        model = CNNRegressor(
            embed_dims=[VOCAB_SIZE, 8],
            kernel_spec=[[3, 8, 1]],
            seq_length=SEQ_LEN,
            output_dim=1,
        )
        model.train(train_data=tokenized_data, val_data=tokenized_data, tracking=False, model_path=f"{tmp}/model", max_epochs=5)
        X = token_X(tokenized_data)
        preds = model.predict(X)

        loaded = ModelRegistry.load(f"{tmp}/model_final")
        assert isinstance(loaded, CNNRegressor)
        np.testing.assert_array_equal(preds, loaded.predict(X))
