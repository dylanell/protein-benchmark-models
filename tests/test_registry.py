"""Tests for model registry.

Covers:
- list/get: Discover and retrieve model classes by name.
- model_name: Each model class declares its own registry name.
- load: Full roundtrip — train a model, then reconstruct it from just the
  directory path (reads config.json, instantiates, loads weights).
"""

import numpy as np
import pytest

from protein_benchmark_models.models import ModelRegistry
from protein_benchmark_models.models.ridge_regressor import RidgeRegressor
from protein_benchmark_models.models.mlp_regressor import MLPRegressor
from protein_benchmark_models.models.cnn_regressor import CNNRegressor
from tests.conftest import SEQ_LEN, VOCAB_SIZE, onehot_X, token_X

_MLP_LAYER_DIMS = [SEQ_LEN * VOCAB_SIZE, 16, 1]
_CNN_EMBED_DIMS = [VOCAB_SIZE, 8]
_CNN_KERNEL_SPEC = [[3, 8, 1]]


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


def test_model_name():
    assert RidgeRegressor.model_name == "ridge_regressor"
    assert MLPRegressor.model_name == "mlp_regressor"
    assert CNNRegressor.model_name == "cnn_regressor"


# tmp_path is a built-in pytest fixture that provides a temporary directory
# unique to each test invocation, automatically cleaned up afterwards.
def test_load_ridge(onehot_data, tmp_path):
    model = RidgeRegressor(alpha=1.0)
    model.train(
        train_data=onehot_data,
        val_data=onehot_data,
        tracking=False,
        model_path=str(tmp_path / "model"),
    )
    X = onehot_X(onehot_data)
    preds = model.predict(X)

    loaded = ModelRegistry.load(str(tmp_path / "model_final"))
    assert isinstance(loaded, RidgeRegressor)
    np.testing.assert_array_equal(preds, loaded.predict(X))


def test_load_mlp(onehot_data, tmp_path):
    model = MLPRegressor(layer_dims=_MLP_LAYER_DIMS)
    model.train(
        train_data=onehot_data,
        val_data=onehot_data,
        tracking=False,
        model_path=str(tmp_path / "model"),
        max_epochs=5,
    )
    X = onehot_X(onehot_data)
    preds = model.predict(X)

    loaded = ModelRegistry.load(str(tmp_path / "model_final"))
    assert isinstance(loaded, MLPRegressor)
    np.testing.assert_array_equal(preds, loaded.predict(X))


def test_load_cnn(tokenized_data, tmp_path):
    model = CNNRegressor(
        embed_dims=_CNN_EMBED_DIMS,
        kernel_spec=_CNN_KERNEL_SPEC,
        seq_length=SEQ_LEN,
        output_dim=1,
    )
    model.train(
        train_data=tokenized_data,
        val_data=tokenized_data,
        tracking=False,
        model_path=str(tmp_path / "model"),
        max_epochs=5,
    )
    X = token_X(tokenized_data)
    preds = model.predict(X)

    loaded = ModelRegistry.load(str(tmp_path / "model_final"))
    assert isinstance(loaded, CNNRegressor)
    np.testing.assert_array_equal(preds, loaded.predict(X))
