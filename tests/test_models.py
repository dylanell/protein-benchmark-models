"""Tests for model implementations.

Each model has two tests:
- test_lifecycle: train (with model_path), verify artifact structure, load into
  fresh instance, confirm predictions match exactly.
- test_get_params: verify get_params() returns expected keys and values.
"""

import json
import os
import tempfile

import numpy as np

from protein_benchmark_models.models.ridge_regressor import RidgeRegressor
from protein_benchmark_models.models.mlp_regressor import MLPRegressor
from protein_benchmark_models.models.cnn_regressor import CNNRegressor
from tests.conftest import SEQ_LEN, SEQUENCES, VOCAB_SIZE, onehot_X, token_X

N = len(SEQUENCES)  # 8


class TestRidgeRegressor:
    def test_lifecycle(self, onehot_data):
        with tempfile.TemporaryDirectory() as tmp:
            model = RidgeRegressor(alpha=1.0)
            model.train(train_data=onehot_data, val_data=onehot_data, tracking=False, model_path=f"{tmp}/model")

            assert os.path.isdir(f"{tmp}/model_final")
            assert os.path.exists(f"{tmp}/model_final/config.json")
            assert os.path.exists(f"{tmp}/model_final/model.joblib")

            with open(f"{tmp}/model_final/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "ridge_regressor"

            X = onehot_X(onehot_data)
            preds = model.predict(X)
            assert preds.shape == (N,)

            model2 = RidgeRegressor.load(f"{tmp}/model_final")
            preds2 = model2.predict(X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        model = RidgeRegressor(alpha=2.0)
        params = model.get_params()
        assert params["alpha"] == 2.0


class TestMLPRegressor:
    def test_lifecycle(self, onehot_data):
        with tempfile.TemporaryDirectory() as tmp:
            model = MLPRegressor(layer_dims=[SEQ_LEN * VOCAB_SIZE, 16, 1])
            model.train(train_data=onehot_data, val_data=onehot_data, tracking=False, model_path=f"{tmp}/model", max_epochs=5)

            assert os.path.isdir(f"{tmp}/model_final")
            assert os.path.exists(f"{tmp}/model_final/config.json")
            assert os.path.exists(f"{tmp}/model_final/model.pt")

            with open(f"{tmp}/model_final/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "mlp_regressor"
            assert config["model_params"]["layer_dims"] == [SEQ_LEN * VOCAB_SIZE, 16, 1]

            X = onehot_X(onehot_data)
            preds = model.predict(X)
            assert preds.shape == (N,)

            model2 = MLPRegressor.load(f"{tmp}/model_final")
            preds2 = model2.predict(X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        model = MLPRegressor(layer_dims=[176, 16, 1], hidden_activation="ReLU")
        params = model.get_params()
        assert params["layer_dims"] == [176, 16, 1]
        assert params["hidden_activation"] == "ReLU"
        assert params["accelerator"] == "auto"


class TestCNNRegressor:
    # Single conv layer: kernel_height=3, out_channels=8, stride=1
    # Output seq_len = (SEQ_LEN - 3) // 1 + 1 = 6
    KERNEL_SPEC = [[3, 8, 1]]
    EMBED_DIMS = [VOCAB_SIZE, 8]

    def test_lifecycle(self, tokenized_data):
        with tempfile.TemporaryDirectory() as tmp:
            model = CNNRegressor(
                embed_dims=self.EMBED_DIMS,
                kernel_spec=self.KERNEL_SPEC,
                seq_length=SEQ_LEN,
                output_dim=1,
            )
            model.train(train_data=tokenized_data, val_data=tokenized_data, tracking=False, model_path=f"{tmp}/model", max_epochs=5)

            assert os.path.isdir(f"{tmp}/model_final")
            assert os.path.exists(f"{tmp}/model_final/config.json")
            assert os.path.exists(f"{tmp}/model_final/model.pt")

            with open(f"{tmp}/model_final/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "cnn_regressor"
            assert config["model_params"]["embed_dims"] == self.EMBED_DIMS
            assert config["model_params"]["kernel_spec"] == self.KERNEL_SPEC

            X = token_X(tokenized_data)
            preds = model.predict(X)
            assert preds.shape == (N,)

            model2 = CNNRegressor.load(f"{tmp}/model_final")
            preds2 = model2.predict(X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        model = CNNRegressor(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            output_dim=1,
        )
        params = model.get_params()
        assert params["embed_dims"] == self.EMBED_DIMS
        assert params["kernel_spec"] == self.KERNEL_SPEC
        assert params["seq_length"] == SEQ_LEN
        assert params["output_dim"] == 1
        assert params["accelerator"] == "auto"
