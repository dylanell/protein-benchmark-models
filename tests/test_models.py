"""Tests for model implementations.

Each model has two tests:
- test_lifecycle: train (with model_path), verify artifact structure, load into
  fresh instance, confirm predictions match exactly.
- test_get_params: verify get_params() returns expected keys and values.
"""

import json
import os

import numpy as np
import pytest

from protein_benchmark_models.models.ridge_regressor import RidgeRegressor
from protein_benchmark_models.models.mlp_regressor import MLPRegressor
from protein_benchmark_models.models.cnn_regressor import CNNRegressor
from protein_benchmark_models.models.siamese_mlp_classifier import (
    SiameseMLPClassifier,
)
from protein_benchmark_models.models.siamese_cnn_classifier import (
    SiameseCNNClassifier,
)
from protein_benchmark_models.models.siamese_cnn_w_att_classifier import (
    SiameseCNNWithAttClassifier,
)
from tests.conftest import (
    SEQ_LEN,
    SEQUENCES,
    VOCAB_SIZE,
    BINARY_TARGETS,
    onehot_X,
    token_X,
    paired_onehot_X,
    paired_token_X,
)

N = len(SEQUENCES)  # 8


class TestRidgeRegressor:
    # tmp_path is a built-in pytest fixture that provides a temporary directory
    # unique to each test invocation, automatically cleaned up afterwards.
    def test_lifecycle(self, onehot_data, tmp_path):
        model = RidgeRegressor(alpha=1.0)
        model.train(
            train_data=onehot_data,
            val_data=onehot_data,
            tracking=False,
            model_path=str(tmp_path / "model"),
        )

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.joblib")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "ridge_regressor"

        X = onehot_X(onehot_data)
        preds = model.predict(X)
        assert preds.shape == (N,)

        model2 = RidgeRegressor.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = RidgeRegressor(alpha=2.0)
        assert model.config["alpha"] == 2.0


class TestMLPRegressor:
    LAYER_DIMS = [SEQ_LEN * VOCAB_SIZE, 16, 1]

    def test_lifecycle(self, onehot_data, tmp_path):
        model = MLPRegressor(layer_dims=self.LAYER_DIMS)
        model.train(
            train_data=onehot_data,
            val_data=onehot_data,
            tracking=False,
            model_path=str(tmp_path / "model"),
            max_epochs=5,
        )

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.pt")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "mlp_regressor"
        assert config["model_params"]["layer_dims"] == self.LAYER_DIMS

        X = onehot_X(onehot_data)
        preds = model.predict(X)
        assert preds.shape == (N,)

        model2 = MLPRegressor.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = MLPRegressor(
            layer_dims=self.LAYER_DIMS, hidden_activation="ReLU"
        )
        assert model.config["layer_dims"] == self.LAYER_DIMS
        assert model.config["hidden_activation"] == "ReLU"
        assert model.config["accelerator"] == "auto"


class TestCNNRegressor:
    # Single conv layer: kernel_height=3, out_channels=8, stride=1
    # Output seq_len = (SEQ_LEN - 3) // 1 + 1 = 6
    KERNEL_SPEC = [[3, 8, 1]]
    EMBED_DIMS = [VOCAB_SIZE, 8]

    def test_lifecycle(self, tokenized_data, tmp_path):
        model = CNNRegressor(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
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

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.pt")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "cnn_regressor"
        assert config["model_params"]["embed_dims"] == self.EMBED_DIMS
        assert config["model_params"]["kernel_spec"] == self.KERNEL_SPEC

        X = token_X(tokenized_data)
        preds = model.predict(X)
        assert preds.shape == (N,)

        model2 = CNNRegressor.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = CNNRegressor(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            output_dim=1,
        )
        assert model.config["embed_dims"] == self.EMBED_DIMS
        assert model.config["kernel_spec"] == self.KERNEL_SPEC
        assert model.config["seq_length"] == SEQ_LEN
        assert model.config["output_dim"] == 1
        assert model.config["accelerator"] == "auto"


class TestSiameseMLPClassifier:
    # encoder: SEQ_LEN*VOCAB_SIZE → 16; head: 32 → 1
    ENCODER_DIMS = [SEQ_LEN * VOCAB_SIZE, 16]
    HEAD_DIMS = [32, 1]

    def test_lifecycle(self, paired_onehot_data, tmp_path):
        model = SiameseMLPClassifier(
            encoder_dims=self.ENCODER_DIMS,
            head_dims=self.HEAD_DIMS,
        )
        model.train(
            train_data=paired_onehot_data,
            val_data=paired_onehot_data,
            tracking=False,
            model_path=str(tmp_path / "model"),
            max_epochs=3,
        )

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.pt")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "siamese_mlp_classifier"
        assert config["model_params"]["encoder_dims"] == self.ENCODER_DIMS
        assert config["model_params"]["head_dims"] == self.HEAD_DIMS

        X_a, X_b = paired_onehot_X(paired_onehot_data)
        preds = model.predict(X_a, X_b)
        assert preds.shape == (N,)
        assert ((preds >= 0.0) & (preds <= 1.0)).all()

        model2 = SiameseMLPClassifier.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X_a, X_b)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = SiameseMLPClassifier(
            encoder_dims=self.ENCODER_DIMS,
            head_dims=self.HEAD_DIMS,
            hidden_activation="LeakyReLU",
        )
        assert model.config["encoder_dims"] == self.ENCODER_DIMS
        assert model.config["head_dims"] == self.HEAD_DIMS
        assert model.config["hidden_activation"] == "LeakyReLU"
        assert model.config["accelerator"] == "auto"

    def test_invalid_head_dims_raises(self):
        with pytest.raises(ValueError, match="head_dims\\[0\\]"):
            SiameseMLPClassifier(
                encoder_dims=self.ENCODER_DIMS,
                head_dims=[99, 1],  # should be 32
            )

    def test_invalid_head_output_raises(self):
        with pytest.raises(ValueError, match="head_dims\\[-1\\]"):
            SiameseMLPClassifier(
                encoder_dims=self.ENCODER_DIMS,
                head_dims=[32, 2],  # should be 1
            )


class TestSiameseCNNClassifier:
    # Single conv layer: kernel_height=3, out_channels=8, stride=1
    # Output seq_len = (SEQ_LEN - 3) // 1 + 1 = 6
    KERNEL_SPEC = [[3, 8, 1]]
    EMBED_DIMS = [VOCAB_SIZE, 8]
    OUTPUT_DIM = 16
    HEAD_DIMS = [32, 1]  # 2 * OUTPUT_DIM = 32

    def test_lifecycle(self, paired_tokenized_data, tmp_path):
        model = SiameseCNNClassifier(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            output_dim=self.OUTPUT_DIM,
            head_dims=self.HEAD_DIMS,
        )
        model.train(
            train_data=paired_tokenized_data,
            val_data=paired_tokenized_data,
            tracking=False,
            model_path=str(tmp_path / "model"),
            max_epochs=3,
        )

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.pt")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "siamese_cnn_classifier"
        assert config["model_params"]["embed_dims"] == self.EMBED_DIMS
        assert config["model_params"]["kernel_spec"] == self.KERNEL_SPEC
        assert config["model_params"]["output_dim"] == self.OUTPUT_DIM
        assert config["model_params"]["head_dims"] == self.HEAD_DIMS

        X_a, X_b = paired_token_X(paired_tokenized_data)
        preds = model.predict(X_a, X_b)
        assert preds.shape == (N,)
        assert ((preds >= 0.0) & (preds <= 1.0)).all()

        model2 = SiameseCNNClassifier.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X_a, X_b)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = SiameseCNNClassifier(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            output_dim=self.OUTPUT_DIM,
            head_dims=self.HEAD_DIMS,
        )
        assert model.config["embed_dims"] == self.EMBED_DIMS
        assert model.config["kernel_spec"] == self.KERNEL_SPEC
        assert model.config["seq_length"] == SEQ_LEN
        assert model.config["output_dim"] == self.OUTPUT_DIM
        assert model.config["head_dims"] == self.HEAD_DIMS
        assert model.config["accelerator"] == "auto"

    def test_invalid_head_dims_raises(self):
        with pytest.raises(ValueError, match="head_dims\\[0\\]"):
            SiameseCNNClassifier(
                embed_dims=self.EMBED_DIMS,
                kernel_spec=self.KERNEL_SPEC,
                seq_length=SEQ_LEN,
                output_dim=self.OUTPUT_DIM,
                head_dims=[99, 1],  # should be 32
            )

    def test_invalid_head_output_raises(self):
        with pytest.raises(ValueError, match="head_dims\\[-1\\]"):
            SiameseCNNClassifier(
                embed_dims=self.EMBED_DIMS,
                kernel_spec=self.KERNEL_SPEC,
                seq_length=SEQ_LEN,
                output_dim=self.OUTPUT_DIM,
                head_dims=[32, 2],  # should be 1
            )


class TestSiameseCNNWithAttClassifier:
    # Single conv layer: kernel_size=3, out_channels=8, stride=1
    # token_dim = 8, final_seq_len = (8 - 3) // 1 + 1 = 6
    # head input dim (2 * 6 * 8 = 96) is prepended automatically
    KERNEL_SPEC = [[3, 8, 1]]
    EMBED_DIMS = [VOCAB_SIZE, 8]
    NUM_HEADS = 2
    HEAD_DIMS = [16, 1]

    def test_lifecycle(self, paired_tokenized_data, tmp_path):
        model = SiameseCNNWithAttClassifier(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            num_attention_heads=self.NUM_HEADS,
            head_dims=self.HEAD_DIMS,
        )
        model.train(
            train_data=paired_tokenized_data,
            val_data=paired_tokenized_data,
            tracking=False,
            model_path=str(tmp_path / "model"),
            max_epochs=3,
        )

        assert os.path.isdir(tmp_path / "model_final")
        assert os.path.exists(tmp_path / "model_final" / "config.json")
        assert os.path.exists(tmp_path / "model_final" / "model.pt")

        with open(tmp_path / "model_final" / "config.json") as f:
            config = json.load(f)
        assert config["model_name"] == "siamese_cnn_w_att_classifier"
        assert config["model_params"]["embed_dims"] == self.EMBED_DIMS
        assert config["model_params"]["kernel_spec"] == self.KERNEL_SPEC
        assert config["model_params"]["num_attention_heads"] == self.NUM_HEADS
        assert config["model_params"]["head_dims"] == self.HEAD_DIMS

        X_a, X_b = paired_token_X(paired_tokenized_data)
        preds = model.predict(X_a, X_b)
        assert preds.shape == (N,)
        assert ((preds >= 0.0) & (preds <= 1.0)).all()

        model2 = SiameseCNNWithAttClassifier.load(str(tmp_path / "model_final"))
        preds2 = model2.predict(X_a, X_b)
        np.testing.assert_array_equal(preds, preds2)

    def test_config(self):
        model = SiameseCNNWithAttClassifier(
            embed_dims=self.EMBED_DIMS,
            kernel_spec=self.KERNEL_SPEC,
            seq_length=SEQ_LEN,
            num_attention_heads=self.NUM_HEADS,
            head_dims=self.HEAD_DIMS,
            attention_dropout=0.1,
        )
        assert model.config["embed_dims"] == self.EMBED_DIMS
        assert model.config["kernel_spec"] == self.KERNEL_SPEC
        assert model.config["seq_length"] == SEQ_LEN
        assert model.config["num_attention_heads"] == self.NUM_HEADS
        assert model.config["head_dims"] == self.HEAD_DIMS
        assert model.config["attention_dropout"] == 0.1
        assert model.config["accelerator"] == "auto"

    def test_bad_num_heads_raises(self):
        with pytest.raises(ValueError, match="token_dim"):
            SiameseCNNWithAttClassifier(
                embed_dims=self.EMBED_DIMS,
                kernel_spec=self.KERNEL_SPEC,  # token_dim = 8
                seq_length=SEQ_LEN,
                num_attention_heads=3,  # 8 not divisible by 3
                head_dims=self.HEAD_DIMS,
            )

    def test_bad_head_output_raises(self):
        with pytest.raises(ValueError, match="head_dims\\[-1\\]"):
            SiameseCNNWithAttClassifier(
                embed_dims=self.EMBED_DIMS,
                kernel_spec=self.KERNEL_SPEC,
                seq_length=SEQ_LEN,
                num_attention_heads=self.NUM_HEADS,
                head_dims=[96, 2],  # should be 1
            )
