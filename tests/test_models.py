"""Tests for model implementations.

Each model has two tests:
- test_lifecycle: Full roundtrip — create model, train (with tracking=False to
  skip MLflow), predict (check output shape), save to a temp directory, verify
  the directory-based artifact structure (config.json + weights file), load into
  a fresh instance, and confirm predictions match exactly.
- test_get_params: Verify that get_params() returns the expected keys and values,
  including both model-specific args and inherited defaults (e.g. Fabric args).
"""

import json
import os
import tempfile

import numpy as np

from ml_project_template.models.mlp_classifier import MLPClassifier
from ml_project_template.models.gb_classifier import GBClassifier


class TestMLPClassifier:
    def test_lifecycle(self, iris_tiny):
        # Create model: 4 input features → 8 hidden units → 3 output classes
        model = MLPClassifier(layer_dims=[4, 8, 3],
            hidden_activation="ReLU",
            output_activation="Identity",
            use_bias=True
        )

        # Train with tracking=False to avoid needing an MLflow server
        model.train(
            train_data=iris_tiny,
            tracking=False,
            max_epochs=5,
        )

        # MLP returns raw logits — shape is (num_samples, num_classes)
        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20, 3)

        with tempfile.TemporaryDirectory() as tmp:
            saved_path = model.save(f"{tmp}/model")

            # save() should create a directory with config.json and model.pt
            assert os.path.isdir(saved_path)
            assert os.path.exists(f"{tmp}/model/config.json")
            assert os.path.exists(f"{tmp}/model/model.pt")

            # config.json should contain the model name and init params
            with open(f"{tmp}/model/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "mlp_classifier"
            assert config["model_params"]["layer_dims"] == [4, 8, 3]
            assert config["model_params"]["hidden_activation"] == "ReLU"
            assert config["model_params"]["output_activation"] == "Identity"

            model2 = MLPClassifier.load(f"{tmp}/model")

            preds2 = model2.predict(iris_tiny.X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        """Verify auto-captured __init__ params include both architecture and Fabric args."""
        model = MLPClassifier(
            layer_dims=[4, 16, 3],
            hidden_activation="ReLU",
            output_activation="Identity",
            use_bias=True
        )
        params = model.get_params()
        # Architecture params
        assert params["layer_dims"] == [4, 16, 3]
        assert params["hidden_activation"] == "ReLU"
        assert params["output_activation"] == "Identity"
        assert params["use_bias"] == True
        # Inherited Fabric default
        assert params["accelerator"] == "auto"


class TestGBClassifier:
    def test_lifecycle(self, iris_tiny):
        model = GBClassifier(n_estimators=10, max_depth=2)

        # Train with tracking=False to avoid needing an MLflow server
        model.train(
            train_data=iris_tiny,
            tracking=False,
        )

        # GB returns class labels — shape is (num_samples,) with values in {0, 1, 2}
        preds = model.predict(iris_tiny.X)
        assert preds.shape == (20,)
        assert set(preds).issubset({0, 1, 2})

        with tempfile.TemporaryDirectory() as tmp:
            saved_path = model.save(f"{tmp}/model")

            # save() should create a directory with config.json and model.joblib
            assert os.path.isdir(saved_path)
            assert os.path.exists(f"{tmp}/model/config.json")
            assert os.path.exists(f"{tmp}/model/model.joblib")

            # config.json should contain the model name and init params
            with open(f"{tmp}/model/config.json") as f:
                config = json.load(f)
            assert config["model_name"] == "gb_classifier"
            assert config["model_params"]["n_estimators"] == 10
            assert config["model_params"]["max_depth"] == 2

            model2 = GBClassifier.load(f"{tmp}/model")

            preds2 = model2.predict(iris_tiny.X)
            np.testing.assert_array_equal(preds, preds2)

    def test_get_params(self):
        """Verify sklearn's get_params() returns the expected hyperparameters."""
        model = GBClassifier(n_estimators=50, max_depth=3, learning_rate=0.05)
        params = model.get_params()
        assert params["n_estimators"] == 50
        assert params["max_depth"] == 3
        assert params["learning_rate"] == 0.05
