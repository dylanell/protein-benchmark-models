"""Tests for the serving endpoint.

Tests use FastAPI's TestClient (via httpx) to make real HTTP requests against
the app without needing a running server. feature_names and class_names are
passed directly to create_app() to skip CSV/S3 loading in tests.

Helpers:
- _make_config: Build a minimal config dict matching the training config format.
- _train_and_save_mlp/_gb: Train a small model, save to a temp dir, return config.
"""

import tempfile

import numpy as np
from fastapi.testclient import TestClient

from ml_project_template.models.mlp_classifier import MLPClassifier
from ml_project_template.models.gb_classifier import GBClassifier
from ml_project_template.serving.app import create_app


def _make_config(model_name, model_params, model_path):
    """Build a minimal config dict in the same format as training JSON configs."""
    return {
        "data": {"path": "unused", "target_column": "species"},
        "model": {"name": model_name, "params": model_params},
        "training": {"model_path": model_path},
    }


def _train_and_save_mlp(iris_tiny, tmp_dir):
    """Train a small MLP, save to tmp_dir, return a config pointing to it."""
    model = MLPClassifier(layer_dims=[4, 8, 3])
    model.train(train_data=iris_tiny, tracking=False, max_epochs=5)
    model.save(f"{tmp_dir}/model")
    return _make_config("mlp_classifier", {"layer_dims": [4, 8, 3]}, f"{tmp_dir}/model")


def _train_and_save_gb(iris_tiny, tmp_dir):
    """Train a small GB classifier, save to tmp_dir, return a config pointing to it."""
    model = GBClassifier(n_estimators=10, max_depth=2)
    model.train(train_data=iris_tiny, tracking=False)
    model.save(f"{tmp_dir}/model")
    return _make_config("gb_classifier", {"n_estimators": 10, "max_depth": 2}, f"{tmp_dir}/model")


class TestHealth:
    def test_health(self, iris_tiny):
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.get("/health")
            assert resp.status_code == 200
            assert resp.json() == {"status": "ok"}


class TestInfo:
    def test_info(self, iris_tiny):
        """Should return model name, params, feature names, and class names."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.get("/info")
            assert resp.status_code == 200
            data = resp.json()
            assert data["model_name"] == "mlp_classifier"
            assert data["feature_names"] == ["f0", "f1", "f2", "f3"]
            assert data["class_names"] == ["a", "b", "c"]
            assert data["model_params"]["layer_dims"] == [4, 8, 3]


class TestPredict:
    def test_predict_mlp(self, iris_tiny):
        """MLP should return (n, num_classes) predictions."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0, 4.0], [5.0, 6.0, 7.0, 8.0]]})
            assert resp.status_code == 200
            preds = resp.json()["predictions"]
            assert len(preds) == 2
            assert len(preds[0]) == 3  # 3 classes — MLP returns raw logits

    def test_predict_gb(self, iris_tiny):
        """GB should return (n, 1) predictions (class labels reshaped to 2D)."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_gb(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            resp = client.post("/predict", json={"features": [[1.0, 2.0, 3.0, 4.0]]})
            assert resp.status_code == 200
            preds = resp.json()["predictions"]
            assert len(preds) == 1
            assert len(preds[0]) == 1  # GB returns scalar labels, normalized to (n, 1)

    def test_predict_invalid_features(self, iris_tiny):
        """Should return 422 when feature count doesn't match the model."""
        with tempfile.TemporaryDirectory() as tmp:
            config = _train_and_save_mlp(iris_tiny, tmp)
            app = create_app(config, feature_names=["f0", "f1", "f2", "f3"], class_names=["a", "b", "c"])
            client = TestClient(app)
            # Send 2 features when model expects 4
            resp = client.post("/predict", json={"features": [[1.0, 2.0]]})
            assert resp.status_code == 422
            assert "expected 4" in resp.json()["detail"]
