"""Tests for seed_everything reproducibility."""

import numpy as np
import torch

from ml_project_template.utils import seed_everything
from ml_project_template.models.mlp_classifier import MLPClassifier


class TestSeedEverything:
    def test_numpy_determinism(self):
        """Seeding twice with the same seed produces identical numpy output."""
        seed_everything(42)
        a = np.random.rand(5)

        seed_everything(42)
        b = np.random.rand(5)

        np.testing.assert_array_equal(a, b)

    def test_torch_determinism(self):
        """Seeding twice with the same seed produces identical torch output."""
        seed_everything(42)
        a = torch.randn(5)

        seed_everything(42)
        b = torch.randn(5)

        torch.testing.assert_close(a, b)

    def test_mlp_training_determinism(self, iris_tiny):
        """Two MLP training runs with the same seed produce identical predictions.

        Seeds before model construction (matching the real script flow:
        seed_everything → create model → model.train(seed=...)).
        """
        def train_and_predict(seed):
            seed_everything(seed)
            model = MLPClassifier(layer_dims=[4, 8, 3])
            model.train(
                train_data=iris_tiny,
                tracking=False,
                seed=seed,
                max_epochs=5,
            )
            return model.predict(iris_tiny.X)

        preds1 = train_and_predict(99)
        preds2 = train_and_predict(99)

        np.testing.assert_array_equal(preds1, preds2)
