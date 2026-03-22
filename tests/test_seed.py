"""Tests for seed_everything reproducibility."""

import numpy as np
import torch

from protein_benchmark_models.utils import seed_everything
from protein_benchmark_models.models.mlp_regressor import MLPRegressor
from tests.conftest import SEQ_LEN, VOCAB_SIZE, onehot_X


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

    # tmp_path is a built-in pytest fixture that provides a temporary directory
    # unique to each test invocation, automatically cleaned up afterwards.
    def test_mlp_training_determinism(self, onehot_data, tmp_path):
        """Two MLP training runs with the same seed produce identical predictions."""
        X = onehot_X(onehot_data)

        def train_and_predict(seed, path):
            seed_everything(seed)
            model = MLPRegressor(layer_dims=[SEQ_LEN * VOCAB_SIZE, 16, 1])
            model.train(
                train_data=onehot_data,
                val_data=onehot_data,
                tracking=False,
                model_path=str(path / "model"),
                max_epochs=5,
            )
            return model.predict(X)

        preds1 = train_and_predict(99, tmp_path / "run1")
        preds2 = train_and_predict(99, tmp_path / "run2")

        np.testing.assert_array_equal(preds1, preds2)
