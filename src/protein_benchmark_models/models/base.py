"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
import functools
import inspect
import json
import os
import tempfile
from typing import ClassVar
import numpy as np

import mlflow
from torch.utils.data import Dataset


class BaseModel(ABC):
    """Abstract base class for all models.

    Subclasses must define a class-level `model_name` string that matches
    the key used to register them in ModelRegistry.

    Uses the template method pattern: train() handles MLflow orchestration
    and delegates to _fit() for model-specific training logic.

    Constructor arguments are captured automatically into self.config after
    __init__ runs. These are logged to MLflow when train() is called.
    """

    model_name: ClassVar[str]

    # --- Automatic __init__ arg capture ---
    # When a subclass defines __init__, this hook wraps it so that all arguments
    # are recorded into self.config after the original __init__ runs.
    # This works across the full inheritance chain: e.g. MLPRegressor.__init__
    # captures both fabric args (from the super().__init__ call) and its own
    # architecture args in a single flat dict.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap classes that define their own __init__
        if "__init__" not in cls.__dict__:
            return

        original_init = cls.__dict__["__init__"]

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kw):
            # Run the original __init__ (which calls super().__init__(), so
            # parent-level params get captured first)
            original_init(self, *args, **kw)

            # Bind the actual call args to the __init__ signature to get
            # a complete dict of param names -> values (including defaults)
            sig = inspect.signature(original_init)
            bound = sig.bind(self, *args, **kw)
            bound.apply_defaults()

            # Build a flat dict of params, unpacking **kwargs if present
            # (e.g. __init__(self, **kwargs) -> unpack kwargs into individual
            # keys rather than storing {"kwargs": {...}})
            params = {}
            for k, v in bound.arguments.items():
                if k == "self":
                    continue
                if sig.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                    params.update(v)
                else:
                    params[k] = v

            # Merge this level's params into config (parent params
            # were already set by the parent's wrapped __init__)
            if not hasattr(self, "config"):
                self.config = {}
            self.config.update(params)

        cls.__init__ = wrapped_init

    def log_param(self, key: str, value) -> None:
        """Log a parameter to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_param(key, value)

    def log_metric(
        self, key: str, value: float, step: int | None = None
    ) -> None:
        """Log a metric to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_metric(key, value, step=step)

    def train(
        self,
        *,
        experiment_name: str = "",
        train_data: Dataset,
        val_data: Dataset,
        run_name: str | None = None,
        model_path: str,
        extra_params: dict | None = None,
        tracking: bool = True,
        **train_kwargs,
    ) -> None:
        """Full training pipeline with optional MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
            train_data: Training dataset
            val_data: Validation dataset
            run_name: Optional MLflow run name
            model_path: Base path for model artifacts. Two directories are
                written: model_path + "_final" (always, after training
                completes) and model_path + "_best" (iterative models only,
                updated whenever val loss improves). Both are logged to MLflow
                when tracking is enabled.
            extra_params: Optional extra params to log (e.g. data config)
            tracking: Whether to enable MLflow tracking (default True)
            **train_kwargs: Model-specific training arguments passed to _fit()
        """
        self._tracking = tracking

        # Resolve the local path that _fit() will use for checkpointing.
        #
        # _fit() only ever writes to a local filesystem path — it has no
        # knowledge of S3. When the final destination is a local path, we pass
        # it through unchanged. When the final destination is an S3 path, we
        # create a temporary local directory for _fit() to checkpoint into,
        # then do a single upload to S3 after training completes. This avoids
        # paying the cost of an S3 write on every checkpoint.
        #
        # ExitStack lets us conditionally enter the TemporaryDirectory context
        # only for the S3 case, while keeping a single unified code path below.
        with contextlib.ExitStack() as stack:
            if model_path.startswith("s3://"):
                tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                checkpoint_path = os.path.join(
                    tmp_dir, os.path.basename(model_path)
                )
            else:
                checkpoint_path = model_path

            # Forward the resolved checkpoint path to _fit() so iterative
            # models can save best-val-loss checkpoints to
            # checkpoint_path + "_best".
            fit_kwargs = dict(train_kwargs)
            fit_kwargs["model_path"] = checkpoint_path

            if not tracking:
                self._fit(train_data, val_data=val_data, **fit_kwargs)
                self.save(checkpoint_path + "_final")
                if checkpoint_path != model_path:
                    from ..utils import get_s3_filesystem

                    fs = get_s3_filesystem()
                    fs.put(
                        checkpoint_path + "_final",
                        model_path + "_final",
                        recursive=True,
                    )
                    if os.path.exists(checkpoint_path + "_best"):
                        fs.put(
                            checkpoint_path + "_best",
                            model_path + "_best",
                            recursive=True,
                        )
                return

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                # Log architecture params (auto-captured from __init__) and
                # model_name.
                mlflow.log_params(
                    {"model_name": self.model_name, **self.config}
                )

                # Log any caller-supplied params (e.g. data config).
                if extra_params:
                    mlflow.log_params(extra_params)

                # Training hyperparams (lr, batch_size, etc.) are logged
                # manually inside _fit() via self.log_param().
                self._fit(train_data, val_data=val_data, **fit_kwargs)

                self.save(checkpoint_path + "_final")
                if checkpoint_path != model_path:
                    from ..utils import get_s3_filesystem

                    fs = get_s3_filesystem()
                    fs.put(
                        checkpoint_path + "_final",
                        model_path + "_final",
                        recursive=True,
                    )
                    if os.path.exists(checkpoint_path + "_best"):
                        fs.put(
                            checkpoint_path + "_best",
                            model_path + "_best",
                            recursive=True,
                        )
                mlflow.log_artifact(checkpoint_path + "_final")
                if os.path.exists(checkpoint_path + "_best"):
                    mlflow.log_artifact(checkpoint_path + "_best")

    @abstractmethod
    def _fit(self, train_data: Dataset, val_data: Dataset, **kwargs) -> None:
        """Model-specific training logic. Subclasses implement this."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input features."""
        raise NotImplementedError

    def save(self, path: str) -> str:
        """Save model to directory with config.json and weights."""
        os.makedirs(path, exist_ok=True)

        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(
                {"model_name": self.model_name, "model_params": self.config},
                f,
                indent=2,
                default=str,
            )

        self._save_weights(path)
        return path

    @abstractmethod
    def _save_weights(self, dir_path: str) -> None:
        """Save model weights to directory. Subclasses implement this."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> BaseModel:
        """Load a model from a saved directory."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        instance = cls(**config["model_params"])
        instance._load_weights(path)
        return instance

    @abstractmethod
    def _load_weights(self, dir_path: str) -> None:
        """Load model weights from directory. Subclasses implement this."""
        raise NotImplementedError


class BasePairedModel(BaseModel):
    """Abstract base class for models that take paired sequence inputs.

    Extends BaseModel with a predict signature that accepts two input arrays
    (one per sequence), suitable for protein-protein interaction tasks.

    Subclasses must implement predict(X_a, X_b) instead of predict(X).
    All other BaseModel machinery (train, save, load, MLflow) is unchanged.
    """

    @abstractmethod
    def predict(  # type: ignore[override]
        self, X_a: np.ndarray, X_b: np.ndarray
    ) -> np.ndarray:
        """Run inference on paired inputs.

        Args:
            X_a: Input array for the first sequence of each pair, shape
                (N, ...).
            X_b: Input array for the second sequence of each pair, shape
                (N, ...).

        Returns:
            Predictions of shape (N,).
        """
        raise NotImplementedError
