"""Base model interfaces."""

from __future__ import annotations

from abc import ABC, abstractmethod
import contextlib
import functools
import inspect
import json
import os
import tempfile
from typing import Literal
import numpy as np

import mlflow

from ml_project_template.data import BaseDataset


class BaseModel(ABC):
    """Abstract base class for all models.

    Uses the template method pattern: train() handles MLflow orchestration
    and delegates to _fit() for model-specific training logic.

    Model params are captured automatically: any argument passed to __init__
    at any level of the inheritance chain is recorded in self._model_params.
    These are logged to MLflow automatically when train() is called.
    Subclasses do NOT need to manually build param dicts or override get_params().
    """

    # --- Automatic __init__ arg capture ---
    # When a subclass defines __init__, this hook wraps it so that all arguments
    # are recorded into self._model_params after the original __init__ runs.
    # This works across the full inheritance chain: BasePytorchModel.__init__
    # captures fabric args, then MLPClassifier.__init__ merges in architecture args.
    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)

        # Only wrap classes that define their own __init__
        if '__init__' not in cls.__dict__:
            return

        original_init = cls.__dict__['__init__']

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
            # (e.g. __init__(self, **kwargs) -> unpack kwargs into individual keys
            # rather than storing {"kwargs": {...}})
            params = {}
            for k, v in bound.arguments.items():
                if k == 'self':
                    continue
                if sig.parameters[k].kind == inspect.Parameter.VAR_KEYWORD:
                    params.update(v)
                else:
                    params[k] = v

            # Merge this level's params into _model_params (parent params
            # were already set by the parent's wrapped __init__)
            if not hasattr(self, '_model_params'):
                self._model_params = {}
            self._model_params.update(params)

        cls.__init__ = wrapped_init

    def get_params(self) -> dict:
        """Return model parameters for logging. Automatically populated from __init__ args."""
        return self._model_params

    def log_param(self, key: str, value) -> None:
        """Log a parameter to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_param(key, value)

    def log_metric(self, key: str, value: float, step: int | None = None) -> None:
        """Log a metric to MLflow if tracking is enabled."""
        if self._tracking:
            mlflow.log_metric(key, value, step=step)

    def train(
        self,
        *,
        experiment_name: str = "",
        train_data: BaseDataset,
        val_data: BaseDataset | None = None,
        run_name: str | None = None,
        model_path: str | None = None,
        extra_params: dict | None = None,
        tracking: bool = True,
        seed: int | None = None,
        save_model: Literal["best", "final"] | None = None,
        **train_kwargs,
    ) -> None:
        """Full training pipeline with optional MLflow tracking.

        Args:
            experiment_name: MLflow experiment name
            train_data: Training dataset
            val_data: Optional validation dataset
            run_name: Optional MLflow run name
            model_path: Optional path to save model artifact
            extra_params: Optional extra params to log (e.g. data/preprocessing config)
            tracking: Whether to enable MLflow tracking (default True)
            seed: Optional random seed for reproducibility (seeds all libraries before training)
            save_model: When to save during training. "best" saves on each new best val loss;
                "final" saves once after training completes. Requires model_path to be set.
                When set, _fit() owns saving and train() skips its own post-training save.
            **train_kwargs: Model-specific training arguments passed to _fit()
        """
        self._tracking = tracking

        if save_model is not None and model_path is None:
            raise ValueError("save_model requires model_path to be set.")

        # Seed Python, NumPy, and PyTorch random number generators before training
        # so that weight initialisation, data shuffling, and dropout are all reproducible.
        if seed is not None:
            from ml_project_template.utils import seed_everything
            seed_everything(seed)

        # Resolve the local path that _fit() will use for checkpointing.
        #
        # _fit() only ever writes to a local filesystem path — it has no knowledge of S3.
        # When the final destination is a local path, we pass it through unchanged.
        # When the final destination is an S3 path, we create a temporary local directory
        # for _fit() to checkpoint into, then do a single upload to S3 after training
        # completes. This avoids paying the cost of an S3 write on every checkpoint
        # (which could be dozens of times for save_model="best").
        #
        # ExitStack lets us conditionally enter the TemporaryDirectory context only for
        # the S3 case, while keeping a single unified code path below.
        with contextlib.ExitStack() as stack:
            if save_model is not None and model_path.startswith("s3://"):
                tmp_dir = stack.enter_context(tempfile.TemporaryDirectory())
                checkpoint_path = os.path.join(tmp_dir, os.path.basename(model_path))
            else:
                # Local destination: _fit() writes directly to the final path.
                checkpoint_path = model_path

            # Forward save_model and the resolved checkpoint_path to _fit() so the
            # training loop knows when and where to write checkpoints.
            fit_kwargs = dict(train_kwargs)
            if save_model is not None:
                fit_kwargs["save_model"] = save_model
                fit_kwargs["model_path"] = checkpoint_path

            if not tracking:
                # Run training without MLflow. After _fit() completes, handle saving:
                # - save_model=None: _fit() didn't save; save the final model now.
                # - save_model set + S3 destination: _fit() saved to temp dir; upload to S3.
                # - save_model set + local destination: _fit() already saved to model_path.
                self._fit(train_data, val_data=val_data, **fit_kwargs)
                if model_path and save_model is None:
                    self.save(model_path)
                elif save_model is not None and checkpoint_path != model_path:
                    from ml_project_template.utils import get_s3_filesystem
                    get_s3_filesystem().put(checkpoint_path, model_path, recursive=True)
                return

            mlflow.set_experiment(experiment_name)
            with mlflow.start_run(run_name=run_name):
                # Log seed so it's recorded alongside the run for reproducibility.
                if seed is not None:
                    mlflow.log_param("seed", seed)

                # Log architecture params (auto-captured from __init__ by BaseModel).
                mlflow.log_params(self.get_params())

                # Log any caller-supplied params (e.g. data config, preprocessing settings).
                if extra_params:
                    mlflow.log_params(extra_params)

                # Run model-specific training. Training hyperparams (lr, batch_size, etc.)
                # are logged manually inside _fit() via self.log_param().
                self._fit(train_data, val_data=val_data, **fit_kwargs)

                # Save the model artifact and log it to MLflow. Three cases:
                #
                # 1. save_model is set: _fit() already wrote the checkpoint to checkpoint_path.
                #    If the final destination is S3, upload it now (one upload after training).
                #    Then log the local checkpoint dir to MLflow.
                #
                # 2. save_model is None + S3 destination: save to a temp dir and upload.
                #    (_save_to_s3 handles the save → upload → return local path dance.)
                #
                # 3. save_model is None + local destination: save directly and log.
                if model_path:
                    if save_model is not None:
                        if checkpoint_path != model_path:
                            from ml_project_template.utils import get_s3_filesystem
                            get_s3_filesystem().put(checkpoint_path, model_path, recursive=True)
                        mlflow.log_artifact(checkpoint_path)
                    elif model_path.startswith("s3://"):
                        with tempfile.TemporaryDirectory() as tmp_dir:
                            saved_path = self._save_to_s3(model_path, tmp_dir)
                            mlflow.log_artifact(saved_path)
                    else:
                        saved_path = self.save(model_path)
                        mlflow.log_artifact(saved_path)

    def _save_to_s3(self, s3_path: str, tmp_dir: str) -> str:
        """Save model to a temp dir, upload to S3, return local path for MLflow logging."""
        from ml_project_template.utils import get_s3_filesystem
        fs = get_s3_filesystem()
        local_path = os.path.join(tmp_dir, os.path.basename(s3_path))
        saved_path = self.save(local_path)
        fs.put(saved_path, s3_path, recursive=True)
        return saved_path

    @abstractmethod
    def _fit(self, train_data: Dataset, val_data: Dataset | None = None, **kwargs) -> None:
        """Model-specific training logic. Subclasses implement this."""
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference on input features."""
        raise NotImplementedError

    def save(self, path: str) -> str:
        """Save model to a directory with config.json and weights. Returns the directory path."""
        from ml_project_template.models.registry import ModelRegistry

        os.makedirs(path, exist_ok=True)

        config = {
            "model_name": ModelRegistry.get_name(type(self)),
            "model_params": self.get_params(),
        }
        with open(os.path.join(path, "config.json"), "w") as f:
            json.dump(config, f, indent=2, default=str)

        self._save_weights(path)
        return path

    @abstractmethod
    def _save_weights(self, dir_path: str) -> None:
        """Save model weights to directory. Subclasses implement this."""
        raise NotImplementedError

    @classmethod
    def load(cls, path: str) -> BaseModel:
        """Load a model from a saved directory. Returns a new instance with weights loaded."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        instance = cls(**config["model_params"])
        instance._load_weights(path)
        return instance

    @abstractmethod
    def _load_weights(self, dir_path: str) -> None:
        """Load model weights from directory. Subclasses implement this."""
        raise NotImplementedError
