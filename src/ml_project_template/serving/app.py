"""FastAPI app factory for model serving."""

import os
import tempfile

import numpy as np
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from ml_project_template.data import TabularDataset
from ml_project_template.models import ModelRegistry
from ml_project_template.utils import get_storage_options


class PredictRequest(PydanticBaseModel):
    features: list[list[float]]


class PredictResponse(PydanticBaseModel):
    predictions: list[list[float]]


def create_app(config: dict, feature_names: list[str] | None = None, class_names: list[str] | None = None) -> FastAPI:
    """Create a FastAPI app for serving a trained model.

    Args:
        config: Parsed JSON config (same format as training configs).
        feature_names: Override feature names (skips CSV load when provided with class_names).
        class_names: Override class names (skips CSV load when provided with feature_names).
    """
    # Load dataset metadata if not provided
    if feature_names is None or class_names is None:
        data_cfg = config["data"]
        dataset = TabularDataset.from_csv(
            data_cfg["path"],
            target_column=data_cfg["target_column"],
            storage_options=get_storage_options(data_cfg["path"]),
        )
        feature_names = feature_names or dataset.feature_names
        class_names = class_names or dataset.class_names

    # Create and load model
    model_cfg = config["model"]
    model_path = config["training"]["model_path"]

    if model_path.startswith("s3://"):
        from ml_project_template.utils import get_s3_filesystem
        fs = get_s3_filesystem()

        tmp_dir = tempfile.mkdtemp()
        local_path = os.path.join(tmp_dir, os.path.basename(model_path))
        fs.get(model_path, local_path, recursive=True)
        model = ModelRegistry.load(local_path)
        print(f"[serve] Loaded model from {model_path}")
    else:
        model = ModelRegistry.load(model_path)
        print(f"[serve] Loaded model from {model_path}")

    num_features = len(feature_names)

    app = FastAPI(title="Iris Classifier API")

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get("/info")
    def info():
        return {
            "model_name": model_cfg["name"],
            "model_params": model_cfg.get("params", {}),
            "feature_names": feature_names,
            "class_names": class_names,
        }

    @app.post("/predict", response_model=PredictResponse)
    def predict(request: PredictRequest):
        for i, sample in enumerate(request.features):
            if len(sample) != num_features:
                raise HTTPException(
                    status_code=422,
                    detail=f"Sample {i} has {len(sample)} features, expected {num_features}: {feature_names}",
                )

        X = np.array(request.features, dtype=np.float32)
        preds = model.predict(X)

        # Normalize to 2D
        if preds.ndim == 1:
            preds = preds.reshape(-1, 1)

        return PredictResponse(predictions=preds.tolist())

    return app
