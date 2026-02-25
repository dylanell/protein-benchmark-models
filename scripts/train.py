"""Config-driven training script for Iris classifiers.

Loads preprocessed data (if available) or raw data, then trains
the model specified in the config.

Usage:
    uv run python scripts/train.py --config configs/iris_mlp_classifier.json
    uv run python scripts/train.py --config configs/iris_gb_classifier.json
"""

import argparse
import json
import sys

from dotenv import load_dotenv

from ml_project_template.data import TabularDataset
from ml_project_template.models import ModelRegistry
from ml_project_template.utils import get_storage_options, seed_everything

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Train an Iris classifier from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[train] Running with config:")
    print(json.dumps(config, indent=2))

    # Validate required top-level keys
    for key in ("data", "model", "training"):
        if key not in config:
            print(f"[train] Error: config missing required key '{key}'")
            sys.exit(1)

    # Seed early for reproducible data splitting
    seed = config.get("seed")
    if seed is not None:
        seed_everything(seed)

    # Load data — use preprocessed output if available, otherwise raw
    data_cfg = config["data"]
    preprocess_cfg = config.get("preprocessing", {})
    data_path = preprocess_cfg.get("output_path", data_cfg["path"])
    print(f"\n[train] Loading data from {data_path}")

    storage_options = get_storage_options(data_path)
    dataset = TabularDataset.from_csv(data_path, target_column=data_cfg["target_column"], storage_options=storage_options)
    train_data, val_data = dataset.split(
        test_size=data_cfg.get("test_size", 0.2),
        random_state=seed,
    )

    # Create model
    model_cfg = config["model"]
    model = ModelRegistry.get(model_cfg["name"])(**model_cfg.get("params", {}))

    # Build training args — shallow copy so we don't mutate the loaded config
    train_cfg = dict(config["training"])
    experiment_name = train_cfg.pop("experiment_name")
    run_name = train_cfg.pop("run_name", None)
    model_path = train_cfg.pop("model_path", None)

    # Flatten data + preprocessing config for MLflow logging
    extra_params = {f"data.{k}": v for k, v in data_cfg.items()}
    extra_params.update({f"preprocessing.{k}": v for k, v in preprocess_cfg.items()})

    # Everything remaining is model-specific training kwargs
    model.train(
        experiment_name=experiment_name,
        train_data=train_data,
        val_data=val_data,
        run_name=run_name,
        model_path=model_path,
        extra_params=extra_params,
        seed=seed,
        **train_cfg,
    )

    print(f"[train] Training complete.")


if __name__ == "__main__":
    main()
