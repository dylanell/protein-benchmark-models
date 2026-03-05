"""Config-driven training script for protein benchmarking tasks.

Loads pre-split train/valid CSVs, builds the appropriate sequence dataset,
injects data-derived shape params into the model constructor, and trains.

Usage:
    uv run python scripts/train.py --config configs/local/tape_fluorescence_ridge_regressor.json
"""

import argparse
import json
import sys

import pandas as pd
from dotenv import load_dotenv

from protein_benchmark_models.data import OneHotSequenceDataset, TokenizedSequenceDataset
from protein_benchmark_models.models import ModelRegistry
from protein_benchmark_models.utils import get_storage_options

load_dotenv()

DATASET_CLASSES = {
    "one_hot": OneHotSequenceDataset,
    "tokenized": TokenizedSequenceDataset,
}


def run(config: dict) -> None:
    """Core training logic. Accepts a config dict — callable locally or from remote runners."""
    # Validate required top-level keys
    for key in ("data", "model", "training"):
        if key not in config:
            print(f"[train] Error: config missing required key '{key}'")
            sys.exit(1)

    seed = config.get("seed")

    # Load pre-split data
    data_cfg = config["data"]
    train_path = data_cfg["train_path"]
    valid_path = data_cfg["valid_path"]

    print(f"\n[train] Loading train data from {train_path}")
    train_df = pd.read_csv(train_path, storage_options=get_storage_options(train_path))
    print(f"[train] Loading valid data from {valid_path}")
    val_df = pd.read_csv(valid_path, storage_options=get_storage_options(valid_path))

    print(f"[train] Train size: {len(train_df)}, Valid size: {len(val_df)}")

    # Compute seq_len from training sequences
    seq_len = max(len(s) for s in train_df["sequence"])
    print(f"[train] Max sequence length: {seq_len}")

    # Build datasets
    dataset_type = data_cfg["dataset_type"]
    if dataset_type not in DATASET_CLASSES:
        print(f"[train] Error: unknown dataset_type '{dataset_type}'. Choose from: {list(DATASET_CLASSES)}")
        sys.exit(1)

    DatasetClass = DATASET_CLASSES[dataset_type]
    train_dataset = DatasetClass(train_df["sequence"].tolist(), train_df["target"].tolist(), seq_len=seq_len)
    val_dataset = DatasetClass(val_df["sequence"].tolist(), val_df["target"].tolist(), seq_len=seq_len)

    # Build model params — inject data-derived shape params and seed
    model_cfg = config["model"]
    model_params = dict(model_cfg.get("params", {}))
    if seed is not None:
        model_params["seed"] = seed

    if dataset_type == "one_hot":
        if "layer_dims" in model_params:
            input_dim = train_dataset[0]["one_hots"].flatten().shape[0]
            model_params["layer_dims"] = [input_dim] + model_params["layer_dims"]
    elif dataset_type == "tokenized":
        model_params["seq_length"] = seq_len

    print(f"\n[train] Constructing model '{model_cfg['name']}' with params: {model_params}")
    model = ModelRegistry.get(model_cfg["name"])(**model_params)

    # Build training args
    train_cfg = dict(config["training"])
    experiment_name = train_cfg.pop("experiment_name")
    run_name = train_cfg.pop("run_name", None)
    model_path = train_cfg.pop("model_path", None)

    extra_params = {f"data.{k}": v for k, v in data_cfg.items()}
    extra_params["data.seq_len"] = seq_len

    model.train(
        experiment_name=experiment_name,
        train_data=train_dataset,
        val_data=val_dataset,
        run_name=run_name,
        model_path=model_path,
        extra_params=extra_params,
        **train_cfg,
    )

    print(f"[train] Training complete.")


def main():
    parser = argparse.ArgumentParser(description="Train a protein benchmark model from a config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[train] Running with config:")
    print(json.dumps(config, indent=2))

    run(config)


if __name__ == "__main__":
    main()
