"""Config-driven preprocessing script for the Iris dataset.

Scales feature columns and writes a processed CSV for downstream training.
Supports "standardize" (zero mean, unit variance) and "min-max" (0-1 range).

Usage:
    uv run python scripts/preprocess.py --config configs/iris_mlp_classifier.json
"""

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ml_project_template.utils import get_storage_options, seed_everything

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Preprocess Iris dataset from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[preprocess] Running with config:")
    print(json.dumps(config, indent=2))

    # Validate required keys
    for key in ("data", "preprocessing"):
        if key not in config:
            print(f"[preprocess] Error: config missing required key '{key}'")
            sys.exit(1)

    # Seed for reproducible preprocessing
    seed = config.get("seed")
    if seed is not None:
        seed_everything(seed)

    data_cfg = config["data"]
    preprocess_cfg = config["preprocessing"]

    # Load raw data
    raw_path = data_cfg["path"]
    storage_options = get_storage_options(raw_path)
    target_column = data_cfg["target_column"]
    print(f"\n[preprocess] Loading raw data from {raw_path}")
    df = pd.read_csv(raw_path, storage_options=storage_options)
    print(f"[preprocess] Loaded {len(df)} rows, {len(df.columns)} columns")

    # Identify feature columns (everything except target)
    feature_cols = [c for c in df.columns if c != target_column]
    print(f"[preprocess] Feature columns: {feature_cols}")
    print(f"[preprocess] Target column: {target_column}")

    # Scale features
    scaling = preprocess_cfg.get("scaling", "standardize")
    valid_methods = ("standardize", "min-max")
    if scaling not in valid_methods:
        print(f"[preprocess] Error: unknown scaling method '{scaling}', must be one of {valid_methods}")
        sys.exit(1)

    features = df[feature_cols].values.astype(np.float64)
    print(f"\n[preprocess] Scaling method: {scaling}")

    if scaling == "standardize":
        means = features.mean(axis=0)
        stds = features.std(axis=0)
        scaled = (features - means) / stds
    elif scaling == "min-max":
        mins = features.min(axis=0)
        maxs = features.max(axis=0)
        scaled = (features - mins) / (maxs - mins)

    df[feature_cols] = scaled

    print(f"[preprocess] Feature stats after scaling:")
    for col in feature_cols:
        print(f"  {col}: min={df[col].min():.6f}, max={df[col].max():.6f}, mean={df[col].mean():.6f}")

    # Write processed CSV
    output_path = preprocess_cfg["output_path"]
    if not output_path.startswith("s3://"):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False, storage_options=get_storage_options(output_path))
    print(f"\n[preprocess] Wrote processed data to {output_path}")
    print(f"[preprocess] Done.")


if __name__ == "__main__":
    main()
