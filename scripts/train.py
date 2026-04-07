"""Config-driven training script for protein benchmarking tasks.

Loads pre-split train/valid CSVs, builds the appropriate sequence dataset,
injects data-derived shape params into the model constructor, and trains.

Task types (set via data.task_type in the config):
    one_hot             — single sequence, one-hot encoded (MLP models)
    tokenized           — single sequence, AA-vocab tokenized (CNN models)
    paired_onehot       — sequence pair, one-hot encoded (Siamese MLP models)
    paired_tokenized    — sequence pair, AA-vocab tokenized (Siamese CNN models)

To add a new task type: implement a _build_* and _inject_* function, then
add a TaskHandler entry to TASK_HANDLERS. No other changes needed.

Usage:
    uv run python scripts/train.py \
        --config configs/local/tape_fluorescence_ridge_regressor.json
"""

import logging
import argparse
import json
import sys
from dataclasses import dataclass
from typing import Callable

import pandas as pd
from dotenv import load_dotenv

from protein_benchmark_models.data import (
    AA_VOCAB,
    OneHotSequenceDataset,
    TokenizedSequenceDataset,
    PairedOneHotSequenceDataset,
    PairedTokenizedSequenceDataset,
)
from protein_benchmark_models.models import ModelRegistry
from protein_benchmark_models.utils import get_storage_options, seed_everything

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
_SCRIPT = "train.py"


# ---------------------------------------------------------------------------
# Task handlers
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskHandler:
    """Bundles per-task dataset construction and model shape injection.

    build_datasets: (train_df, val_df, data_cfg) -> (train_ds, val_ds, seq_len)
        Reads the appropriate columns, computes seq_len, and constructs the
        train and validation Dataset objects.

    inject_shape: (model_params, seq_len) -> model_params
        Inserts data-derived dimensions (input size, seq_length, etc.) into
        the raw model params dict from the config. Returns an updated copy.
    """

    build_datasets: Callable
    inject_shape: Callable


def _build_single_onehot(train_df, val_df, data_cfg):
    seq_len = data_cfg.get("seq_len") or max(len(s) for s in train_df["sequence"])
    train_ds = OneHotSequenceDataset(
        train_df["sequence"].tolist(),
        train_df["target"].tolist(),
        seq_len=seq_len,
    )
    val_ds = OneHotSequenceDataset(
        val_df["sequence"].tolist(), val_df["target"].tolist(), seq_len=seq_len
    )
    return train_ds, val_ds, seq_len


def _inject_onehot_shape(model_params, seq_len):
    # MLP operates on flattened one-hot vectors: prepend the computed input
    # dim to layer_dims so the config only stores hidden/output dims.
    if "layer_dims" in model_params:
        input_dim = seq_len * len(AA_VOCAB)
        return {
            **model_params,
            "layer_dims": [input_dim] + model_params["layer_dims"],
        }
    return model_params


def _build_single_tokenized(train_df, val_df, data_cfg):
    seq_len = data_cfg.get("seq_len") or max(len(s) for s in train_df["sequence"])
    train_ds = TokenizedSequenceDataset(
        train_df["sequence"].tolist(),
        train_df["target"].tolist(),
        seq_len=seq_len,
    )
    val_ds = TokenizedSequenceDataset(
        val_df["sequence"].tolist(), val_df["target"].tolist(), seq_len=seq_len
    )
    return train_ds, val_ds, seq_len


def _inject_tokenized_shape(model_params, seq_len):
    # CNN conv layers slide along the sequence dimension, so seq_length must
    # be passed explicitly to the model constructor.
    return {**model_params, "seq_length": seq_len}


def _build_paired_onehot(train_df, val_df, data_cfg):
    # Accept an explicit seq_len from the config (recommended for large
    # datasets where computing max length would be slow or memory-heavy).
    seq_len = data_cfg.get("seq_len") or max(
        max(len(s) for s in train_df["sequence_a"]),
        max(len(s) for s in train_df["sequence_b"]),
    )
    train_ds = PairedOneHotSequenceDataset(
        train_df["sequence_a"].tolist(),
        train_df["sequence_b"].tolist(),
        train_df["target"].tolist(),
        seq_len=seq_len,
    )
    val_ds = PairedOneHotSequenceDataset(
        val_df["sequence_a"].tolist(),
        val_df["sequence_b"].tolist(),
        val_df["target"].tolist(),
        seq_len=seq_len,
    )
    return train_ds, val_ds, seq_len


def _inject_paired_onehot_shape(model_params, seq_len):
    # Siamese MLP: prepend the per-sequence input dim to encoder_dims.
    # head_dims must be fully specified in the config.
    if "encoder_dims" in model_params:
        input_dim = seq_len * len(AA_VOCAB)
        return {
            **model_params,
            "encoder_dims": [input_dim] + model_params["encoder_dims"],
        }
    return model_params


def _build_paired_tokenized(train_df, val_df, data_cfg):
    # Accept an explicit seq_len from the config (recommended for large
    # datasets where computing max length would be slow or memory-heavy).
    seq_len = data_cfg.get("seq_len") or max(
        max(len(s) for s in train_df["sequence_a"]),
        max(len(s) for s in train_df["sequence_b"]),
    )
    train_ds = PairedTokenizedSequenceDataset(
        train_df["sequence_a"].tolist(),
        train_df["sequence_b"].tolist(),
        train_df["target"].tolist(),
        seq_len=seq_len,
    )
    val_ds = PairedTokenizedSequenceDataset(
        val_df["sequence_a"].tolist(),
        val_df["sequence_b"].tolist(),
        val_df["target"].tolist(),
        seq_len=seq_len,
    )
    return train_ds, val_ds, seq_len


def _inject_paired_tokenized_shape(model_params, seq_len):
    # Siamese CNN: conv layers slide along the sequence dimension, so
    # seq_length must be passed explicitly to the model constructor.
    return {**model_params, "seq_length": seq_len}


TASK_HANDLERS: dict[str, TaskHandler] = {
    "one_hot": TaskHandler(_build_single_onehot, _inject_onehot_shape),
    "tokenized": TaskHandler(_build_single_tokenized, _inject_tokenized_shape),
    "paired_onehot": TaskHandler(
        _build_paired_onehot, _inject_paired_onehot_shape
    ),
    "paired_tokenized": TaskHandler(
        _build_paired_tokenized, _inject_paired_tokenized_shape
    ),
}


# ---------------------------------------------------------------------------
# Core training logic
# ---------------------------------------------------------------------------


def run(config: dict) -> None:
    """Core training logic. Accepts a config dict — callable locally or from
    remote runners."""
    for key in ("data", "model", "training"):
        if key not in config:
            logging.error(f"[{_SCRIPT}] config missing required key '{key}'")
            sys.exit(1)

    seed = config.get("seed")
    data_cfg = config["data"]

    task_type = data_cfg.get("task_type")
    if task_type is None:
        logging.error(f"[{_SCRIPT}] config missing 'data.task_type'")
        sys.exit(1)
    if task_type not in TASK_HANDLERS:
        logging.error(
            f"[{_SCRIPT}] unknown task_type '{task_type}'. "
            f"Choose from: {list(TASK_HANDLERS)}"
        )
        sys.exit(1)

    train_path = data_cfg["train_path"]
    valid_path = data_cfg["valid_path"]

    logging.info(f"[{_SCRIPT}] Loading train data from {train_path}")
    train_df = pd.read_csv(
        train_path, storage_options=get_storage_options(train_path)
    )
    logging.info(f"[{_SCRIPT}] Loading valid data from {valid_path}")
    val_df = pd.read_csv(
        valid_path, storage_options=get_storage_options(valid_path)
    )
    logging.info(
        f"[{_SCRIPT}] Train: {len(train_df):,} rows, "
        f"Valid: {len(val_df):,} rows"
    )

    handler = TASK_HANDLERS[task_type]
    train_dataset, val_dataset, seq_len = handler.build_datasets(
        train_df, val_df, data_cfg
    )
    logging.info(f"[{_SCRIPT}] seq_len: {seq_len}")

    model_cfg = config["model"]
    model_params = handler.inject_shape(
        dict(model_cfg.get("params", {})), seq_len
    )

    if seed is not None:
        seed_everything(seed)

    logging.info(
        f"[{_SCRIPT}] Constructing model '{model_cfg['name']}' "
        f"with params: {model_params}"
    )
    model = ModelRegistry.get(model_cfg["name"])(**model_params)

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

    logging.info(f"[{_SCRIPT}] Training complete.")


def main():
    parser = argparse.ArgumentParser(
        description="Train a protein benchmark model from a config."
    )
    parser.add_argument(
        "--config", required=True, help="Path to JSON config file"
    )
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    logging.info(f"[{_SCRIPT}] Running with config:")
    logging.info(json.dumps(config, indent=2))

    run(config)


if __name__ == "__main__":
    main()
