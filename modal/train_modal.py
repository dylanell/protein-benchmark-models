"""Modal runner for GPU training during development.

Reads a local config file and runs scripts/train.py remotely on a GPU.
Dependencies are installed from uv.lock via `uv export` + `uv pip install --system`.
The export runs inside the Linux container so the correct CUDA torch wheel is resolved.
Update deps locally with `uv add <pkg>` as usual — the Modal image picks up changes
on next build.
Local source and scripts are synced at container start — code changes are
reflected immediately without rebuilding the image.
Data is read from MinIO and results logged to MLflow, same as any other training run.

Prerequisites:
  - Modal account set up: uv run modal setup
  - Modal secret named "protein-benchmark" with env vars from .env:
      AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY,
      MLFLOW_TRACKING_URI, S3_ENDPOINT_URL, MLFLOW_S3_ENDPOINT_URL
  - Config data paths must be S3 URIs (s3://...) not local paths

Usage:
  uv run modal run modal/train_modal.py --config configs/remote/tape_fluorescence_mlp_regressor.json
"""

import json

import modal

APP_NAME = "protein-benchmark-train"

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04",
        add_python="3.11",
    )
    # Install deps from uv.lock — runs on Linux so resolves the correct CUDA torch wheel.
    # uv pip install --system handles the +cu121 version specifier and custom index correctly.
    .run_commands("pip install uv")
    .add_local_file("pyproject.toml", "/app/pyproject.toml", copy=True)
    .add_local_file("uv.lock", "/app/uv.lock", copy=True)
    .run_commands(
        "cd /app"
        " && uv export --frozen --no-dev --no-install-project -o /tmp/requirements.txt"
        " && uv pip install --system --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy unsafe-best-match -r /tmp/requirements.txt"
    )
    # Sync local source at container start (copy=False default) — no image rebuild on code changes
    .add_local_python_source("protein_benchmark_models")
    .add_local_file("scripts/train.py", "/root/train.py")
)

app = modal.App(APP_NAME)


@app.function(
    image=image,
    gpu="T4",
    timeout=7200,  # 2 hours
    secrets=[modal.Secret.from_name("protein-benchmark")],
)
def run_training(config: dict) -> None:
    import train

    train.run(config)


@app.local_entrypoint()
def main(
    config: str = "configs/remote/tape_fluorescence_mlp_regressor.json",
) -> None:
    with open(config) as f:
        cfg = json.load(f)
    print(f"[modal] Submitting training run with config: {config}")
    print(json.dumps(cfg, indent=2))
    run_training.remote(cfg)
