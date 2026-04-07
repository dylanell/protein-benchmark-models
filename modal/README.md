# Modal — GPU Training for Development

Run training jobs on a GPU during development without managing any infrastructure.
Data is read from MinIO and results logged to MLflow — identical flow to local training
and GKE/Argo runs. Modal provisions and tears down compute per run; you pay only for
the seconds the job is running.

Use Modal for fast dev iteration. Use [Argo on GKE](../argo/README.md) for
reproducible production-style batch runs.

## Prerequisites

- Modal account: [modal.com](https://modal.com)
- Modal CLI authenticated: `uv run modal setup`
- Modal secret configured (see below)
- Config `data.train_path` / `data.valid_path` must be S3 URIs — use configs from `configs/remote/`

## One-Time Setup

### 1. Authenticate the CLI

```bash
uv run modal setup
```

### 2. Create the credentials secret

All five env vars from `.env` are needed so the training container can reach MinIO and MLflow:

```bash
uv run modal secret create protein-benchmark \
  AWS_ACCESS_KEY_ID=<value> \
  AWS_SECRET_ACCESS_KEY=<value> \
  MLFLOW_TRACKING_URI=http://34.42.116.216:5000 \
  S3_ENDPOINT_URL=http://34.42.116.216:7000 \
  MLFLOW_S3_ENDPOINT_URL=http://34.42.116.216:7000
```

Or create it in the Modal dashboard under Secrets.

## Usage

Always use `--detach` to avoid heartbeat failures if your terminal disconnects or your
machine sleeps during a long run:

```bash
# Default config (tape fluorescence MLP)
uv run modal run --detach modal/train_modal.py

# Specify any remote config
uv run modal run --detach modal/train_modal.py --config configs/remote/tape_fluorescence_cnn_regressor.json
uv run modal run --detach modal/train_modal.py --config configs/remote/flip2_amylase_random_split_mlp_regressor.json
```

The run URL is printed immediately and the job continues on Modal's infrastructure
regardless of local connectivity. Monitor logs via the URL or:

```bash
modal app logs <app-id>
```

The MLflow run appears in the remote tracking server at `MLFLOW_TRACKING_URI` as soon
as training starts.

## How It Works

- `modal/train_modal.py` defines a Modal App with a CUDA 12.1 image and T4 GPU
- Dependencies are installed from `uv.lock` via `uv export` + `uv pip install --system`
  at image build time — the lock file is the single source of truth
- `protein_benchmark_models` and `scripts/train.py` are synced from your local machine
  at container start (`copy=False`), so code changes are picked up without a rebuild
- The local config file is read, serialised, and passed to the remote function as a dict
- Data is pulled from MinIO via S3 API; results and model artifacts are logged to MLflow

## Image Caching

Modal caches image layers. The dep install step (slow, ~3 min) only reruns when
`pyproject.toml` or `uv.lock` changes. Code changes to `src/` or `scripts/` are synced
at container start and never trigger a rebuild.

## Cost

T4 GPU: ~$0.59/hr on Modal (on-demand). A typical training run of a few hundred epochs
costs under $1. No charge when no jobs are running.
