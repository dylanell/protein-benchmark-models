# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Session Notes

At the end of each session, write a session notes file to `.claude/session_notes_<N>.md`
at the project root, where N is the next session number. Follow the style of existing
notes in that directory. Capture: what was done, state at end of session, and TODOs for
next session.

Note: `.claude/` is gitignored (see `.gitignore`), so session notes are local only.

## Gitignored Directories

The following project-specific paths are gitignored:
- `.claude/` — Claude session notes and local AI context
- `.data/` — raw and processed datasets
- `.models/` — saved model artifacts
- `mlruns/`, `mlartifacts/`, `mlflow.db` — MLflow local tracking

## Project Overview

ML project template for model R&D and API serving. Uses uv for package management with Python 3.11+.

## Commands

```bash
# Setup
uv venv && uv pip install -e "." --group dev

# Run tests
uv run pytest tests/ -v

# Start local services (MinIO + MLflow)
docker compose up -d

# Onboard data
uv run python scripts/onboard.py --task fluorescence
uv run python scripts/onboard.py --task amylase

# Run scripts/notebooks
uv run jupyter notebook

# Train a model from config
uv run python scripts/train.py --config configs/tape_fluorescence_ridge_regressor.json
uv run python scripts/train.py --config configs/tape_fluorescence_mlp_regressor.json

# Serve a trained model
uv run python scripts/serve.py --config configs/tape_fluorescence_mlp_regressor.json

# Docker
docker build -t training-job -f docker/train/Dockerfile .
docker build -t serving-job -f docker/serve/Dockerfile .

# S3 and MLflow endpoints are read from .env
docker run --env-file .env \
  -v $(pwd)/configs:/app/configs \
  training-job --config configs/tape_fluorescence_mlp_regressor.json

# Argo Workflows — first-time setup
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/

# Argo Workflows — run pipeline (preprocess → train)
argo submit -n argo argo/train-pipeline.yaml --watch
```

## Architecture

```
src/protein_benchmark_models/
├── data/                    # Dataset abstractions
│   ├── base.py              # BaseDataset ABC (legacy)
│   ├── tabular.py           # TabularDataset (legacy)
│   └── sequence.py          # SequenceDataset, TokenizedSequenceDataset, OneHotSequenceDataset
├── models/                  # Model implementations
│   ├── base.py              # BaseModel ABC (MLflow, save/load)
│   ├── registry.py          # ModelRegistry for model discovery
│   ├── ridge_regressor.py   # Sklearn Ridge wrapper
│   ├── mlp_regressor.py     # PyTorch MLP regressor (Fabric)
│   └── cnn_regressor.py     # PyTorch 1D CNN regressor (Fabric)
├── modules/                 # Reusable nn.Module building blocks
│   ├── fully_connected.py   # FullyConnected (MLP block with norm/activation)
│   ├── sequence_cnn.py      # SequenceCNN (stacked 1D convolutions)
│   └── utils.py             # Shared nn.Module utilities (Transpose)
├── serving/
│   └── app.py               # FastAPI app factory (legacy, Iris-era)
└── utils/
    ├── io.py                # S3-compatible I/O utilities
    ├── metrics.py           # evaluate() — RMSE, R2, SpearmanR
    └── seed.py              # seed_everything() for reproducibility

configs/                     # Training configs (JSON)
docker/                      # Dockerfiles per pipeline stage
├── train/Dockerfile         # Training image
└── serve/Dockerfile         # Serving image
argo/                        # Argo Workflow pipelines
.data/                       # Raw datasets (gitignored)
.models/                     # Saved model artifacts (gitignored)
scripts/                     # Pipeline stage entry points
notebooks/                   # R&D notebooks
```

## Key Patterns

### Data Loading
```python
from protein_benchmark_models.data import OneHotSequenceDataset, TokenizedSequenceDataset
import pandas as pd

df = pd.read_csv(".data/tape/fluorescence/train.csv")
seq_len = df["sequence"].map(len).max()

# For ridge/MLP: flatten one-hot encoded sequences
dataset = OneHotSequenceDataset(df["sequence"].tolist(), df["target"].tolist(), seq_len=seq_len)

# For CNN: integer-tokenized sequences
dataset = TokenizedSequenceDataset(df["sequence"].tolist(), df["target"].tolist(), seq_len=seq_len)
```

### Model Registry
```python
from protein_benchmark_models.models import ModelRegistry
ModelRegistry.list()  # ['ridge_regressor', 'mlp_regressor', 'cnn_regressor']
model = ModelRegistry.get("ridge_regressor")(alpha=1.0)

# Load a saved model — class is inferred from config.json (no need to know it upfront)
model = ModelRegistry.load(".models/my_model_final")
# Or explicitly, when you know the model type
model = RidgeRegressor.load(".models/my_model_final")
```

### Training with MLflow
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
# Model-specific training kwargs are forwarded to _fit()
# Training params are logged manually inside _fit()
model.train(
    experiment_name="my-experiment",
    train_data=train_data,
    val_data=val_data,           # required
    model_path=".models/my_model",  # required; _final and _best suffixes are appended automatically
    run_name="run-1",            # optional
    # Model-specific training kwargs (e.g. for MLP):
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=100,
    batch_size=32,
)
# Writes: .models/my_model_final (always) and .models/my_model_best (PyTorch only, on val improvement)
```

### Adding New Models
1. Create `src/ml_project_template/models/my_model.py` extending `BaseModel` (from `base.py`)
2. Implement `_fit()`, `_save_weights()`, `_load_weights()`, and `predict()` (and `get_params()` only if automatic capture doesn't work — see below)
3. Register in `registry.py`

For PyTorch models, initialize `lightning.Fabric` in `__init__` directly (see `mlp_classifier.py`).

### BaseModel Interface
```python
class BaseModel(ABC):
    # Public API — MLflow orchestration (set experiment, start run, log params, save artifact)
    def train(self, *, experiment_name, train_data, **kwargs) -> None

    # Public API — saves config.json + calls _save_weights()
    def save(self, path: str) -> str

    # Public classmethod — reads config.json, instantiates model, loads weights, returns instance
    @classmethod
    def load(cls, path: str) -> BaseModel

    # Abstract — subclasses must implement
    def _fit(self, train_data, val_data, **kwargs) -> None
    def _save_weights(self, dir_path: str) -> None
    def _load_weights(self, dir_path: str) -> None
    def predict(self, X: np.ndarray) -> np.ndarray

    # Auto-populated from __init__ args via __init_subclass__ — no override needed
    # Override only if automatic capture is insufficient (e.g. sklearn **kwargs)
    def get_params(self) -> dict
```

### Automatic `__init__` Param Capture
`BaseModel` uses `__init_subclass__` to automatically record all `__init__` arguments
into `self._model_params`. Logged to MLflow automatically in `train()`. Override
`get_params()` only when needed (e.g. sklearn models where `**kwargs` doesn't capture
individual params with defaults).

See README.md "Automatic `__init__` param capture" for a full walkthrough with examples.

## Conventions

- **Notebooks run from project root** - VS Code setting `jupyter.notebookFileRoot` is set
- **Data in `.data/`** - Raw datasets, gitignored
- **Models in `.models/`** - Saved artifacts, gitignored
- **PyTorch models**: Separate `nn.Module` class from `BaseModel` wrapper in same file; initialize `lightning.Fabric` directly in `__init__`
- **Sklearn models**: Override `get_params()` to delegate to sklearn's introspection
- **NumPy I/O**: All models return raw numpy output from `predict()` — caller handles post-processing (argmax, etc.)
- **Training params**: Logged manually inside `_fit()`, not auto-captured (only `__init__` args are auto-captured)
- **Reproducibility**: Configs have a top-level `"seed"` key. Scripts call `seed_everything(seed)` early, and pass `seed=seed` to `model.train()`. The seed is logged to MLflow automatically.
- **`embed_dims[0]` in CNN configs**: Must equal `len(AA_VOCAB)` = 22 (the fixed protein amino acid vocabulary size including PAD and UNK). This is intentionally explicit in configs — it's a constant property of the protein alphabet, not a data-derived value like `seq_len`, so it is not auto-injected by `train.py`.
- **`_fit()` arg ordering**: `model_path` is always first after `*` in both MLP and CNN `_fit()` signatures, followed by training hyperparams (`lr`, `weight_decay`, etc.).

## Environment Variables

Configured via `.env` file (loaded automatically by `python-dotenv`). See `.env.example` for both local and remote configurations.

- `MLFLOW_TRACKING_URI` - MLflow server URL (e.g. `http://localhost:5000` or `http://<vm-ip>:5000`)
- `S3_ENDPOINT_URL` - S3-compatible endpoint for data I/O (e.g. `http://localhost:7000` for MinIO)
- `MLFLOW_S3_ENDPOINT_URL` - S3-compatible endpoint for MLflow artifact storage
- `AWS_ACCESS_KEY_ID` - S3/MinIO access key
- `AWS_SECRET_ACCESS_KEY` - S3/MinIO secret key

## Local Services (MLflow + MinIO)

```bash
# Start
docker compose up -d

# MLflow UI: http://localhost:5000
# MinIO console: http://localhost:7001
```

## Remote Services (GCE VM)

Deploy the same `docker-compose.yaml` on a GCP VM for a persistent, GPU-job-accessible
MLflow + MinIO server. Only `.env` needs to change — the rest of the codebase is identical.

```bash
# 1. Create a VM (e2-micro is free-tier eligible)
gcloud compute instances create mlflow-server \
  --zone us-central1-a \
  --machine-type e2-micro \
  --image-family debian-12 \
  --image-project debian-cloud \
  --tags mlflow-server

# 2. Open ports for MLflow (5000) and MinIO (7000, 7001)
gcloud compute firewall-rules create allow-mlflow \
  --allow tcp:5000,tcp:7000,tcp:7001 \
  --target-tags mlflow-server

# 3. SSH in, install Docker, copy docker-compose.yaml, and start
gcloud compute ssh mlflow-server
# On the VM:
curl -fsSL https://get.docker.com | sh
# Copy docker-compose.yaml to the VM, then:
docker compose up -d

# 4. Get the VM's external IP
gcloud compute instances describe mlflow-server \
  --zone us-central1-a \
  --format='get(networkInterfaces[0].accessConfigs[0].natIP)'
```

Then update `.env` with the VM's external IP (see `.env.example` Option B).

> **Security note**: This exposes MLflow and MinIO publicly with no authentication.
> Acceptable for private R&D; for shared or production use, restrict firewall rules
> to known IPs or add a reverse proxy with authentication.
