# Protein Benchmark Models

Benchmark suite comparing protein ML models on regression tasks from [TAPE](https://github.com/songlab-gpu/tape) and [FLIP2](https://zenodo.org/records/18433203). Simple baselines (ridge regression, MLP, CNN) are benchmarked first; pretrained protein LLM embeddings will follow.

## TODO

- [x] Add 1D CNN baseline model over tokenized sequences
- [x] Onboard FLIP2 tasks and run baselines
- [ ] Add pretrained protein LLM embedding models (ESM, etc.)
- [ ] Train baselines to convergence on all FLIP2 splits
- [ ] Wire up Docker training image and Argo pipeline for GPU training runs

## Setup

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e "." --group dev
```

Configure VSCode notebooks (add to `.vscode/settings.json`):
```json
{
  "jupyter.notebookFileRoot": "${workspaceFolder}"
}
```

## Architecture

```
src/protein_benchmark_models/
├── data/                          # Dataset abstractions
│   ├── base.py                    # BaseDataset ABC
│   ├── sequence.py                # SequenceDataset, TokenizedSequenceDataset, OneHotSequenceDataset
│   └── tabular.py                 # TabularDataset for numerical data
├── models/                        # Model implementations
│   ├── base.py                    # BaseModel ABC (MLflow, save/load)
│   ├── registry.py                # ModelRegistry for model discovery
│   ├── ridge_regressor.py         # Sklearn Ridge wrapper
│   ├── mlp_regressor.py           # PyTorch MLP regressor (Fabric)
│   └── cnn_regressor.py           # PyTorch 1D CNN regressor (Fabric)
├── modules/                       # Reusable nn.Module building blocks
│   ├── fully_connected.py         # FullyConnected (MLP block with norm/activation)
│   ├── sequence_cnn.py            # SequenceCNN (stacked 1D convolutions)
│   └── utils.py                   # Shared utilities (Transpose)
├── serving/
│   └── app.py                     # FastAPI app factory
└── utils/
    ├── io.py                      # S3-compatible I/O utilities
    ├── metrics.py                 # evaluate() — RMSE, R2, SpearmanR
    └── seed.py                    # seed_everything() for reproducibility

configs/                           # Training configs (JSON)
docker/                            # Dockerfiles per pipeline stage
scripts/                           # Data onboarding and pipeline entry points
notebooks/                         # R&D notebooks
tests/                             # Test suite (no external services needed)
```

## Key Patterns

### Data Loading
```python
from protein_benchmark_models.data import OneHotSequenceDataset, TokenizedSequenceDataset
import pandas as pd

df = pd.read_csv(".data/tape/fluorescence/train.csv")
sequences = df["sequence"].tolist()
targets = df["target"].to_numpy()
max_seq_len = df["sequence"].map(len).max()

# For ridge/MLP: flatten one-hot encoded sequences
dataset = OneHotSequenceDataset(sequences=sequences, targets=targets, seq_len=max_seq_len)
# dataset[i] -> {"one_hots": FloatTensor(seq_len, vocab_size), "target": float32}

# For CNN: integer-tokenized sequences
dataset = TokenizedSequenceDataset(sequences=sequences, targets=targets, seq_len=max_seq_len)
# dataset[i] -> {"tokens": LongTensor(seq_len,), "target": float32}
```

### Model Registry
```python
from protein_benchmark_models.models import ModelRegistry

ModelRegistry.list()  # ['ridge_regressor', 'mlp_regressor', 'cnn_regressor']

model = ModelRegistry.get("ridge_regressor")(alpha=1.0)

# Load a saved model — class is inferred from config.json (no need to know it upfront)
model = ModelRegistry.load(".models/my_model")
```

### Training
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
model.train(
    experiment_name="my-experiment",
    train_data=train_dataset,
    val_data=val_dataset,           # required
    model_path=".models/my_model",  # required; _final and _best suffixes appended automatically
    run_name="run-1",               # optional
    tracking=True,
    # Model-specific training kwargs (e.g. for MLP/CNN):
    lr=1e-4,
    weight_decay=0.01,
    max_epochs=100,
    batch_size=256,
)
# Writes: .models/my_model_final (always) and .models/my_model_best (PyTorch only, on val improvement)

# Quick iteration without MLflow
model.train(train_data=train_dataset, val_data=val_dataset, model_path=".models/my_model", tracking=False, max_epochs=5)
```

### Evaluation
```python
import numpy as np
from protein_benchmark_models.utils import evaluate

X = np.stack([dataset[i]["one_hots"].numpy().flatten() for i in range(len(dataset))])
y = dataset.targets.numpy()

metrics = evaluate(model, X, y)
# {"rmse": float, "r2": float, "spearmanr": float}
```

### Reproducibility

Pass `seed=42` to the model constructor. It seeds before weight init and again at the start of `_fit()` for shuffle/dropout reproducibility. The seed is logged to MLflow automatically via `get_params()`.

```python
from protein_benchmark_models.utils import seed_everything
seed_everything(42)  # Seeds random, numpy, and torch — call before model construction in notebooks

model = MLPRegressor(layer_dims=[..., 1], seed=42)
```

### Model Authoring Guide

#### Steps to add a new model

1. Create `src/protein_benchmark_models/models/my_model.py` extending `BaseModel` (from `base.py`). For PyTorch models, initialize `lightning.Fabric` in `__init__` and use it for device/optimizer setup in `_fit()`
2. Implement `_fit()`, `_save_weights()`, `_load_weights()`, and `predict()`
3. Register in `registry.py`
4. Add lifecycle and get_params tests in `tests/test_models.py`

You do **not** need to implement `get_params()` in most cases — see below.

#### BaseModel Interface

```python
class BaseModel(ABC):
    # Public API — MLflow orchestration (set experiment, start run, log params, save artifact)
    def train(self, *, experiment_name, train_data, tracking=True, **kwargs) -> None

    # Public API — saves config.json + calls _save_weights()
    def save(self, path: str) -> str

    # Public classmethod — reads config.json, instantiates model, loads weights, returns instance
    @classmethod
    def load(cls, path: str) -> BaseModel

    # Abstract — subclasses must implement
    def _fit(self, train_data, val_data, **kwargs) -> None
    def _save_weights(self, dir_path: str) -> None
    def _load_weights(self, dir_path: str) -> None
    def predict(self, X: np.ndarray) -> np.ndarray  # always returns shape (N,)

    # Log to MLflow (no-op when tracking=False) — use in _fit() instead of mlflow directly
    def log_param(self, key, value) -> None
    def log_metric(self, key, value, step=None) -> None

    # Auto-populated from __init__ args via __init_subclass__ — no override needed
    # Override only if automatic capture is insufficient (e.g. sklearn **kwargs)
    def get_params(self) -> dict
```

#### Automatic `__init__` param capture

`BaseModel` uses `__init_subclass__` to automatically record all `__init__` arguments into `self._model_params`. This means `get_params()` works out of the box — you don't need to build param dicts manually or override it.

Training-time arguments (`lr`, `batch_size`, `max_epochs`, etc.) are passed to `_fit()`, not `__init__()`, so they are **not** auto-captured. Log them manually inside `_fit()` using `self.log_param()`.

Override `get_params()` only when auto-capture is insufficient — e.g. when `__init__` uses `**kwargs` to forward arguments to an underlying library that has its own defaults (see `ridge_regressor.py` for an example).

## Datasets

### Onboarding

Download a dataset to a local `.data/` subdirectory (or an S3 prefix with `--dest s3://...`):

```bash
# TAPE tasks
uv run python scripts/onboard.py --task fluorescence
uv run python scripts/onboard.py --task stability

# FLIP2 tasks (all splits downloaded into subdirectories)
uv run python scripts/onboard.py --task amylase
uv run python scripts/onboard.py --task ired
uv run python scripts/onboard.py --task nucb
uv run python scripts/onboard.py --task hydro
uv run python scripts/onboard.py --task rhomax
```

Output structure:
```
.data/tape/<task>/train.csv, valid.csv, test.csv
.data/flip2/<task>/<split>/train.csv, valid.csv, test.csv
```

All CSVs share a consistent schema:

| column | type | notes |
|---|---|---|
| `sequence` | str | Amino acid sequence |
| `target` | float | Regression target |
| `num_mutations` | int | TAPE fluorescence only |

### TAPE — Fluorescence

- **Source:** Sarkisyan et al. 2016 (GFP), via [TAPE](https://github.com/songlab-gpu/tape)
- **Task:** Predict log-fluorescence from amino acid sequence
- **Splits:** train 21,446 / valid 5,362 / test 27,217
- **OOD split:** test set contains sequences with ≥4 mutations; train/valid have ≤3

Most sequences are 237 AA (wild-type GFP length). A small fraction are shorter (236 or 235 AA) due to deletions introduced during error-prone PCR mutagenesis. The `num_mutations` field counts all edit-distance differences from wild-type — substitutions and deletions alike. Filter on `seq_len == 237` if you want substitution-only variants.

### TAPE — Stability

- **Source:** Rocklin et al. 2017 (de novo proteins), via [TAPE](https://github.com/songlab-gpu/tape)
- **Task:** Predict thermodynamic stability score from amino acid sequence
- **Splits:** train 53,614 / valid 2,512 / test 12,851

### FLIP2 — Mutation Fitness Tasks

All FLIP2 tasks are sourced from [Zenodo record 18433203](https://zenodo.org/records/18433203). Each task has multiple named splits with different generalization challenges.

| task | splits |
|---|---|
| `amylase` | `one_to_many`, `close_to_far`, `far_to_close`, `by_mutation`, `random_split` |
| `ired` | `two_to_many`, `random` |
| `nucb` | `two_to_many`, `random` |
| `hydro` | `three_to_many`, `low_to_high`, `to_P06241`, `to_P0A9X9`, `to_P01053`, `random_split` |
| `rhomax` | `by_wild_type` |

## Services (MinIO + MLflow)

MLflow (experiment tracking) and MinIO (S3-compatible artifact storage) are run via the same `docker-compose.yaml` in both local and remote setups. The only difference is the `.env` file. See `.env.example` for both configurations.

### Option A — Local (Docker Desktop)

```bash
docker compose up -d
```

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **MinIO console:** [http://localhost:7001](http://localhost:7001) (login: `minioadmin`/`minioadmin`)

Stop with `docker compose down`. Data persists in Docker volumes.

### Option B — Remote (GCE VM)

A persistent GCE VM (`e2-medium`, `us-central1-a`) runs the same `docker-compose.yaml` with a reserved static IP. This allows training jobs running anywhere (local, Modal, GKE) to log to a shared MLflow server.

```bash
# Start VM
gcloud compute instances start mlflow-server --zone us-central1-a

# Stop VM when not in use (no compute charge while stopped)
gcloud compute instances stop mlflow-server --zone us-central1-a

# SSH in
gcloud compute ssh mlflow-server --zone us-central1-a
```

To deploy from scratch on a new VM:

```bash
# 1. Create VM
gcloud compute instances create mlflow-server \
  --zone us-central1-a \
  --machine-type e2-medium \
  --image-family debian-12 \
  --image-project debian-cloud \
  --tags mlflow-server \
  --boot-disk-size 20GB

# 2. Open ports
gcloud compute firewall-rules create allow-mlflow-minio \
  --allow tcp:5000,tcp:7000,tcp:7001 \
  --target-tags mlflow-server

# 3. Reserve and attach a static IP
gcloud compute addresses create mlflow-server-ip --region us-central1
gcloud compute instances delete-access-config mlflow-server --access-config-name "external-nat" --zone us-central1-a
gcloud compute instances add-access-config mlflow-server --access-config-name "external-nat" \
  --address $(gcloud compute addresses describe mlflow-server-ip --region us-central1 --format='get(address)') \
  --zone us-central1-a

# 4. Install Docker, copy docker-compose.yaml, create .env, start services
gcloud compute ssh mlflow-server --zone us-central1-a
# On the VM:
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER && newgrp docker
# Copy docker-compose.yaml to the VM (from local):
# gcloud compute scp docker-compose.yaml mlflow-server:~/docker-compose.yaml --zone us-central1-a
docker compose up -d
```

Update `.env` locally to point at the VM's static IP (see `.env.example` Option B).

> **Security note:** MLflow and MinIO are exposed publicly with no authentication. Acceptable for private R&D; restrict firewall rules to known IPs for shared use.

## Docker

Dockerfiles for each pipeline stage live in `docker/`. These are not yet wired up for active use but are intended for running training jobs remotely (e.g. on a GPU instance).

```bash
# Build images
docker build -t preprocess-job -f docker/preprocess/Dockerfile .
docker build -t train-job -f docker/train/Dockerfile .
docker build -t serve-job -f docker/serve/Dockerfile .

# Run a training job — S3 and MLflow endpoints are read from .env
docker run --env-file .env \
  -v $(pwd)/configs:/app/configs \
  train-job --config configs/<your_config>.json
```

> When running against a local Docker Compose stack on macOS/Windows, replace `localhost` with `host.docker.internal` in your `.env` so the container can reach services on the host (e.g. `S3_ENDPOINT_URL=http://host.docker.internal:7000`). On Linux, also add `--add-host=host.docker.internal:host-gateway`. When using the remote GCE setup, `.env` already has the correct public IP and no changes are needed.

## Argo Workflows

Argo Workflow pipelines live in `argo/` and are intended for orchestrating multi-step training runs on Kubernetes (e.g. preprocess → train). See [argo/README.md](argo/README.md) for details.

```bash
# First-time: install Argo Workflows on Docker Desktop's K8s cluster
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml

# First-time: create S3 credentials secret and configs config map
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/

# Submit a pipeline
argo submit -n argo argo/train-pipeline.yaml --watch

# Argo UI (optional)
kubectl port-forward -n argo svc/argo-server 2746:2746
# Then open https://localhost:2746
```

To update the configs config map after changing files locally:

```bash
kubectl create configmap training-configs --namespace argo --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Testing

```bash
uv run pytest tests/ -v
```

Tests use `tracking=False` to skip MLflow — no external services needed. They run entirely from in-memory numpy data.

When adding a new model, add matching tests in `tests/test_models.py` following the existing pattern:
- **`test_lifecycle`** — create, train (`tracking=False`), predict (check output shape), save, load, predict again (outputs match)
- **`test_get_params`** — verify `get_params()` returns the expected keys and values
