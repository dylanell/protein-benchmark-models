# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

# Onboard data (local or S3)
uv run python scripts/onboard.py
uv run python scripts/onboard.py --dest s3://data/iris/

# Run scripts/notebooks
uv run jupyter notebook

# Preprocess data
uv run python scripts/preprocess.py --config configs/iris_mlp_classifier.json

# Train a model from config
uv run python scripts/train.py --config configs/iris_mlp_classifier.json
uv run python scripts/train.py --config configs/iris_gb_classifier.json

# Serve a trained model
uv run python scripts/serve.py --config configs/iris_mlp_classifier.json
uv run python scripts/serve.py --config configs/iris_gb_classifier.json

# Docker
docker build -t preprocessing-job -f docker/preprocess/Dockerfile .
docker build -t training-job -f docker/train/Dockerfile .
docker build -t serving-job -f docker/serve/Dockerfile .

docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  preprocessing-job --config configs/iris_mlp_classifier.json

docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  training-job --config configs/iris_mlp_classifier.json

docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  serving-job --config configs/iris_mlp_classifier.json

# Argo Workflows — first-time setup
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/

# Argo Workflows — run pipeline (preprocess → train)
argo submit -n argo argo/train-classifier-pipeline.yaml --watch
argo submit -n argo argo/train-classifier-pipeline.yaml -p config=configs/iris_gb_classifier.json --watch
```

## Architecture

```
src/ml_project_template/
├── data/                    # Dataset abstractions
│   ├── base.py              # BaseDataset ABC
│   └── tabular.py           # TabularDataset for numerical data
├── models/                  # Model implementations
│   ├── base.py              # BaseModel ABC (MLflow, save/load)
│   ├── registry.py          # ModelRegistry for model discovery
│   ├── gb_classifier.py     # Sklearn GradientBoosting wrapper
│   └── mlp_classifier.py    # PyTorch MLP classifier
├── modules/                 # Reusable nn.Module building blocks
│   └── fully_connected.py   # FullyConnected (MLP block with norm/activation)
├── serving/
│   └── app.py               # FastAPI app factory (generic, config-driven)
├── utils/
│   ├── io.py                # S3-compatible I/O utilities (get_storage_options, get_s3_filesystem)
│   └── seed.py              # seed_everything() for reproducibility

configs/                     # Training configs (JSON)
docker/                      # Dockerfiles per pipeline stage
├── preprocess/Dockerfile    # Preprocessing image
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
from ml_project_template.data import TabularDataset
from ml_project_template.utils import get_storage_options

dataset = TabularDataset.from_csv("s3://data/iris/iris.csv", target_column="species", storage_options=get_storage_options("s3://data/iris/iris.csv"))
train_data, test_data = dataset.split(test_size=0.2, random_state=42)
```

### Model Registry
```python
from ml_project_template.models import ModelRegistry
ModelRegistry.list()  # ['gb_classifier', 'mlp_classifier']
model = ModelRegistry.get("mlp_classifier")(layer_dims=[4, 16, 3])

# Load a saved model — class is inferred from config.json (no need to know it upfront)
model = ModelRegistry.load(".models/my_model")
# Or explicitly, when you know the model type
model = MLPClassifier.load(".models/my_model")
```

### Training with MLflow
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
# Model-specific training kwargs are forwarded to _fit()
# Training params are logged manually inside _fit()
model.train(
    experiment_name="my-experiment",
    train_data=train_data,
    val_data=val_data,
    model_path=".models/my_model",
    run_name="run-1",        # optional
    save_model="best",       # optional: "best" saves on each new best val loss,
                             #           "final" saves once after training completes.
                             #           Works with local and S3 paths.
    # Model-specific training kwargs (e.g. for MLP):
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=100,
    batch_size=32,
)
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
    def _fit(self, train_data, val_data=None, **kwargs) -> None
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

## Environment Variables

Configured via `.env` file (loaded automatically by `python-dotenv`) or set in the environment directly. In production, these are set by the deployment platform (K8s Secrets, IAM roles, etc.).

- `MLFLOW_TRACKING_URI` - MLflow server URL (default: local `./mlruns`)
- `S3_ENDPOINT_URL` - S3-compatible endpoint (e.g. `http://localhost:7000` for local MinIO)
- `AWS_ACCESS_KEY_ID` - S3 access key
- `AWS_SECRET_ACCESS_KEY` - S3 secret key
