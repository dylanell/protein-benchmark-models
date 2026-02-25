# ML Project Template

Template for machine learning (ML) projects and API serving. 

## Using This Template

To start a new project from this template, rename the package directory and update all references:

1. Rename `src/ml_project_template/` to `src/<your_project_name>/`
2. Update `pyproject.toml`:
   - `name` (line 2) — the installable package name (use hyphens, e.g. `my-project`)
   - `[project.scripts]` entry — the CLI name and module path
3. Find and replace `ml_project_template` with `your_project_name` in all Python files (imports)
4. Update `CLAUDE.md` and this `README.md`

Then reinstall:
```bash
uv pip install -e "."
```

## Setup

```bash
# Install uv if you haven't
curl -LsSf https://astral.sh/uv/install.sh | sh

# Create virtual environment and install dependencies
uv venv
source .venv/bin/activate
uv pip install -e "." --group dev
```

Configure VSCode notebooks (add to .vscode/settings.json)
```
{                                                                                                                                                           
  "jupyter.notebookFileRoot": "${workspaceFolder}"                                                                                                          
}
```

## Architecture

```
src/ml_project_template/
├── data/                          # Dataset abstractions
│   ├── base.py                    # BaseDataset ABC
│   └── tabular.py                 # TabularDataset for numerical data
├── models/                        # Model implementations
│   ├── base.py                    # BaseModel ABC (MLflow, save/load)
│   ├── registry.py                # ModelRegistry for model discovery
│   ├── gb_classifier.py           # Sklearn GradientBoosting wrapper
│   └── mlp_classifier.py          # PyTorch MLP classifier (Fabric)
├── modules/                       # Reusable nn.Module building blocks
│   └── fully_connected.py         # FullyConnected (MLP block with norm/activation)
├── serving/
│   └── app.py                     # FastAPI app factory
├── utils/
│   ├── io.py                      # S3-compatible I/O utilities
│   └── seed.py                    # seed_everything() for reproducibility

configs/                           # Training configs (JSON)
docker/                            # Dockerfiles per pipeline stage
├── preprocess/Dockerfile          # Preprocessing image
├── train/Dockerfile               # Training image
└── serve/Dockerfile               # Serving image
argo/                              # Argo Workflow pipelines
scripts/                           # Data onboarding, preprocessing + training scripts
notebooks/                         # R&D notebooks
tests/                             # Test suite (no external services needed)
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

### Training
```python
# BaseModel.train() handles MLflow orchestration (params, artifacts)
# Model-specific training kwargs are forwarded to _fit()
model.train(
    experiment_name="my-experiment",
    train_data=train_data,
    val_data=val_data,
    model_path=".models/my_model",  # local or s3:// path
    run_name="run-1",               # optional
    save_model="best",              # optional: "best" or "final" (works with S3 paths too)
    # Model-specific training kwargs (e.g. for MLP):
    lr=1e-3,
    weight_decay=1e-4,
    max_epochs=100,
    batch_size=32,
)

# Or train without MLflow tracking for quick iteration
model.train(train_data=train_data, tracking=False, max_epochs=10)
```

### Reproducibility

Set a top-level `"seed"` key in your config JSON to seed all random number generators (Python, NumPy, PyTorch) for reproducible runs:

```json
{
    "seed": 42,
    "data": { ... },
    "model": { ... },
    "training": { ... }
}
```

Scripts call `seed_everything(seed)` before data loading, and pass `seed=seed` to `model.train()` which re-seeds before training. The seed is logged to MLflow automatically.

```python
from ml_project_template.utils import seed_everything
seed_everything(42)  # Seeds random, numpy, and torch (if available)
```

### Model Authoring Guide

#### Steps to add a new model

1. Create `src/ml_project_template/models/my_model.py` extending `BaseModel` (from `base.py`). For PyTorch models, initialize `lightning.Fabric` in `__init__` and use it for device/optimizer setup in `_fit()`
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
    def _fit(self, train_data, val_data=None, **kwargs) -> None
    def _save_weights(self, dir_path: str) -> None
    def _load_weights(self, dir_path: str) -> None
    def predict(self, X: np.ndarray) -> np.ndarray

    # Log to MLflow (no-op when tracking=False) — use in _fit() instead of mlflow directly
    def log_param(self, key, value) -> None
    def log_metric(self, key, value, step=None) -> None

    # Auto-populated from __init__ args via __init_subclass__ — no override needed
    # Override only if automatic capture is insufficient (e.g. sklearn **kwargs)
    def get_params(self) -> dict
```

#### Automatic `__init__` param capture

`BaseModel` uses `__init_subclass__` to automatically record all `__init__` arguments into `self._model_params`. This means `get_params()` works out of the box — you don't need to build param dicts manually or override it.

**How it works:**

1. `BaseModel.__init_subclass__` wraps every subclass `__init__` with a decorator.
2. After the original `__init__` runs, the wrapper inspects the method signature with `inspect.signature()`, binds the actual call arguments (including defaults) via `sig.bind()` + `apply_defaults()`, and stores them in `self._model_params`.
3. This works across the inheritance chain: when `super().__init__()` is called, the parent's wrapped `__init__` runs first and captures its own params. The child's wrapper then merges its params on top.
4. `get_params()` returns the combined `_model_params` dict. `train()` logs it to MLflow automatically.

See the implementation in `src/ml_project_template/models/base.py` lines 35–74.

**Example — what happens when `MLPClassifier(layer_dims=[4, 8, 3])` is created:**

1. `MLPClassifier.__init__` runs and completes
2. The wrapper captures all arguments — `layer_dims=[4, 8, 3]`, `hidden_activation="ReLU"`, `output_activation="Identity"`, `use_bias=True`, `norm=None`, plus all Fabric defaults (`accelerator="auto"`, etc.) — into `self._model_params`
3. `model.get_params()` returns the full dict

**When to override `get_params()`:**

Override when auto-capture is insufficient — specifically when your `__init__` uses `**kwargs` to forward arguments to an underlying library. Auto-capture will record the explicitly passed kwargs, but won't capture the library's internal defaults.

Example: `GBClassifier.__init__(self, **kwargs)` forwards to sklearn's `GradientBoostingClassifier(**kwargs)`. If you create `GBClassifier(n_estimators=200)`, auto-capture only sees `{"n_estimators": 200}`. But sklearn has dozens of other params with defaults (`learning_rate=0.1`, `max_depth=3`, etc.) that are important for reproducibility. So `GBClassifier` overrides `get_params()` to delegate to `self.model.get_params()`, which returns the full set.

**What NOT to worry about — training params:**

Training-time arguments like `lr`, `batch_size`, `max_epochs` are passed to `_fit()`, not `__init__()`, so they are **not** auto-captured. These are logged manually inside `_fit()` using `self.log_param()`. This is by design: `__init__` params define the model architecture (what gets saved/loaded), while training params are run-specific.

## Quick Start

### Local Services (MinIO + MLflow)

Start MinIO (S3-compatible storage) and MLflow (experiment tracking) via Docker Compose:

```bash
docker compose up -d
```

- **MLflow UI:** [http://localhost:5000](http://localhost:5000)
- **MinIO Console:** [http://localhost:7001](http://localhost:7001) (login: `minioadmin`/`minioadmin`)

Create `data` and `models` buckets in the MinIO console, then onboard data:

```bash
uv run python scripts/onboard.py --dest s3://data/iris/
```

Stop services with `docker compose down`. Data persists in Docker volumes across restarts.

### Docker

```bash
# Build images
docker build -t preprocess-job -f docker/preprocess/Dockerfile .
docker build -t train-job -f docker/train/Dockerfile .

# Run preprocessing (reads/writes data via S3)
docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  preprocess-job --config configs/iris_mlp_classifier.json

# Run training (reads data via S3, saves model to S3, logs to MLflow)
docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MLFLOW_S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -v $(pwd)/configs:/app/configs \
  train-job --config configs/iris_mlp_classifier.json
```

> `--env-file .env` loads S3 and MLflow credentials. The `-e` flags override the endpoint URLs to use `host.docker.internal`, which resolves to the host machine from inside Docker containers (Mac/Windows). On Linux, add `--add-host=host.docker.internal:host-gateway` to the `docker run` command.

### Model Serving

Serve a trained model via FastAPI. The server reads the same JSON config used for training to determine which model to load.

```bash
# Local (requires a trained model saved to the configured model_path)
uv run python scripts/serve.py --config configs/iris_mlp_classifier.json

# Docker
docker build -t serve-job -f docker/serve/Dockerfile .

docker run --env-file .env \
  -e S3_ENDPOINT_URL=http://host.docker.internal:7000 \
  -p 8000:8000 \
  -v $(pwd)/configs:/app/configs \
  serve-job --config configs/iris_mlp_classifier.json
```

Test the endpoints:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/info
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [[5.1, 3.5, 1.4, 0.2]]}'
```

Auto-generated API docs are available at `http://localhost:8000/docs`.

### Argo Workflows (Local)

Run the full pipeline (preprocess → train) as an Argo Workflow DAG on Docker Desktop's K8s cluster. See [argo/README.md](argo/README.md) for details.

```bash
# First-time: install Argo Workflows
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml

# First-time: create secret and config map in argo namespace
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/

# Submit a pipeline (uses MLP config by default)
argo submit -n argo argo/train-classifier-pipeline.yaml --watch

# Or specify a different config
argo submit -n argo argo/train-classifier-pipeline.yaml -p config=configs/iris_gb_classifier.json --watch

# Argo UI (optional)
kubectl port-forward -n argo svc/argo-server 2746:2746
# Then open https://localhost:2746
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --namespace argo --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Testing

Run the test suite with:

```bash
uv run pytest tests/ -v
```

Tests use `tracking=False` to skip MLflow, so no external services (MLflow, MinIO) are needed. They run entirely from in-memory numpy data.

When adding a new model, add matching tests in `tests/test_models.py` following the existing pattern:
- **`test_lifecycle`** — create, train (with `tracking=False`), predict (check output shape), save to a temp dir, load into a fresh instance, predict again (outputs match)
- **`test_get_params`** — verify `get_params()` returns the expected keys and values
