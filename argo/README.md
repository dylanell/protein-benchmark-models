# Argo Workflows

Run the full pipeline (preprocess → train) as an Argo Workflow DAG on a local Docker Desktop K8s cluster. Argo handles sequencing and dependency management.

## Prerequisites

1. **Docker Desktop** with Kubernetes enabled (Settings > Kubernetes > Enable)
2. **Local services running**: `docker compose up -d` (MinIO + MLflow)
3. **Images built locally**:
   ```bash
   docker build -t preprocessing-job -f docker/preprocess-iris-dataset/Dockerfile .
   docker build -t training-job -f docker/train-iris-classifier/Dockerfile .
   ```
4. **Data onboarded to S3**: `uv run python scripts/onboard_iris_dataset.py --dest s3://data/iris/`
5. **Argo CLI installed**: `brew install argo`

## First-Time Setup

Install Argo Workflows and create the Secret and ConfigMap in the `argo` namespace:

```bash
# Install Argo Workflows (use --server-side to avoid CRD size limits)
kubectl create namespace argo
kubectl apply -n argo --server-side -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml

# Create secret and config map in the argo namespace
source .env && kubectl create secret generic s3-credentials --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY
kubectl create configmap training-configs --namespace argo --from-file=configs/
```

To update configs after changing them locally:

```bash
kubectl create configmap training-configs --namespace argo --from-file=configs/ --dry-run=client -o yaml | kubectl apply -f -
```

## Usage

### Submit a Pipeline

```bash
# Uses MLP config by default
argo submit -n argo argo/iris-classifier-pipeline.yaml --watch

# Or specify a different config via -p
argo submit -n argo argo/iris-classifier-pipeline.yaml -p config=configs/iris_gb_classifier.json --watch
```

### Argo UI (Optional)

```bash
kubectl port-forward -n argo svc/argo-server 2746:2746
```

Then open [https://localhost:2746](https://localhost:2746) for a visual workflow dashboard.

## How It Works

- The pipeline accepts a `config` parameter (`-p config=...`) so one workflow file handles any model config
- A Workflow defines a DAG of pipeline steps. Each step runs as a pod, with Argo automatically sequencing based on `dependencies`
- Workflows run in the `argo` namespace, so the S3 Secret and configs ConfigMap must be created there separately from the `default` namespace
- `generateName` allows resubmitting without deleting previous runs — each run gets a unique suffix
- Argo's emissary executor can't look up entrypoints from local-only images, so container specs must include an explicit `command`
- Data and model artifacts are stored in S3 (MinIO locally) — pipeline stages read/write via the S3 API
- Training configs are stored in a ConfigMap and mounted at `/app/configs`, overriding the baked-in defaults from the Docker image
- S3 credentials (`AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`) are stored in a K8s Secret (`s3-credentials`) and injected via `envFrom`. Endpoint URLs are plain env vars in the workflow spec
