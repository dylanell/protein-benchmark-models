# Argo Workflows — GPU Training on GKE

Run training jobs as Argo Workflows on a GKE cluster with T4 GPU spot nodes.
Data is read from MinIO (S3-compatible), results logged to MLflow — both running
on the `mlflow-server` GCE VM.

## Prerequisites

- `gcloud` CLI authenticated (`gcloud auth login`)
- `kubectl` installed
- `argo` CLI installed (`brew install argo`)
- Docker installed (for building and pushing the training image)
- `.env` populated with MinIO and MLflow credentials (see `.env.example`)

## One-Time GKE Setup

### 1. Enable required APIs

```bash
gcloud services enable container.googleapis.com artifactregistry.googleapis.com
```

### 2. Create the GKE cluster

```bash
gcloud container clusters create protein-benchmark \
  --zone us-central1-a \
  --num-nodes 1 \
  --machine-type e2-small \
  --no-enable-autoupgrade
```

The default node pool is CPU-only (for Argo control plane pods). GPU nodes are a separate pool.

### 3. Create the GPU spot node pool (scales to 0 when idle)

```bash
gcloud container node-pools create gpu-spot \
  --cluster protein-benchmark \
  --zone us-central1-a \
  --machine-type n1-standard-4 \
  --accelerator type=nvidia-tesla-t4,count=1,gpu-driver-installation-config=google-managed \
  --spot \
  --num-nodes 0 \
  --enable-autoscaling \
  --min-nodes 0 \
  --max-nodes 2
```

The `google-managed` GPU driver flag tells GKE to automatically install NVIDIA drivers.

### 4. Get cluster credentials

```bash
gcloud container clusters get-credentials protein-benchmark --zone us-central1-a
```

### 5. Install Argo Workflows

```bash
kubectl create namespace argo
kubectl apply -n argo --server-side \
  -f https://github.com/argoproj/argo-workflows/releases/latest/download/quick-start-minimal.yaml
```

### 6. Create the credentials secret

All five env vars from `.env` are needed so the training container can reach MinIO and MLflow:

```bash
source .env && kubectl create secret generic mlflow-minio-credentials \
  --namespace argo \
  --from-literal=AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID \
  --from-literal=AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY \
  --from-literal=MLFLOW_TRACKING_URI=$MLFLOW_TRACKING_URI \
  --from-literal=S3_ENDPOINT_URL=$S3_ENDPOINT_URL \
  --from-literal=MLFLOW_S3_ENDPOINT_URL=$MLFLOW_S3_ENDPOINT_URL
```

### 7. Create the Artifact Registry repository

```bash
gcloud artifacts repositories create protein-benchmark \
  --repository-format docker \
  --location us-central1
```

### 8. Build and push the training image

Your GCP project ID is `modular-scout-486816-g3`:

```bash
gcloud auth configure-docker us-central1-docker.pkg.dev

docker build --platform linux/amd64 \
  -t us-central1-docker.pkg.dev/modular-scout-486816-g3/protein-benchmark/train:latest \
  -f docker/train/Dockerfile .

docker push us-central1-docker.pkg.dev/modular-scout-486816-g3/protein-benchmark/train:latest
```

Update the `image` field in `argo/train-pipeline.yaml` to match your `PROJECT_ID`.

## Submitting a Training Run

### Config requirements

Configs used for remote training must reference data via S3 URIs, not local paths.
Update `data.train_path` and `data.valid_path` in your config to point to MinIO:

```json
"data": {
    "train_path": "s3://data/tape/fluorescence/train.csv",
    "valid_path": "s3://data/tape/fluorescence/valid.csv",
    "dataset_type": "one_hot"
}
```

### Submit

```bash
# Use the default config (tape_fluorescence_mlp_regressor.json)
argo submit -n argo argo/train-pipeline.yaml --watch

# Or specify a different config
argo submit -n argo argo/train-pipeline.yaml \
  -p config=configs/remote/tape_fluorescence_cnn_regressor.json \
  --watch
```

### Argo UI

```bash
kubectl port-forward -n argo svc/argo-server 2746:2746
```

Then open https://localhost:2746 for a visual workflow dashboard.

## Cost Notes

- GPU spot T4 node: ~$0.28/hr (VM + GPU). The pool scales to 0 when no jobs are running.
- CPU default node pool (e2-small): ~$0.017/hr — always running while cluster exists.
- Delete the cluster when not in use for an extended period:
  ```bash
  gcloud container clusters delete protein-benchmark --zone us-central1-a
  ```

## How It Works

- Each `argo submit` creates a K8s Pod on the GPU spot node pool
- The Pod runs `scripts/train.py --config <config>` inside the training container
- Data is read from MinIO via S3 API (resolved by `get_storage_options()` when path starts with `s3://`)
- MLflow run is logged to the `mlflow-server` VM at the URI in `MLFLOW_TRACKING_URI`
- `generateName` gives each run a unique suffix — resubmit without conflicts
