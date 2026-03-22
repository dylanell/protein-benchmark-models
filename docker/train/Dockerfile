FROM nvidia/cuda:12.1.0-cudnn8-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive

# Install Python 3.11 (not in Ubuntu 22.04 default repos, requires deadsnakes)
RUN apt-get update && apt-get install -y software-properties-common curl \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update && apt-get install -y \
        python3.11 python3.11-venv python3.11-dev \
    && rm -rf /var/lib/apt/lists/*

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

ENV UV_PYTHON=/usr/bin/python3.11

WORKDIR /app

# Install dependencies first (cached layer)
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project

# Copy project source and install
COPY README.md ./
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/
RUN uv sync --frozen

# Silence git warnings
ENV GIT_PYTHON_REFRESH=quiet

ENTRYPOINT ["uv", "run", "python", "scripts/train.py"]
