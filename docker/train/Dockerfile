FROM python:3.11-slim

# Install uv
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

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
