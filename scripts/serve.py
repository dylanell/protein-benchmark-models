"""Config-driven serving script for Iris classifiers.

Loads a trained model from the path in the config and exposes it
via a FastAPI server with /health, /info, and /predict endpoints.

Usage:
    uv run python scripts/serve.py --config configs/iris_mlp_classifier.json
    uv run python scripts/serve.py --config configs/iris_gb_classifier.json
"""

import argparse
import json
import sys

import uvicorn
from dotenv import load_dotenv

from ml_project_template.serving.app import create_app

load_dotenv()


def main():
    parser = argparse.ArgumentParser(description="Serve an Iris classifier from a JSON config.")
    parser.add_argument("--config", required=True, help="Path to JSON config file")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to (default: 8000)")
    args = parser.parse_args()

    with open(args.config) as f:
        config = json.load(f)

    print(f"[serve] Running with config:")
    print(json.dumps(config, indent=2))

    for key in ("data", "model", "training"):
        if key not in config:
            print(f"[serve] Error: config missing required key '{key}'")
            sys.exit(1)

    app = create_app(config)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
