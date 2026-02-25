"""Download the Iris dataset to a local directory or S3 prefix.

Usage:
    uv run python scripts/onboard.py
    uv run python scripts/onboard.py --dest s3://data/iris/
"""

import argparse
import urllib.request
from pathlib import Path

from dotenv import load_dotenv

from ml_project_template.utils import get_s3_filesystem

load_dotenv()

DATA_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
COLUMN_NAMES = ["sepal_length", "sepal_width", "petal_length", "petal_width", "species"]
FILENAME = "iris.csv"


def main():
    parser = argparse.ArgumentParser(description="Download the Iris dataset.")
    parser.add_argument("--dest", default=".data/iris", help="Destination directory (local or s3://)")
    args = parser.parse_args()

    # Download raw data
    print(f"Downloading from '{DATA_URL}'")
    response = urllib.request.urlopen(DATA_URL)
    raw_data = response.read().decode("utf-8").strip()
    csv_content = ",".join(COLUMN_NAMES) + "\n" + raw_data + "\n"

    # Write to destination
    dest = args.dest.rstrip("/")
    file_path = f"{dest}/{FILENAME}"

    if dest.startswith("s3://"):
        fs = get_s3_filesystem()
        with fs.open(file_path, "w") as fp:
            fp.write(csv_content)
    else:
        Path(dest).mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as fp:
            fp.write(csv_content)

    print(f"Saved {FILENAME} to {dest}/")


if __name__ == "__main__":
    main()
