"""Download protein benchmark datasets to a local directory or S3 prefix.

TAPE tasks (one fixed train/valid/test split):
    uv run python scripts/onboard.py --task fluorescence
    uv run python scripts/onboard.py --task stability
    uv run python scripts/onboard.py --task fluorescence --dest s3://data/tape/fluorescence/

FLIP2 tasks (multiple named splits, each gets its own subdirectory):
    uv run python scripts/onboard.py --task amylase
    uv run python scripts/onboard.py --task ired
    uv run python scripts/onboard.py --task nucb
    uv run python scripts/onboard.py --task hydro
    uv run python scripts/onboard.py --task rhomax --dest s3://data/flip2/fluorescence/

Output structure for FLIP2:
    .data/<task>/<split>/train.csv
    .data/<task>/<split>/valid.csv
    .data/<task>/<split>/test.csv

Bernett PPI task (gold-standard leakage-free human PPI binary classification):
    uv run python scripts/onboard.py --task bernett_ppi

Output structure for bernett_ppi:
    .data/bernett_ppi/train.csv  — columns: sequence_a, sequence_b, target
    .data/bernett_ppi/valid.csv
    .data/bernett_ppi/test.csv

Source: Bernett et al. (2024), Briefings in Bioinformatics.
        https://doi.org/10.6084/m9.figshare.21591618
"""

import logging
import argparse
import io
import json
import tarfile
import urllib.request
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from protein_benchmark_models.utils import get_s3_filesystem

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_SCRIPT = "onboard.py"
TAPE_BASE = "http://s3.amazonaws.com/songlabdata/proteindata/data_raw_pytorch"
ZENODO_BASE = "https://zenodo.org/records/18433203/files"
FIGSHARE_API = "https://api.figshare.com/v2/articles"

BERNETT_PPI_ARTICLE_ID = "21591618"
# Intra-1 = train, Intra-0 = valid, Intra-2 = test (paper convention)
BERNETT_PPI_SPLITS = {
    "train": "Intra1",
    "valid": "Intra0",
    "test": "Intra2",
}

TAPE_TASKS = {
    "fluorescence": {
        "url": f"{TAPE_BASE}/fluorescence.tar.gz",
        "target_field": "log_fluorescence",
        "extra_fields": ["num_mutations"],
    },
    "stability": {
        "url": f"{TAPE_BASE}/stability.tar.gz",
        "target_field": "stability_score",
        "extra_fields": [],
    },
}

FLIP2_TASKS = {
    "amylase": [
        "one_to_many",
        "close_to_far",
        "far_to_close",
        "by_mutation",
        "random_split",
    ],
    "ired": ["two_to_many", "random"],
    "nucb": ["two_to_many", "random"],
    "hydro": [
        "three_to_many",
        "low_to_high",
        "to_P06241",
        "to_P0A9X9",
        "to_P01053",
        "random_split",
    ],
    "rhomax": ["by_wild_type"],
}

SPLITS = ["train", "valid", "test"]


def parse_tape_json(
    data: bytes, target_field: str, extra_fields: list
) -> pd.DataFrame:
    """Parse a TAPE JSON file (array of records) into a DataFrame.

    Maps `primary` → `sequence` and `{target_field}` → `target`.
    Handles targets stored as a [float] list or plain scalar.
    Includes any extra_fields that exist in the record.
    """
    records = json.loads(data.decode("utf-8"))
    rows = []
    for record in records:
        target = record[target_field]
        if isinstance(target, list):
            target = target[0]
        row = {"sequence": record["primary"], "target": float(target)}
        for field in extra_fields:
            if field in record:
                row[field] = record[field]
        rows.append(row)
    return pd.DataFrame(rows)


def find_tar_member(
    tar: tarfile.TarFile, task_name: str, split: str
) -> tarfile.ExFileObject:
    """Return the file object for *_{split}.json inside the tarball."""
    suffix = f"_{split}.json"
    for member in tar.getmembers():
        if member.name.endswith(suffix):
            f = tar.extractfile(member)
            if f is not None:
                return f
    raise FileNotFoundError(
        f"No member matching '*{suffix}' found in {task_name} tarball. "
        f"Available: {[m.name for m in tar.getmembers()]}"
    )


def write_df(df: pd.DataFrame, path: str) -> None:
    """Write DataFrame as CSV to a local path or S3 path."""
    if path.startswith("s3://"):
        fs = get_s3_filesystem()
        with fs.open(path, "w") as fp:
            fp.write(df.to_csv(index=False))
    else:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)


def onboard_tape_task(task_name: str, dest: str) -> None:
    """Download and extract a TAPE task, writing one CSV per split."""
    task = TAPE_TASKS[task_name]
    url = task["url"]
    target_field = task["target_field"]
    extra_fields = task["extra_fields"]

    logger.info(f"[{_SCRIPT}] Downloading {task_name} from {url} ...")
    with urllib.request.urlopen(url) as response:
        tar_bytes = response.read()
    logger.info(f"[{_SCRIPT}] Downloaded {len(tar_bytes) / 1_000_000:.1f} MB")

    dest = dest.rstrip("/")
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for split in SPLITS:
            f = find_tar_member(tar, task_name, split)
            df = parse_tape_json(f.read(), target_field, extra_fields)
            out_path = f"{dest}/{split}.csv"
            write_df(df, out_path)
            logger.info(
                f"[{_SCRIPT}] Saved {split}.csv"
                f" ({len(df):,} rows) to {out_path}"
            )


def onboard_flip2_task(task_name: str, dest: str) -> None:
    """Download all splits for a FLIP2 task, writing CSVs per split.

    Each split is fetched as a .csv.gz from Zenodo and written to:
        <dest>/<split>/train.csv   — set=="train" and validation==False
        <dest>/<split>/valid.csv   — set=="train" and validation==True
        <dest>/<split>/test.csv    — set=="test"
    """
    splits = FLIP2_TASKS[task_name]
    dest = dest.rstrip("/")

    for split in splits:
        url = f"{ZENODO_BASE}/{task_name}/{split}.csv.gz?download=1"
        logger.info(
            f"[{_SCRIPT}] Downloading {task_name}/{split} from {url} ..."
        )
        with urllib.request.urlopen(url) as response:
            data = response.read()
        logger.info(f"[{_SCRIPT}] Downloaded {len(data) / 1_000_000:.1f} MB")

        df = pd.read_csv(io.BytesIO(data), compression="gzip")

        train_df = df[(df["set"] == "train") & ~df["validation"]][
            ["sequence", "target"]
        ].reset_index(drop=True)
        valid_df = df[(df["set"] == "train") & df["validation"]][
            ["sequence", "target"]
        ].reset_index(drop=True)
        test_df = df[df["set"] == "test"][["sequence", "target"]].reset_index(
            drop=True
        )

        split_dest = f"{dest}/{split}"
        for split_name, split_df in [
            ("train", train_df),
            ("valid", valid_df),
            ("test", test_df),
        ]:
            out_path = f"{split_dest}/{split_name}.csv"
            write_df(split_df, out_path)
            logger.info(
                f"[{_SCRIPT}] Saved {split_name}.csv"
                f" ({len(split_df):,} rows) to {out_path}"
            )


def get_figshare_files(article_id: str) -> dict[str, str]:
    """Return {filename: download_url} for all files in a figshare article."""
    url = f"{FIGSHARE_API}/{article_id}/files"
    with urllib.request.urlopen(url) as response:
        files = json.loads(response.read().decode("utf-8"))
    return {f["name"]: f["download_url"] for f in files}


def parse_fasta_oneliner(data: bytes) -> dict[str, str]:
    """Parse a one-liner FASTA file into a {uniprot_id: sequence} dict.

    Expects headers in Swiss-Prot format: >sp|<ID>|<NAME> ...
    Falls back to the first whitespace-delimited token for other formats.
    """
    seqs: dict[str, str] = {}
    current_id: str | None = None
    for line in data.decode("utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith(">"):
            header = line[1:]
            parts = header.split("|")
            current_id = parts[1] if len(parts) >= 3 else header.split()[0]
        elif current_id is not None:
            seqs[current_id] = line
            current_id = None
    return seqs


def onboard_bernett_ppi(dest: str) -> None:
    """Download the Bernett et al. PPI gold-standard dataset from figshare.

    Fetches:
      - human_swissprot_oneliner.fasta — protein sequences
      - Intra{0,1,2}_{pos,neg}_rr.txt — UniProt ID pairs per split

    Writes one CSV per split (train/valid/test) with columns:
      sequence_a, sequence_b, target (1 = interacting, 0 = non-interacting)

    Pairs for which either protein is absent from the FASTA are silently
    dropped (this should not occur with the bundled FASTA but guards against
    corrupt downloads).
    """
    logger.info(f"[{_SCRIPT}] Fetching file list from figshare ...")
    files = get_figshare_files(BERNETT_PPI_ARTICLE_ID)

    logger.info(f"[{_SCRIPT}] Downloading sequence FASTA ...")
    with urllib.request.urlopen(
        files["human_swissprot_oneliner.fasta"]
    ) as response:
        fasta_bytes = response.read()
    logger.info(
        f"[{_SCRIPT}] Downloaded {len(fasta_bytes) / 1_000_000:.1f} MB"
    )
    seqs = parse_fasta_oneliner(fasta_bytes)
    logger.info(f"[{_SCRIPT}] Parsed {len(seqs):,} sequences from FASTA")

    dest = dest.rstrip("/")
    for split_name, intra in BERNETT_PPI_SPLITS.items():
        rows = []
        missing = 0
        for label, target in [("pos", 1), ("neg", 0)]:
            filename = f"{intra}_{label}_rr.txt"
            logger.info(
                f"[{_SCRIPT}] Downloading {filename} ..."
            )
            with urllib.request.urlopen(files[filename]) as response:
                text = response.read().decode("utf-8")
            for line in text.strip().splitlines():
                line = line.strip()
                if not line:
                    continue
                id_a, id_b = line.split()
                seq_a = seqs.get(id_a)
                seq_b = seqs.get(id_b)
                if seq_a is None or seq_b is None:
                    missing += 1
                    continue
                rows.append(
                    {
                        "sequence_a": seq_a,
                        "sequence_b": seq_b,
                        "target": target,
                    }
                )
        if missing:
            logger.warning(
                f"[{_SCRIPT}] {missing} pairs dropped "
                f"(sequence not found in FASTA)"
            )
        df = pd.DataFrame(rows)
        out_path = f"{dest}/{split_name}.csv"
        write_df(df, out_path)
        logger.info(
            f"[{_SCRIPT}] Saved {split_name}.csv"
            f" ({len(df):,} rows) to {out_path}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Download protein benchmark datasets."
    )
    parser.add_argument(
        "--task",
        required=True,
        choices=list(TAPE_TASKS.keys()) + list(FLIP2_TASKS.keys()) + [
            "bernett_ppi"
        ],
        help="Task to download",
    )
    parser.add_argument(
        "--dest",
        default=None,
        help="Destination (local or s3://). Defaults to .data/<task>/",
    )
    args = parser.parse_args()

    if args.task in TAPE_TASKS:
        dest = args.dest if args.dest is not None else f".data/tape/{args.task}"
        onboard_tape_task(args.task, dest)
    elif args.task in FLIP2_TASKS:
        dest = (
            args.dest if args.dest is not None else f".data/flip2/{args.task}"
        )
        onboard_flip2_task(args.task, dest)
    else:  # bernett_ppi
        dest = (
            args.dest if args.dest is not None else ".data/bernett_ppi"
        )
        onboard_bernett_ppi(dest)


if __name__ == "__main__":
    main()
