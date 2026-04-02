"""Shared test fixtures and helpers."""

import numpy as np
import pytest

from protein_benchmark_models.data import (
    AA_VOCAB,
    SequenceDataset,
    OneHotSequenceDataset,
    TokenizedSequenceDataset,
    PairedOneHotSequenceDataset,
)

# Small fixed sequences for fast, deterministic tests.
# Includes: exact-length, short (needs padding), long (needs trimming),
# unknown AA.
SEQ_LEN = 8
SEQUENCES = [
    "ACDEFGHI",  # exact length, all known AAs
    "ACDX",  # short — will be padded; 'X' is unknown
    "ACDEFGHIKLMN",  # long — will be trimmed
    "KLMNPQRS",
    "RSTVWYAC",
    "ACKLMNPQ",
    "DEFGHIKL",
    "MNPQRSTV",
]
TARGETS = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]
VOCAB_SIZE = len(AA_VOCAB)  # 22


def onehot_X(dataset):
    """Build a flattened one-hot input array (N, SEQ_LEN * VOCAB_SIZE) from
    an OneHotSequenceDataset."""
    return np.stack(
        [dataset[i]["one_hots"].numpy().flatten() for i in range(len(dataset))]
    )


def token_X(dataset):
    """Build a token index array (N, SEQ_LEN) from a
    TokenizedSequenceDataset."""
    return np.stack([dataset[i]["tokens"].numpy() for i in range(len(dataset))])


# Binary targets for paired classification fixtures (alternating 0/1).
BINARY_TARGETS = [float(i % 2) for i in range(len(SEQUENCES))]


def paired_onehot_X(dataset):
    """Return (X_a, X_b) flattened one-hot arrays from a
    PairedOneHotSequenceDataset."""
    X_a = np.stack(
        [dataset[i]["one_hots_a"].numpy().flatten() for i in range(len(dataset))]
    )
    X_b = np.stack(
        [dataset[i]["one_hots_b"].numpy().flatten() for i in range(len(dataset))]
    )
    return X_a, X_b


@pytest.fixture
def sequence_data():
    return SequenceDataset(sequences=SEQUENCES, targets=TARGETS)


@pytest.fixture
def onehot_data():
    return OneHotSequenceDataset(
        sequences=SEQUENCES, targets=TARGETS, seq_len=SEQ_LEN
    )


@pytest.fixture
def tokenized_data():
    return TokenizedSequenceDataset(
        sequences=SEQUENCES, targets=TARGETS, seq_len=SEQ_LEN
    )


@pytest.fixture
def paired_onehot_data():
    return PairedOneHotSequenceDataset(
        sequences_a=SEQUENCES,
        sequences_b=SEQUENCES[::-1],
        targets=BINARY_TARGETS,
        seq_len=SEQ_LEN,
    )
