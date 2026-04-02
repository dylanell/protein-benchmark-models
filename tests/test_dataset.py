"""Tests for sequence dataset classes.

Covers:
- SequenceDataset: raw string passthrough, correct item types and values.
- TokenizedSequenceDataset: shape, dtype, padding, trimming, vocab encoding.
- OneHotSequenceDataset: shape, dtype, valid one-hot rows, padding, trimming,
  encoding.
"""

import torch
import pytest

from protein_benchmark_models.data import (
    AA_VOCAB,
    PairedSequenceDataset,
    PairedTokenizedSequenceDataset,
    PairedOneHotSequenceDataset,
)
from tests.conftest import (
    SEQ_LEN,
    SEQUENCES,
    TARGETS,
    BINARY_TARGETS,
    VOCAB_SIZE,
)


class TestSequenceDataset:
    def test_len(self, sequence_data):
        assert len(sequence_data) == len(SEQUENCES)

    def test_item_types(self, sequence_data):
        item = sequence_data[0]
        assert isinstance(item["sequence"], str)
        assert isinstance(item["target"], torch.Tensor)
        assert item["target"].dtype == torch.float32
        assert item["target"].shape == ()  # scalar

    def test_item_values(self, sequence_data):
        for i, (seq, target) in enumerate(zip(SEQUENCES, TARGETS)):
            item = sequence_data[i]
            assert item["sequence"] == seq
            assert item["target"].item() == pytest.approx(target)


class TestTokenizedSequenceDataset:
    def test_len(self, tokenized_data):
        assert len(tokenized_data) == len(SEQUENCES)

    def test_tokens_shape(self, tokenized_data):
        item = tokenized_data[0]
        assert item["tokens"].shape == (SEQ_LEN,)

    def test_tokens_dtype(self, tokenized_data):
        assert tokenized_data[0]["tokens"].dtype == torch.long

    def test_target_dtype(self, tokenized_data):
        assert tokenized_data[0]["target"].dtype == torch.float32

    def test_padding(self, tokenized_data):
        # SEQUENCES[1] = "ACDX" (4 chars) — positions 4-7 should be PAD (0)
        tokens = tokenized_data[1]["tokens"]
        assert tokens.shape == (SEQ_LEN,)
        assert (tokens[4:] == 0).all()

    def test_trimming(self, tokenized_data):
        # SEQUENCES[2] = "ACDEFGHIKLMN" (12 chars) — trimmed to SEQ_LEN
        tokens = tokenized_data[2]["tokens"]
        assert tokens.shape == (SEQ_LEN,)
        # Last token: "ACDEFGHI"[7] = 'I' → AA_VOCAB['I']
        assert tokens[-1].item() == AA_VOCAB["I"]

    def test_known_aa_encoding(self, tokenized_data):
        # SEQUENCES[0] = "ACDEFGHI" — first token should be AA_VOCAB['A']
        tokens = tokenized_data[0]["tokens"]
        assert tokens[0].item() == AA_VOCAB["A"]

    def test_unknown_aa_encoding(self, tokenized_data):
        # SEQUENCES[1] = "ACDX" — 'X' at index 3 should map to UNK (1)
        tokens = tokenized_data[1]["tokens"]
        assert tokens[3].item() == AA_VOCAB["<UNK>"]


class TestOneHotSequenceDataset:
    def test_len(self, onehot_data):
        assert len(onehot_data) == len(SEQUENCES)

    def test_one_hots_shape(self, onehot_data):
        item = onehot_data[0]
        assert item["one_hots"].shape == (SEQ_LEN, VOCAB_SIZE)

    def test_one_hots_dtype(self, onehot_data):
        assert onehot_data[0]["one_hots"].dtype == torch.float32

    def test_target_dtype(self, onehot_data):
        assert onehot_data[0]["target"].dtype == torch.float32

    def test_one_hots_valid_rows(self, onehot_data):
        # Every row should be a valid one-hot: exactly one 1.0, rest 0.0
        one_hots = onehot_data[0]["one_hots"]
        row_sums = one_hots.sum(dim=-1)
        assert (row_sums == 1.0).all()

    def test_padding_position(self, onehot_data):
        # SEQUENCES[1] = "ACDX" (4 chars) — padded: 1.0 at PAD index (0)
        one_hots = onehot_data[1]["one_hots"]
        padded_rows = one_hots[4:]
        assert (padded_rows[:, 0] == 1.0).all()
        assert (padded_rows[:, 1:] == 0.0).all()

    def test_trimming(self, onehot_data):
        # SEQUENCES[2] = "ACDEFGHIKLMN" (12 chars) — trimmed to SEQ_LEN rows
        # Last position: "ACDEFGHI"[7] = 'I' → one-hot at AA_VOCAB['I']
        one_hots = onehot_data[2]["one_hots"]
        assert one_hots.shape == (SEQ_LEN, VOCAB_SIZE)
        assert one_hots[-1, AA_VOCAB["I"]].item() == 1.0

    def test_known_aa_encoding(self, onehot_data):
        # SEQUENCES[0][0] = 'A' — one-hot should have 1.0 at AA_VOCAB['A']
        one_hots = onehot_data[0]["one_hots"]
        assert one_hots[0, AA_VOCAB["A"]].item() == 1.0
        assert one_hots[0].sum().item() == 1.0


# ---------------------------------------------------------------------------
# Paired dataset tests
# ---------------------------------------------------------------------------

SEQS_A = SEQUENCES
SEQS_B = SEQUENCES[::-1]


class TestPairedSequenceDataset:
    def setup_method(self):
        self.ds = PairedSequenceDataset(
            sequences_a=SEQS_A,
            sequences_b=SEQS_B,
            targets=BINARY_TARGETS,
        )

    def test_len(self):
        assert len(self.ds) == len(SEQS_A)

    def test_item_keys(self):
        item = self.ds[0]
        assert set(item.keys()) == {"sequence_a", "sequence_b", "target"}

    def test_item_types(self):
        item = self.ds[0]
        assert isinstance(item["sequence_a"], str)
        assert isinstance(item["sequence_b"], str)
        assert item["target"].dtype == torch.float32
        assert item["target"].shape == ()

    def test_values(self):
        for i in range(len(SEQS_A)):
            item = self.ds[i]
            assert item["sequence_a"] == SEQS_A[i]
            assert item["sequence_b"] == SEQS_B[i]
            assert item["target"].item() == pytest.approx(BINARY_TARGETS[i])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError):
            PairedSequenceDataset(
                sequences_a=SEQS_A,
                sequences_b=SEQS_B[:-1],
                targets=BINARY_TARGETS,
            )


class TestPairedTokenizedSequenceDataset:
    def setup_method(self):
        self.ds = PairedTokenizedSequenceDataset(
            sequences_a=SEQS_A,
            sequences_b=SEQS_B,
            targets=BINARY_TARGETS,
            seq_len=SEQ_LEN,
        )

    def test_len(self):
        assert len(self.ds) == len(SEQS_A)

    def test_item_keys(self):
        assert set(self.ds[0].keys()) == {"tokens_a", "tokens_b", "target"}

    def test_shapes(self):
        item = self.ds[0]
        assert item["tokens_a"].shape == (SEQ_LEN,)
        assert item["tokens_b"].shape == (SEQ_LEN,)

    def test_dtypes(self):
        item = self.ds[0]
        assert item["tokens_a"].dtype == torch.long
        assert item["tokens_b"].dtype == torch.long
        assert item["target"].dtype == torch.float32


class TestPairedOneHotSequenceDataset:
    def setup_method(self):
        self.ds = PairedOneHotSequenceDataset(
            sequences_a=SEQS_A,
            sequences_b=SEQS_B,
            targets=BINARY_TARGETS,
            seq_len=SEQ_LEN,
        )

    def test_len(self):
        assert len(self.ds) == len(SEQS_A)

    def test_item_keys(self):
        assert set(self.ds[0].keys()) == {
            "one_hots_a", "one_hots_b", "target"
        }

    def test_shapes(self):
        item = self.ds[0]
        assert item["one_hots_a"].shape == (SEQ_LEN, VOCAB_SIZE)
        assert item["one_hots_b"].shape == (SEQ_LEN, VOCAB_SIZE)

    def test_dtypes(self):
        item = self.ds[0]
        assert item["one_hots_a"].dtype == torch.float32
        assert item["one_hots_b"].dtype == torch.float32
        assert item["target"].dtype == torch.float32

    def test_valid_one_hots(self):
        item = self.ds[0]
        for key in ("one_hots_a", "one_hots_b"):
            assert (item[key].sum(dim=-1) == 1.0).all()

    def test_sequences_are_independent(self):
        # sequences_a and sequences_b are different (reversed), so encodings
        # should differ for most samples.
        item = self.ds[0]
        assert not torch.equal(item["one_hots_a"], item["one_hots_b"])
