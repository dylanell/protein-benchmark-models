"""Pytorch datasets for protein sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Paired-sequence datasets (two sequences per sample, e.g. PPI tasks)
# ---------------------------------------------------------------------------

# Standard 20 canonical amino acids + special tokens.
# PAD (0) is used for padding; UNK (1) for non-standard characters.
AA_VOCAB: dict[str, int] = {"<PAD>": 0, "<UNK>": 1}
AA_VOCAB.update({aa: i + 2 for i, aa in enumerate("ACDEFGHIKLMNPQRSTVWY")})


class SequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        targets: list[float] | np.ndarray | pd.Series,
    ):
        """PyTorch Dataset returning raw amino acid sequences as strings.

        Intended for models that perform their own tokenization (e.g. protein
        LLMs from HuggingFace). Each sample is a dict with keys "sequence"
        (str) and "target" (float32 scalar tensor).

        Args:
            sequences: List of amino acid sequences as strings.
            targets: Regression targets, one per sequence.
        """
        super().__init__()
        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "sequence": self.sequences[index],
            "target": self.targets[index],
        }


class TokenizedSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        targets: list[float] | np.ndarray | pd.Series,
        seq_len: int,
    ):
        """PyTorch Dataset returning vocabulary-encoded amino acid sequences.

        Sequences are tokenized using AA_VOCAB lazily in __getitem__. Sequences
        longer than seq_len are trimmed from the right; shorter sequences are
        right-padded with PAD (index 0).

        Each sample is a dict with keys "tokens" (int64 tensor of shape
        (seq_len,)) and "target" (float32 scalar tensor).

        Args:
            sequences: List of amino acid sequences as strings.
            targets: Regression targets, one per sequence.
            seq_len: Fixed length every sequence is padded/trimmed to.
        """
        super().__init__()
        self.seq_len = seq_len
        self.vocab = AA_VOCAB

        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def _encode(self, sequence: str) -> torch.Tensor:
        ids = [
            self.vocab.get(aa, self.vocab["<UNK>"])
            for aa in sequence[: self.seq_len]
        ]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        tokens = torch.tensor(ids, dtype=torch.long)
        return tokens

    def _decode(self, tokens: torch.Tensor) -> str:
        inv_vocab = {i: aa for aa, i in self.vocab.items()}
        sequence = "".join(
            inv_vocab[i] for i in tokens.tolist() if i != self.vocab["<PAD>"]
        )
        return sequence

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "tokens": self._encode(self.sequences[index]),
            "target": self.targets[index],
        }


class OneHotSequenceDataset(Dataset):
    def __init__(
        self,
        sequences: list[str],
        targets: list[float] | np.ndarray | pd.Series,
        seq_len: int,
    ):
        """PyTorch Dataset returning one-hot-encoded amino acid sequences.

        Sequences are tokenized and one-hot encoded using AA_VOCAB lazily in
        __getitem__. Sequences longer than seq_len are trimmed from the right;
        shorter sequences are right-padded with PAD (index 0).

        Each sample is a dict with keys "one_hots" (float32 tensor of shape
        (seq_len, vocab_size)) and "target" (float32 scalar tensor).

        Args:
            sequences: List of amino acid sequences as strings.
            targets: Regression targets, one per sequence.
            seq_len: Fixed length every sequence is padded/trimmed to.
        """
        super().__init__()
        self.seq_len = seq_len
        self.vocab = AA_VOCAB

        self.sequences = sequences
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def _encode(self, sequence: str) -> torch.Tensor:
        ids = [
            self.vocab.get(aa, self.vocab["<UNK>"])
            for aa in sequence[: self.seq_len]
        ]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        one_hots = torch.eye(len(self.vocab))[
            torch.tensor(ids, dtype=torch.long)
        ]
        return one_hots

    def _decode(self, one_hots: torch.Tensor) -> str:
        inv_vocab = {i: aa for aa, i in self.vocab.items()}
        ids = torch.argmax(one_hots, dim=-1).tolist()
        return "".join(inv_vocab[i] for i in ids if i != self.vocab["<PAD>"])

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "one_hots": self._encode(self.sequences[index]),
            "target": self.targets[index],
        }


class PairedSequenceDataset(Dataset):
    def __init__(
        self,
        sequences_a: list[str],
        sequences_b: list[str],
        targets: list[float] | np.ndarray | pd.Series,
    ):
        """PyTorch Dataset returning raw amino acid sequence pairs as strings.

        Intended for models that perform their own tokenization (e.g. protein
        LLMs). Each sample is a dict with keys "sequence_a" (str),
        "sequence_b" (str), and "target" (float32 scalar tensor).

        Args:
            sequences_a: First sequence of each pair.
            sequences_b: Second sequence of each pair.
            targets: Targets (regression or binary classification), one per
                pair.
        """
        super().__init__()
        if len(sequences_a) != len(sequences_b) or len(sequences_a) != len(
            targets
        ):
            raise ValueError(
                "sequences_a, sequences_b, and targets must have the same "
                f"length; got {len(sequences_a)}, {len(sequences_b)}, "
                f"{len(targets)}"
            )
        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "sequence_a": self.sequences_a[index],
            "sequence_b": self.sequences_b[index],
            "target": self.targets[index],
        }


class PairedTokenizedSequenceDataset(Dataset):
    def __init__(
        self,
        sequences_a: list[str],
        sequences_b: list[str],
        targets: list[float] | np.ndarray | pd.Series,
        seq_len: int,
    ):
        """PyTorch Dataset returning vocabulary-encoded amino acid sequence
        pairs.

        Both sequences are tokenized using AA_VOCAB lazily in __getitem__.
        Sequences longer than seq_len are trimmed from the right; shorter
        sequences are right-padded with PAD (index 0).

        Each sample is a dict with keys "tokens_a" (int64 tensor of shape
        (seq_len,)), "tokens_b" (int64 tensor of shape (seq_len,)), and
        "target" (float32 scalar tensor).

        Args:
            sequences_a: First sequence of each pair.
            sequences_b: Second sequence of each pair.
            targets: Targets (regression or binary classification), one per
                pair.
            seq_len: Fixed length every sequence is padded/trimmed to.
        """
        super().__init__()
        if len(sequences_a) != len(sequences_b) or len(sequences_a) != len(
            targets
        ):
            raise ValueError(
                "sequences_a, sequences_b, and targets must have the same "
                f"length; got {len(sequences_a)}, {len(sequences_b)}, "
                f"{len(targets)}"
            )
        self.seq_len = seq_len
        self.vocab = AA_VOCAB
        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def _encode(self, sequence: str) -> torch.Tensor:
        ids = [
            self.vocab.get(aa, self.vocab["<UNK>"])
            for aa in sequence[: self.seq_len]
        ]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        return torch.tensor(ids, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "tokens_a": self._encode(self.sequences_a[index]),
            "tokens_b": self._encode(self.sequences_b[index]),
            "target": self.targets[index],
        }


class PairedOneHotSequenceDataset(Dataset):
    def __init__(
        self,
        sequences_a: list[str],
        sequences_b: list[str],
        targets: list[float] | np.ndarray | pd.Series,
        seq_len: int,
    ):
        """PyTorch Dataset returning one-hot-encoded amino acid sequence pairs.

        Both sequences are tokenized and one-hot encoded using AA_VOCAB lazily
        in __getitem__. Sequences longer than seq_len are trimmed from the
        right; shorter sequences are right-padded with PAD (index 0).

        Each sample is a dict with keys "one_hots_a" (float32 tensor of shape
        (seq_len, vocab_size)), "one_hots_b" (float32 tensor of shape
        (seq_len, vocab_size)), and "target" (float32 scalar tensor).

        Args:
            sequences_a: First sequence of each pair.
            sequences_b: Second sequence of each pair.
            targets: Targets (regression or binary classification), one per
                pair.
            seq_len: Fixed length every sequence is padded/trimmed to.
        """
        super().__init__()
        if len(sequences_a) != len(sequences_b) or len(sequences_a) != len(
            targets
        ):
            raise ValueError(
                "sequences_a, sequences_b, and targets must have the same "
                f"length; got {len(sequences_a)}, {len(sequences_b)}, "
                f"{len(targets)}"
            )
        self.seq_len = seq_len
        self.vocab = AA_VOCAB
        self.sequences_a = sequences_a
        self.sequences_b = sequences_b
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def _encode(self, sequence: str) -> torch.Tensor:
        ids = [
            self.vocab.get(aa, self.vocab["<UNK>"])
            for aa in sequence[: self.seq_len]
        ]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        return torch.eye(len(self.vocab))[torch.tensor(ids, dtype=torch.long)]

    def __len__(self) -> int:
        return len(self.targets)

    def __getitem__(self, index: int) -> dict:
        return {
            "one_hots_a": self._encode(self.sequences_a[index]),
            "one_hots_b": self._encode(self.sequences_b[index]),
            "target": self.targets[index],
        }
