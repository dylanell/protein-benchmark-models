"""Pytorch datasets for protein sequences."""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

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
        ids = [self.vocab.get(aa, self.vocab["<UNK>"]) for aa in sequence[: self.seq_len]]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        tokens = torch.tensor(ids, dtype=torch.long)
        return tokens

    def _decode(self, tokens: torch.Tensor) -> str:
        inv_vocab = {i: aa for aa, i in self.vocab.items()}
        sequence = "".join(inv_vocab[i] for i in tokens.tolist() if i != self.vocab["<PAD>"])
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
        ids = [self.vocab.get(aa, self.vocab["<UNK>"]) for aa in sequence[: self.seq_len]]
        if len(ids) < self.seq_len:
            ids += [self.vocab["<PAD>"]] * (self.seq_len - len(ids))
        one_hots = torch.eye(len(self.vocab))[torch.tensor(ids, dtype=torch.long)]
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