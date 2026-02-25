"""Base dataset interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from torch.utils.data import DataLoader


class BaseDataset(ABC):
    """Abstract base class for all datasets."""

    @abstractmethod
    def split(
        self, test_size: float = 0.2, random_state: int | None = None
    ) -> tuple[BaseDataset, BaseDataset]:
        """Split into train and test datasets."""
        raise NotImplementedError

    @abstractmethod
    def to_pytorch(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Convert to PyTorch DataLoader."""
        raise NotImplementedError

    @abstractmethod
    def __len__(self) -> int:
        """Return the number of samples."""
        raise NotImplementedError
