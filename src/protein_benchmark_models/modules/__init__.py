"""Pytorch modules package."""

from protein_benchmark_models.modules.fully_connected import FullyConnected
from protein_benchmark_models.modules.sequence_cnn import SequenceCNN
from protein_benchmark_models.modules.utils import Transpose

__all__ = ["FullyConnected", "SequenceCNN", "Transpose"]
