"""Tabular dataset implementation."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

from ml_project_template.data.base import BaseDataset


@dataclass
class TabularDataset(BaseDataset):
    """Dataset for tabular/numerical data."""

    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    class_names: list[str]
    label_encoder: LabelEncoder | None = None

    @classmethod
    def from_csv(cls, path: str, target_column: str, storage_options: dict | None = None) -> TabularDataset:
        """Load dataset from CSV file (local or S3)."""
        df = pd.read_csv(path, storage_options=storage_options or {})

        feature_names = [col for col in df.columns if col != target_column]
        X = df[feature_names].values

        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(df[target_column])
        class_names = list(label_encoder.classes_)

        return cls(
            X=X,
            y=y,
            feature_names=feature_names,
            class_names=class_names,
            label_encoder=label_encoder,
        )

    def split(
        self, test_size: float = 0.2, random_state: int | None = None
    ) -> tuple[TabularDataset, TabularDataset]:
        """Split into train and test datasets."""
        X_train, X_test, y_train, y_test = train_test_split(
            self.X, self.y, test_size=test_size, random_state=random_state
        )

        train = TabularDataset(
            X=X_train,
            y=y_train,
            feature_names=self.feature_names,
            class_names=self.class_names,
            label_encoder=self.label_encoder,
        )
        test = TabularDataset(
            X=X_test,
            y=y_test,
            feature_names=self.feature_names,
            class_names=self.class_names,
            label_encoder=self.label_encoder,
        )
        return train, test

    def to_pytorch(self, batch_size: int = 32, shuffle: bool = True) -> DataLoader:
        """Convert to PyTorch DataLoader."""
        X_tensor = torch.from_numpy(self.X).float()
        y_tensor = torch.from_numpy(self.y).long()
        tensor_dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(tensor_dataset, batch_size=batch_size, shuffle=shuffle)

    def __len__(self) -> int:
        """Return the number of samples."""
        return len(self.X)
