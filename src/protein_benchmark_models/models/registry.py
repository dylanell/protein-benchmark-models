"""Model registry for discovering and instantiating models."""

from __future__ import annotations

import json
import os

from .base import BaseModel


class ModelRegistry:
    """Registry for model classes."""

    _models: dict[str, type[BaseModel]] = {}

    @classmethod
    def list(cls) -> list[str]:
        """List available model names."""
        return list(cls._models.keys())

    @classmethod
    def get(cls, name: str) -> type[BaseModel]:
        """Get model class by name."""
        if name not in cls._models:
            raise ValueError(f"Unknown model: {name}. Available: {cls.list()}")
        return cls._models[name]

    @classmethod
    def load(cls, path: str) -> BaseModel:
        """Load a model from a directory containing config.json and weights."""
        with open(os.path.join(path, "config.json")) as f:
            config = json.load(f)
        model_class = cls.get(config["model_name"])
        return model_class.load(path)


def register(cls: type[BaseModel]) -> type[BaseModel]:
    """Class decorator that registers a model in ModelRegistry.

    Usage::

        @register
        class MyModel(BaseModel):
            model_name = "my_model"
            ...
    """
    ModelRegistry._models[cls.model_name] = cls
    return cls
