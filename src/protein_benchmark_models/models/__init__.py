"""Models package.

On import, all model modules in this package are automatically discovered and
imported. Any class decorated with @register is added to ModelRegistry at that
point, so contributors only need to decorate their class — no manual edits to
this file or registry.py are required.
"""

import importlib
import pkgutil
from pathlib import Path

from .base import BaseModel
from .registry import ModelRegistry, register

_package_dir = Path(__file__).parent
for _, _module_name, _ in pkgutil.iter_modules([str(_package_dir)]):
    if _module_name not in ("base", "registry"):
        importlib.import_module(f".{_module_name}", package=__name__)

__all__ = ["BaseModel", "ModelRegistry", "register"]
