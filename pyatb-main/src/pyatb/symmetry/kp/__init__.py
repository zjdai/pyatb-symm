from __future__ import annotations

import sys
import types as _types

from . import model as _model


def _export_model_names() -> None:
    for name in dir(_model):
        if not name.startswith("__"):
            globals()[name] = getattr(_model, name)


class _KPPackageModule(_types.ModuleType):
    def __setattr__(self, name, value):
        super().__setattr__(name, value)
        if hasattr(_model, name):
            setattr(_model, name, value)


_export_model_names()
sys.modules[__name__].__class__ = _KPPackageModule

__all__ = [name for name in globals() if not name.startswith("__")]
