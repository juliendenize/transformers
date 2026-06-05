"""Mistral native format integration: tokenizer conversion utilities."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


__all__ = [
    "MistralConverter",
]

_MODULE = "tokenizer"

_ATTR_TO_MODULE = {
    "MistralConverter": _MODULE,
}


class _LazyModule(ModuleType):
    def __getattr__(self, name: str):
        if name in _ATTR_TO_MODULE:
            module = importlib.import_module(f".{_ATTR_TO_MODULE[name]}", __name__)
            value = getattr(module, name)
            setattr(self, name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__(self):
        return list(__all__) + list(super().__dir__())


sys.modules[__name__].__class__ = _LazyModule
