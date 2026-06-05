"""Mistral native format integration: config, weight, and tokenizer conversion."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


__all__ = [
    "MistralConverter",
    "convert_tekken_image_processor",
    "convert_tekken_tokenizer",
    "resolve_mistral_format",
    "save_as_tekken",
]

_MODULE = "tokenizer"

_ATTR_TO_MODULE = {
    "MistralConverter": _MODULE,
    "convert_tekken_image_processor": _MODULE,
    "convert_tekken_tokenizer": _MODULE,
    "resolve_mistral_format": _MODULE,
    "save_as_tekken": _MODULE,
}


class _LazyModule(ModuleType):
    """Lazily re-export public names to break circular imports."""

    def __getattr__(self, name: str):
        if name in _ATTR_TO_MODULE:
            submodule = importlib.import_module(f".{_ATTR_TO_MODULE[name]}", __name__)
            value = getattr(submodule, name)
            setattr(self, name, value)
            return value
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    def __dir__(self):
        return list(__all__) + list(super().__dir__())


sys.modules[__name__].__class__ = _LazyModule
