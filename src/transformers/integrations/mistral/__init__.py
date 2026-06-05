"""Mistral native format integration: tokenizer and config conversion utilities."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


__all__ = [
    "MistralConverter",
    "MistralNativeConfig",
    "_hf_config_to_native_config",
    "_native_config_to_hf_config",
    "_parse_native_config_from_dict",
    "native_config_for_model_type",
    "native_config_from_hf_config",
]

_TOKENIZER_MODULE = "tokenizer"
_PARAMS_MODULE = "params_conversion"

_ATTR_TO_MODULE = {
    "MistralConverter": _TOKENIZER_MODULE,
    "MistralNativeConfig": _PARAMS_MODULE,
    "_hf_config_to_native_config": _PARAMS_MODULE,
    "_native_config_to_hf_config": _PARAMS_MODULE,
    "_parse_native_config_from_dict": _PARAMS_MODULE,
    "native_config_for_model_type": _PARAMS_MODULE,
    "native_config_from_hf_config": _PARAMS_MODULE,
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
