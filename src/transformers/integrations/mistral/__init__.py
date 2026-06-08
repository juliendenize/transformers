"""Mistral native format integration: config, weight, and tokenizer conversion."""

from __future__ import annotations

import importlib
import sys
from types import ModuleType


__all__ = [
    "MistralConverter",
    "MistralFormatConfig",
    "MistralNativeConfig",
    "_hf_config_to_native_config",
    "_native_config_to_hf_config",
    "_parse_native_config_from_dict",
    "convert_state_dict_to_native",
    "convert_tekken_image_processor",
    "convert_tekken_tokenizer",
    "mistral3_native_text_converters",
    "mistral3_native_text_renamings",
    "mistral3_native_vision_converters",
    "mistral3_native_vision_renamings",
    "mistral4_native_converters",
    "mistral4_native_renamings",
    "mistral_base_native_converters",
    "mistral_base_native_renamings",
    "native_config_for_model_type",
    "native_config_from_hf_config",
    "resolve_mistral_format",
    "save_as_tekken",
    "save_native_mistral_format",
]

_TOKENIZER_MODULE = "tokenizer"
_PARAMS_MODULE = "params_conversion"
_CONFIG_FORMAT_MODULE = "config_format"
_WEIGHT_MODULE = "weight_conversion"

_ATTR_TO_MODULE = {
    "MistralConverter": _TOKENIZER_MODULE,
    "MistralFormatConfig": _CONFIG_FORMAT_MODULE,
    "MistralNativeConfig": _PARAMS_MODULE,
    "_hf_config_to_native_config": _PARAMS_MODULE,
    "_native_config_to_hf_config": _PARAMS_MODULE,
    "_parse_native_config_from_dict": _PARAMS_MODULE,
    "convert_state_dict_to_native": _WEIGHT_MODULE,
    "convert_tekken_image_processor": _TOKENIZER_MODULE,
    "convert_tekken_tokenizer": _TOKENIZER_MODULE,
    "mistral3_native_text_converters": _WEIGHT_MODULE,
    "mistral3_native_text_renamings": _WEIGHT_MODULE,
    "mistral3_native_vision_converters": _WEIGHT_MODULE,
    "mistral3_native_vision_renamings": _WEIGHT_MODULE,
    "mistral4_native_converters": _WEIGHT_MODULE,
    "mistral4_native_renamings": _WEIGHT_MODULE,
    "mistral_base_native_converters": _WEIGHT_MODULE,
    "mistral_base_native_renamings": _WEIGHT_MODULE,
    "native_config_for_model_type": _PARAMS_MODULE,
    "native_config_from_hf_config": _PARAMS_MODULE,
    "resolve_mistral_format": _TOKENIZER_MODULE,
    "save_as_tekken": _TOKENIZER_MODULE,
    "save_native_mistral_format": _WEIGHT_MODULE,
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
