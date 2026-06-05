# Copyright 2026 Mistral AI and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Config format detection and loading for Mistral native `params.json`.

Provides `MistralFormatConfig`, a `PreTrainedConfig` subclass that overrides
`get_config_dict()` to auto-detect and load from `params.json` when `config.json`
is absent. Mistral-family config classes inherit from this to gain automatic
native format support.
"""

import dataclasses
import json
import os
from typing import Any

from huggingface_hub import file_exists as hf_hub_file_exists
from huggingface_hub.constants import SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE

from ...configuration_utils import PreTrainedConfig
from ...utils import cached_file, logging


logger = logging.get_logger(__name__)

# Weight file names
_CONSOLIDATED_SINGLE_FILE = "consolidated.safetensors"
_CONSOLIDATED_INDEX_FILE = "consolidated.safetensors.index.json"
_PARAMS_JSON = "params.json"


class MistralFormatConfig(PreTrainedConfig):
    """PreTrainedConfig subclass with automatic Mistral native format detection.

    Overrides `get_config_dict` to fall back to `params.json` when `config.json`
    is absent.
    """

    _INTERNAL_KEYS = ("transformers_weights", "_loaded_from_mistral_format", "quantization_config")

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | Any, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load config, falling back to `params.json` when `config.json` is absent.

        Args:
            pretrained_model_name_or_path (`str | Any`): Path or Hub identifier.
            **kwargs: Passed through to `PreTrainedConfig.get_config_dict`.
                Additional kwarg `mistral_format` (`bool | None`):
                - `True`: force loading from `params.json`.
                - `False`: force loading from `config.json`.
                - `None` (default): auto-detect.

        Returns:
            Tuple of (config_dict, remaining_kwargs).
        """
        mistral_format = kwargs.pop("mistral_format", None)

        if mistral_format is not True:
            try:
                config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
                if not config_dict:
                    raise OSError(
                        f"Can't find 'config.json' at '{pretrained_model_name_or_path}'. "
                        f"Set `mistral_format=True` to load from 'params.json' instead."
                    )
                return config_dict, kwargs
            except OSError as e:
                if mistral_format is False:
                    raise
                # Only fall through to params.json for missing-file errors.
                # Re-raise real errors (network, permission, etc.) even during auto-detect.
                error_msg = str(e).lower()
                is_missing_file = (
                    "does not appear to have" in error_msg
                    or "is not a local folder" in error_msg
                    or "can't find" in error_msg
                )
                if not is_missing_file:
                    raise

        return cls._get_config_dict_from_params_json(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def _get_config_dict_from_params_json(
        cls, pretrained_model_name_or_path: str | Any, model_type: str | None = None, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Load config from a native `params.json` file.

        Resolves `params.json` via `cached_file`, converts the native config
        to an HF config dict, and detects weight file format.

        Args:
            pretrained_model_name_or_path: Path or Hub identifier.
            model_type: Explicit model type override. When ``None``, falls back
                to ``cls.model_type`` if available, otherwise auto-detects from
                the ``params.json`` contents via structural heuristics.
            **kwargs: Passed through (hub download options are popped internally).
        """
        # Lazy import: params_conversion transitively imports model configs (e.g. Ministral3Config)
        # which in turn import MistralFormatConfig from this module, creating a circular dependency.
        from .params_conversion import _parse_native_config_from_dict, native_config_for_model_type

        # Pop the same kwargs that PreTrainedConfig._get_config_dict pops,
        # so they don't leak through to model_kwargs in from_pretrained.
        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        proxies = kwargs.pop("proxies", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)
        revision = kwargs.pop("revision", None)
        subfolder = kwargs.pop("subfolder", "")
        commit_hash = kwargs.pop("_commit_hash", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("_from_pipeline", None)
        kwargs.pop("_from_auto", None)

        resolved_params_file = cached_file(
            path_or_repo_id=pretrained_model_name_or_path,
            filename=_PARAMS_JSON,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            token=token,
            revision=revision,
            subfolder=subfolder,
            local_files_only=local_files_only,
            _commit_hash=commit_hash,
            _raise_exceptions_for_missing_entries=False,
        )
        if resolved_params_file is None:
            raise OSError(
                f"Can't find '{_PARAMS_JSON}' at '{pretrained_model_name_or_path}'. "
                f"Make sure the directory contains either a 'config.json' or a '{_PARAMS_JSON}' file."
            )

        with open(resolved_params_file, encoding="utf-8") as f:
            params_dict = json.load(f)

        logger.info("Loading Mistral native config from %s", resolved_params_file)

        # Resolve model_type: explicit arg > cls.model_type > auto-detect
        # Guard against empty string from PretrainedConfig base class.
        resolved_model_type = model_type or getattr(cls, "model_type", None) or None
        if resolved_model_type is not None:
            native_config = native_config_for_model_type(resolved_model_type, params_dict)
        else:
            native_config = _parse_native_config_from_dict(params_dict)
        hf_config = native_config.to_hf_config()
        config_dict = hf_config.to_dict()

        config_dict["_loaded_from_mistral_format"] = True

        weight_file = cls._detect_weight_file(
            pretrained_model_name_or_path,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
            _commit_hash=commit_hash,
        )
        if weight_file is not None:
            config_dict["transformers_weights"] = weight_file

        cls._propagate_internal_keys(config_dict)

        return config_dict, kwargs

    @classmethod
    def _propagate_internal_keys(cls, config_dict: dict[str, Any]) -> None:
        """Propagate internal metadata into nested sub-config dicts.

        Composite configs (e.g. Mistral3) have nested dicts like `text_config`
        and `vision_config`.
        """
        top_level = {k: config_dict[k] for k in cls._INTERNAL_KEYS if k in config_dict}
        if not top_level:
            return
        for value in config_dict.values():
            if isinstance(value, dict) and "model_type" in value:
                for key, val in top_level.items():
                    value.setdefault(key, val)

    @staticmethod
    def _make_file_exists_fn(pretrained_model_name_or_path: Any, is_local: bool, **kwargs):
        """Return a callable that checks whether *filename* exists in the model location.

        For local directories this uses ``os.path.isfile``; for Hub repos it
        uses a lightweight HEAD request via ``huggingface_hub.file_exists``.
        """
        if is_local:
            subfolder = kwargs.get("subfolder", "")
            base = (
                os.path.join(str(pretrained_model_name_or_path), subfolder)
                if subfolder
                else str(pretrained_model_name_or_path)
            )

            def _local_exists(filename):
                return os.path.isfile(os.path.join(base, filename))

            return _local_exists
        else:
            hub_kwargs = {}
            for key in ("revision", "token"):
                if key in kwargs and kwargs[key] is not None:
                    hub_kwargs[key] = kwargs[key]
            repo_id = str(pretrained_model_name_or_path)

            def _remote_exists(filename):
                try:
                    return hf_hub_file_exists(repo_id, filename, **hub_kwargs)
                except Exception:
                    return False

            return _remote_exists

    @classmethod
    def _detect_weight_file(cls, pretrained_model_name_or_path: Any, **kwargs) -> str | None:
        """Detect the weight file format, preferring Mistral native consolidated files.

        This method is only called from the Mistral format config loading path
        (``_get_config_dict_from_params_json``), so it prioritises consolidated
        weight files over HF-format ones.  Uses lightweight existence checks
        (local ``os.path.isfile`` or Hub HEAD request via
        ``huggingface_hub.file_exists``) to avoid downloading large weight files
        just to detect the format.

        Returns:
            Weight filename if consolidated format detected, ``None`` otherwise.
        """
        is_local = os.path.isdir(str(pretrained_model_name_or_path))
        file_exists_fn = cls._make_file_exists_fn(pretrained_model_name_or_path, is_local, **kwargs)

        for native_filename in (_CONSOLIDATED_INDEX_FILE, _CONSOLIDATED_SINGLE_FILE):
            if file_exists_fn(native_filename):
                return native_filename

        for hf_filename in (SAFETENSORS_SINGLE_FILE, SAFETENSORS_INDEX_FILE):
            if file_exists_fn(hf_filename):
                return None

        return None

    def _config_to_params_json(self) -> dict:
        """Convert this HF config back to a native `params.json` dict.

        Uses the reverse dispatcher to create a native config dataclass,
        then serializes it via `dataclasses.asdict`.  When the HF
        quantization config was successfully reverse-mapped to native
        ``quantization`` fields, the HF-only ``quantization_config`` key
        is stripped.  Otherwise it is preserved so non-reversible quant
        methods (GPTQ, AWQ, …) are not silently dropped.
        """
        # Lazy import: params_conversion transitively imports model configs (e.g. Ministral3Config)
        # which in turn import MistralFormatConfig from this module, creating a circular dependency.
        from .params_conversion import native_config_from_hf_config

        native = native_config_from_hf_config(self.model_type, self)
        result = dataclasses.asdict(native)
        # Only strip quantization_config when the native `quantization` field
        # was populated (FP8 case).
        if result.get("quantization") is not None:
            result.pop("quantization_config", None)
        return result
