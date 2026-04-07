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
r"""Config format detection and loading for Mistral native `params.json`.

Provides `MistralFormatConfig`, a `PreTrainedConfig` subclass that overrides
`get_config_dict()` to auto-detect and load from `params.json` when `config.json`
is absent. Mistral-family config classes inherit from this to gain automatic
native format support.
"""

import dataclasses
import json
from typing import Any

from ...configuration_utils import PreTrainedConfig
from ...utils import cached_file, logging


logger = logging.get_logger(__name__)

# Weight file names
_CONSOLIDATED_SINGLE = "consolidated.safetensors"
_CONSOLIDATED_INDEX = "consolidated.safetensors.index.json"
_HF_SINGLE = "model.safetensors"
_HF_INDEX = "model.safetensors.index.json"
_PARAMS_JSON = "params.json"


class MistralFormatConfig(PreTrainedConfig):
    r"""PreTrainedConfig subclass with automatic Mistral native format detection.

    Overrides `get_config_dict` to fall back to `params.json` when `config.json`
    is absent. Subclasses (e.g., `MistralConfig`, `Mistral4Config`) inherit this
    behavior transparently.
    """

    @classmethod
    def get_config_dict(
        cls, pretrained_model_name_or_path: str | Any, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Load config, falling back to `params.json` when `config.json` is absent.

        Args:
            pretrained_model_name_or_path: Path or Hub identifier.
            **kwargs: Passed through to `PreTrainedConfig.get_config_dict`.
                Additional kwarg `mistral_format` (bool | None):
                - `True`: force loading from `params.json`.
                - `False`: force loading from `config.json`.
                - `None` (default): auto-detect.

        Returns:
            Tuple of (config_dict, remaining_kwargs).
        """
        mistral_format = kwargs.pop("mistral_format", None)

        if not mistral_format:  # None or False
            try:
                config_dict, kwargs = super().get_config_dict(pretrained_model_name_or_path, **kwargs)
                if not config_dict:
                    raise OSError(
                        f"Can't find 'config.json' at '{pretrained_model_name_or_path}'. "
                        f"Set `mistral_format=True` to load from 'params.json' instead."
                    )
                return config_dict, kwargs
            except OSError as e:
                if mistral_format is not None:
                    raise e

        return cls._get_config_dict_from_params_json(pretrained_model_name_or_path, **kwargs)

    @classmethod
    def _get_config_dict_from_params_json(
        cls, pretrained_model_name_or_path: str | Any, **kwargs
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        r"""Load config from a native `params.json` file.

        Resolves `params.json` via `cached_file`, converts the native config
        to an HF config dict, and detects weight file format.
        """
        from .params_conversion import native_config_for_model_type  # lazy: avoid circular import

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
            pretrained_model_name_or_path,
            _PARAMS_JSON,
            cache_dir=cache_dir,
            force_download=force_download,
            proxies=proxies,
            local_files_only=local_files_only,
            token=token,
            revision=revision,
            subfolder=subfolder,
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

        logger.info(f"Loading Mistral native config from {resolved_params_file}")

        native_config = native_config_for_model_type(cls.model_type, params_dict)
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

        return config_dict, kwargs

    @classmethod
    def _detect_weight_file(cls, pretrained_model_name_or_path: str | Any, **kwargs) -> str | None:
        r"""Detect the weight file format.

        Probes for HF format files first (`model.safetensors`, `model.safetensors.index.json`).
        If found, returns `None` (HF format, no override needed).
        Otherwise probes for consolidated format files and returns the filename.

        Returns:
            Weight filename if consolidated format detected, `None` otherwise.
        """
        for hf_filename in (_HF_SINGLE, _HF_INDEX):
            resolved = cached_file(
                pretrained_model_name_or_path,
                hf_filename,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **kwargs,
            )
            if resolved is not None:
                return None

        for native_filename in (_CONSOLIDATED_INDEX, _CONSOLIDATED_SINGLE):
            resolved = cached_file(
                pretrained_model_name_or_path,
                native_filename,
                _raise_exceptions_for_missing_entries=False,
                _raise_exceptions_for_connection_errors=False,
                **kwargs,
            )
            if resolved is not None:
                return native_filename

        return None

    def _config_to_params_json(self) -> dict:
        r"""Convert this HF config back to a native `params.json` dict.

        Uses the reverse dispatcher to create a native config dataclass,
        then serializes it via `dataclasses.asdict`.
        """
        from .params_conversion import native_config_from_hf_config  # lazy: avoid circular import

        native = native_config_from_hf_config(self.model_type, self)
        return dataclasses.asdict(native)
