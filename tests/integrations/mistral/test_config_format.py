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
r"""Tests for Phase 4: config format detection and loading."""

import json
import os
from pathlib import Path
from unittest.mock import patch

import pytest

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.config_format import (
    _CONSOLIDATED_INDEX_FILE,
    _CONSOLIDATED_SINGLE_FILE,
    _HF_INDEX,
    _HF_SINGLE,
    MistralFormatConfig,
)
from transformers.integrations.mistral.params_conversion import native_config_for_model_type


def _make_cached_file_side_effect(existing_files: set[str]):
    def _side_effect(path, filename, **kwargs):
        if filename in existing_files:
            return os.path.join(str(path), filename)
        return None

    return _side_effect


def _write_params_json(tmpdir: Path, params: dict) -> None:
    with open(tmpdir / "params.json", "w") as f:
        json.dump(params, f)


class TestMistralFormat:
    def _detect(self, existing_files: set[str]) -> str | None:
        side_effect = _make_cached_file_side_effect(existing_files)
        with patch("transformers.integrations.mistral.config_format.cached_file", side_effect=side_effect):
            return MistralFormatConfig._detect_weight_file("/fake/path")

    @pytest.mark.parametrize(
        "existing_files, expected",
        [
            ({_HF_SINGLE}, None),
            ({_HF_INDEX}, None),
            ({_CONSOLIDATED_SINGLE_FILE}, _CONSOLIDATED_SINGLE_FILE),
            ({_CONSOLIDATED_INDEX_FILE}, _CONSOLIDATED_INDEX_FILE),
            (set(), None),
            ({_HF_SINGLE, _CONSOLIDATED_SINGLE_FILE}, None),
            ({_HF_INDEX, _CONSOLIDATED_SINGLE_FILE}, None),
        ],
        ids=[
            "hf_single",
            "hf_index",
            "consolidated_single",
            "consolidated_index",
            "nothing",
            "both_hf_and_consolidated",
            "hf_index_and_consolidated_single",
        ],
    )
    def test_detect_weight_file(self, existing_files, expected):
        assert self._detect(existing_files) == expected

    def test_get_config_dict_prefers_config_json(self, mistral_params, tmp_path):
        MistralConfig().save_pretrained(tmp_path)
        _write_params_json(tmp_path, mistral_params)
        config_dict, _ = MistralConfig.get_config_dict(tmp_path)
        assert isinstance(config_dict, dict)
        assert "_loaded_from_mistral_format" not in config_dict

    def test_get_config_dict_falls_back_to_params_json(self, mistral_params, tmp_path):
        _write_params_json(tmp_path, mistral_params)
        config_dict, _ = MistralConfig.get_config_dict(tmp_path)
        assert isinstance(config_dict, dict)
        assert config_dict["_loaded_from_mistral_format"]

    def test_get_config_dict_mistral_format_true_forces_params(self, mistral_params, tmp_path):
        MistralConfig().save_pretrained(tmp_path)
        _write_params_json(tmp_path, mistral_params)
        config_dict, _ = MistralConfig.get_config_dict(tmp_path, mistral_format=True)
        assert config_dict["_loaded_from_mistral_format"]

    def test_get_config_dict_mistral_format_false_errors(self, mistral_params, tmp_path):
        _write_params_json(tmp_path, mistral_params)
        with pytest.raises(OSError):
            MistralConfig.get_config_dict(tmp_path, mistral_format=False)

    def test_get_config_dict_sets_transformers_weights(self, mistral_params, tmp_path):
        _write_params_json(tmp_path, mistral_params)
        (tmp_path / "consolidated.safetensors").write_bytes(b"\x00")
        config_dict, _ = MistralConfig.get_config_dict(tmp_path)
        assert config_dict["transformers_weights"] == _CONSOLIDATED_SINGLE_FILE

    @pytest.mark.parametrize(
        "config_cls, params_fixture, skip_keys",
        [
            (MistralConfig, "mistral_params", set()),
            (Ministral3Config, "ministral3_params", {"quantization"}),
            (Mistral4Config, "mistral4_params", set()),
        ],
        ids=["mistral", "ministral3", "mistral4"],
    )
    def test_config_to_params_json(self, config_cls, params_fixture, skip_keys, request, tmp_path):
        params = request.getfixturevalue(params_fixture)
        _write_params_json(tmp_path, params)
        config = config_cls.from_pretrained(tmp_path)
        result = config._config_to_params_json()
        for key, value in params.items():
            if key not in skip_keys:
                if isinstance(value, dict):
                    # Roundtrip may add default fields to nested dicts (e.g. MOEModelArgs),
                    # so check that the original keys are a subset of the result.
                    for sub_key, sub_value in value.items():
                        assert result[key][sub_key] == sub_value, f"Mismatch on key {key!r}.{sub_key!r}"
                else:
                    assert result[key] == value, f"Mismatch on key {key!r}"

    @pytest.mark.parametrize(
        "config_cls, params_fixture",
        [
            (MistralConfig, "mistral_params"),
            (Ministral3Config, "ministral3_params"),
            (Mistral4Config, "mistral4_params"),
            (Mistral3Config, "mistral3_params"),
        ],
        ids=["mistral", "ministral3", "mistral4", "mistral3"],
    )
    def test_from_pretrained(self, config_cls, params_fixture, request, tmp_path):
        params = request.getfixturevalue(params_fixture)
        _write_params_json(tmp_path, params)
        loaded = config_cls.from_pretrained(tmp_path)

        config_dict = native_config_for_model_type(config_cls.model_type, params).to_hf_config().to_dict()
        config_dict["_loaded_from_mistral_format"] = True
        expected = config_cls.from_dict(config_dict)
        assert loaded == expected
