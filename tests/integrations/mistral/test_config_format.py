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

import copy
import json
import os
import tempfile
import unittest
from pathlib import Path

from huggingface_hub.constants import SAFETENSORS_INDEX_FILE, SAFETENSORS_SINGLE_FILE

from transformers import AutoConfig, GPTQConfig, Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.config_format import (
    _CONSOLIDATED_INDEX_FILE,
    _CONSOLIDATED_SINGLE_FILE,
    MistralFormatConfig,
)
from transformers.integrations.mistral.params_conversion import native_config_for_model_type

from .mistral_fixture_data import MINISTRAL3_PARAMS, MISTRAL3_PARAMS, MISTRAL4_PARAMS, MISTRAL_PARAMS, mistral_params


def _make_tmpdir_with_files(tmpdir: str, filenames: set[str]) -> None:
    """Create empty files in a temporary directory for weight detection tests."""
    for name in filenames:
        Path(os.path.join(tmpdir, name)).touch()


def _write_params_json(tmpdir: Path, params: dict) -> None:
    with open(tmpdir / "params.json", "w") as f:
        json.dump(params, f)


_PARAMS_MAP = {
    "mistral_params": MISTRAL_PARAMS,
    "ministral3_params": MINISTRAL3_PARAMS,
    "mistral4_params": MISTRAL4_PARAMS,
    "mistral3_params": MISTRAL3_PARAMS,
}


class TestMistralFormat(unittest.TestCase):
    def test_detect_weight_file(self) -> None:
        test_cases = [
            ("hf_single", {SAFETENSORS_SINGLE_FILE}, None),
            ("hf_index", {SAFETENSORS_INDEX_FILE}, None),
            ("consolidated_single", {_CONSOLIDATED_SINGLE_FILE}, _CONSOLIDATED_SINGLE_FILE),
            ("consolidated_index", {_CONSOLIDATED_INDEX_FILE}, _CONSOLIDATED_INDEX_FILE),
            ("nothing", set(), None),
            # When both formats exist, consolidated is preferred (this method is only
            # called from the Mistral-format config path).
            (
                "both_hf_and_consolidated",
                {SAFETENSORS_SINGLE_FILE, _CONSOLIDATED_SINGLE_FILE},
                _CONSOLIDATED_SINGLE_FILE,
            ),
            (
                "hf_index_and_consolidated_single",
                {SAFETENSORS_INDEX_FILE, _CONSOLIDATED_SINGLE_FILE},
                _CONSOLIDATED_SINGLE_FILE,
            ),
        ]
        for test_id, existing_files, expected in test_cases:
            with self.subTest(test_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _make_tmpdir_with_files(tmp_dir, existing_files)
                    self.assertEqual(MistralFormatConfig._detect_weight_file(tmp_dir), expected)

    def test_loaded_from_mistral_format(self) -> None:
        test_cases = [
            ("prefers_config_json", True, None, False),
            ("falls_back_to_params_json", False, None, True),
            ("mistral_format_true_forces_params", True, True, True),
        ]
        for test_id, save_config_json, mistral_format, expect_mistral_format in test_cases:
            with self.subTest(test_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    params = mistral_params()
                    if save_config_json:
                        MistralConfig().save_pretrained(tmp_path)
                    _write_params_json(tmp_path, params)
                    kwargs = {} if mistral_format is None else {"mistral_format": mistral_format}
                    config_dict, _ = MistralConfig.get_config_dict(tmp_path, **kwargs)
                    self.assertIsInstance(config_dict, dict)
                    if expect_mistral_format:
                        self.assertIn("_loaded_from_mistral_format", config_dict)
                    else:
                        self.assertNotIn("_loaded_from_mistral_format", config_dict)

    def test_get_config_dict_mistral_format_false_errors(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            params = mistral_params()
            _write_params_json(tmp_path, params)
            with self.assertRaises(OSError):
                MistralConfig.get_config_dict(tmp_path, mistral_format=False)

    def test_get_config_dict_sets_transformers_weights(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            params = mistral_params()
            _write_params_json(tmp_path, params)
            (tmp_path / "consolidated.safetensors").write_bytes(b"\x00")
            config_dict, _ = MistralConfig.get_config_dict(tmp_path)
            self.assertEqual(config_dict["transformers_weights"], _CONSOLIDATED_SINGLE_FILE)

    def test_config_to_params_json(self) -> None:
        test_cases = [
            ("mistral", MistralConfig, "mistral_params", set()),
            ("ministral3", Ministral3Config, "ministral3_params", {"quantization"}),
            ("mistral4", Mistral4Config, "mistral4_params", set()),
            ("mistral3", Mistral3Config, "mistral3_params", set()),
        ]
        for test_id, config_cls, params_name, skip_keys in test_cases:
            with self.subTest(test_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    params = copy.deepcopy(_PARAMS_MAP[params_name])
                    _write_params_json(tmp_path, params)
                    config = config_cls.from_pretrained(tmp_path)
                    result = config._config_to_params_json()
                    for key, value in params.items():
                        if key not in skip_keys:
                            if isinstance(value, dict):
                                # Roundtrip may add default fields to nested dicts (e.g. MOEModelArgs),
                                # so check that the original keys are a subset of the result.
                                for sub_key, sub_value in value.items():
                                    self.assertEqual(
                                        result[key][sub_key], sub_value, f"Mismatch on key {key!r}.{sub_key!r}"
                                    )
                            else:
                                self.assertEqual(result[key], value, f"Mismatch on key {key!r}")

    def test_config_to_params_json_preserves_non_fp8_quantization(self) -> None:
        """Non-FP8 quantization_config should be preserved in params.json output."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            params = copy.deepcopy(MISTRAL_PARAMS)
            _write_params_json(tmp_path, params)
            config = MistralConfig.from_pretrained(tmp_path)
            # Simulate a GPTQ-quantized model (no native Mistral representation)
            config.quantization_config = GPTQConfig(bits=4)
            result = config._config_to_params_json()
            self.assertIn("quantization_config", result)
            self.assertIsNotNone(result["quantization_config"])

    def test_config_to_params_json_strips_fp8_quantization(self) -> None:
        """FP8 quantization_config should be stripped when native quantization is populated."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            params = copy.deepcopy(MINISTRAL3_PARAMS)
            _write_params_json(tmp_path, params)
            config = Ministral3Config.from_pretrained(tmp_path)
            result = config._config_to_params_json()
            # quantization_config should be stripped (native quantization is populated)
            self.assertNotIn("quantization_config", result)
            # But native quantization should be present
            self.assertIn("quantization", result)
            self.assertIsNotNone(result["quantization"])

    def test_from_pretrained(self) -> None:
        test_cases = [
            ("mistral", MistralConfig, "mistral_params"),
            ("ministral3", Ministral3Config, "ministral3_params"),
            ("mistral4", Mistral4Config, "mistral4_params"),
            ("mistral3", Mistral3Config, "mistral3_params"),
        ]
        for test_id, config_cls, params_name in test_cases:
            with self.subTest(test_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    params = copy.deepcopy(_PARAMS_MAP[params_name])
                    _write_params_json(tmp_path, params)
                    loaded = config_cls.from_pretrained(tmp_path)

                    config_dict = native_config_for_model_type(config_cls.model_type, params).to_hf_config().to_dict()
                    config_dict["_loaded_from_mistral_format"] = True
                    expected = config_cls.from_dict(config_dict)
                    self.assertEqual(loaded, expected)

    def test_autoconfig_params_json_fallback(self) -> None:
        """AutoConfig.from_pretrained falls back to params.json when config.json is absent."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _write_params_json(tmp_path, copy.deepcopy(MISTRAL_PARAMS))
            config = AutoConfig.from_pretrained(tmp_path)
            self.assertIsInstance(config, MistralConfig)
            self.assertTrue(config._loaded_from_mistral_format)
            self.assertEqual(config.hidden_size, 4096)
            self.assertEqual(config.num_attention_heads, 32)

    def test_autoconfig_no_config_no_params_raises(self) -> None:
        """AutoConfig.from_pretrained raises when neither config.json nor params.json exist."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            with self.assertRaises((OSError, ValueError)):
                AutoConfig.from_pretrained(tmp_dir)
