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
r"""Tests for Phase 1: native config dataclasses (pure Python, no torch)."""

import unittest

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.params_conversion import (
    FP8NativeConfig,
    Ministral3NativeConfig,
    Mistral3NativeConfig,
    Mistral4NativeConfig,
    MistralNativeConfig,
    MoeNativeConfig,
    VisionEncoderNativeConfig,
    YarnNativeConfig,
    native_config_for_model_type,
    native_config_from_hf_config,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig


# ---------------------------------------------------------------------------
# Fixtures – params.json-style dicts for each model variant
# ---------------------------------------------------------------------------

_MISTRAL_BASE_FIELDS = {
    "dim": 4096,
    "n_layers": 32,
    "hidden_dim": 14336,
    "n_heads": 32,
    "norm_eps": 1e-5,
    "head_dim": 128,
    "vocab_size": 32000,
}

_MISTRAL_PARAMS = {
    **_MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "max_position_embeddings": 32768,
}

_MINISTRAL3_PARAMS = {
    **_MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 262144,
    "tied_embeddings": True,
    "yarn": {
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "beta": 32.0,
        "alpha": 1.0,
    },
    "quantization": {
        "qformat_weight": "fp8_e4m3",
        "qscheme_act": "TENSOR",
    },
}

_MISTRAL4_PARAMS = {
    **_MISTRAL_BASE_FIELDS,
    "n_kv_heads": 32,
    "rope_theta": 10000.0,
    "max_position_embeddings": 1048576,
    "q_lora_rank": 1024,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 64,
    "kv_lora_rank": 256,
    "v_head_dim": 128,
    "moe": {
        "num_experts": 128,
        "num_experts_per_tok": 4,
        "expert_hidden_dim": 2048,
        "first_k_dense_replace": 0,
        "num_shared_experts": 1,
        "routed_scale": 1.0,
        "num_expert_groups": 1,
        "num_expert_groups_per_tok": 1,
    },
    "yarn": {
        "factor": 128.0,
        "original_max_position_embeddings": 8192,
        "beta": 32.0,
        "alpha": 1.0,
    },
}

_MISTRAL3_PARAMS = {
    **_MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 1000000000.0,
    "max_position_embeddings": 131072,
    "vision_encoder": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 1540,
        "head_dim": 64,
        "intermediate_size": 4096,
        "adapter_bias": False,
        "spatial_merge_size": 2,
        "image_token_id": 10,
    },
}


class TestFP8NativeConfig(unittest.TestCase):
    r"""Tests for `FP8NativeConfig`."""

    def test_valid_construction(self):
        config = FP8NativeConfig("fp8_e4m3", "TENSOR")
        self.assertEqual(config, FP8NativeConfig(qformat_weight="fp8_e4m3", qscheme_act="TENSOR"))

    def test_unsupported_format_raises(self):
        with self.assertRaises(ValueError) as ctx:
            FP8NativeConfig("int8", "TENSOR")
        self.assertIn("fp8_e4m3", str(ctx.exception))

    def test_dynamic_scheme(self):
        config = FP8NativeConfig("fp8_e4m3", "DYNAMIC")
        self.assertEqual(config, FP8NativeConfig(qformat_weight="fp8_e4m3", qscheme_act="DYNAMIC"))


class TestMoeNativeConfig(unittest.TestCase):
    r"""Tests for `MoeNativeConfig`."""

    def test_defaults(self):
        config = MoeNativeConfig(num_experts=128, num_experts_per_tok=4, expert_hidden_dim=2048)
        expected = MoeNativeConfig(
            num_experts=128,
            num_experts_per_tok=4,
            expert_hidden_dim=2048,
            first_k_dense_replace=0,
            num_shared_experts=1,
            routed_scale=1.0,
            num_expert_groups=1,
            num_expert_groups_per_tok=1,
        )
        self.assertEqual(config, expected)


class TestMistralNativeConfig(unittest.TestCase):
    r"""Tests for `MistralNativeConfig`."""

    def test_from_params_json(self):
        config = MistralNativeConfig.from_params_json(_MISTRAL_PARAMS)
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            hidden_dim=14336,
            n_heads=32,
            norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            n_kv_heads=8,
            rope_theta=10000.0,
            sliding_window=4096,
            max_position_embeddings=32768,
        )
        self.assertEqual(config, expected)

    def test_from_params_json_minimal_required(self):
        config = MistralNativeConfig.from_params_json(_MISTRAL_BASE_FIELDS)
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            hidden_dim=14336,
            n_heads=32,
            norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
        )
        self.assertEqual(config, expected)

    def test_from_params_json_missing_required_raises(self):
        params = {k: v for k, v in _MISTRAL_BASE_FIELDS.items() if k != "dim"}
        with self.assertRaises(KeyError):
            MistralNativeConfig.from_params_json(params)

    def test_from_params_json_ignores_unknown_keys(self):
        params = {**_MISTRAL_BASE_FIELDS, "unknown_field": 999}
        config = MistralNativeConfig.from_params_json(params)
        self.assertFalse(hasattr(config, "unknown_field"))

    def test_to_hf_config(self):
        hf = MistralNativeConfig.from_params_json(_MISTRAL_PARAMS).to_hf_config()
        expected = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            rope_theta=10000.0,
            sliding_window=4096,
            max_position_embeddings=32768,
            tie_word_embeddings=False,
        )
        self.assertEqual(hf, expected)

    def test_roundtrip(self):
        native = MistralNativeConfig.from_params_json(_MISTRAL_PARAMS)
        restored = MistralNativeConfig.from_hf_config(native.to_hf_config())
        self.assertEqual(restored, native)


class TestMinistral3NativeConfig(unittest.TestCase):
    r"""Tests for `Ministral3NativeConfig`."""

    def test_from_params_json(self):
        config = Ministral3NativeConfig.from_params_json(_MINISTRAL3_PARAMS)
        expected = Ministral3NativeConfig(
            dim=4096,
            n_layers=32,
            hidden_dim=14336,
            n_heads=32,
            norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            n_kv_heads=8,
            rope_theta=1000000.0,
            max_position_embeddings=262144,
            tied_embeddings=True,
            yarn=YarnNativeConfig(factor=16.0, original_max_position_embeddings=16384, beta=32.0, alpha=1.0),
            quantization=FP8NativeConfig(qformat_weight="fp8_e4m3", qscheme_act="TENSOR"),
        )
        self.assertEqual(config, expected)

    def test_from_params_json_no_quantization(self):
        params = {k: v for k, v in _MINISTRAL3_PARAMS.items() if k != "quantization"}
        config = Ministral3NativeConfig.from_params_json(params)
        self.assertIsNone(config.quantization)

    def test_to_hf_config_with_yarn(self):
        hf = Ministral3NativeConfig.from_params_json(_MINISTRAL3_PARAMS).to_hf_config()
        expected = Ministral3Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 1000000.0,
                "factor": 16.0,
                "original_max_position_embeddings": 16384,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
            },
            quantization_config={"quant_method": "fp8", "activation_scheme": "static"},
        )
        self.assertEqual(hf, expected)

    def test_to_hf_config_with_fp8(self):
        params_no_yarn = {k: v for k, v in _MINISTRAL3_PARAMS.items() if k != "yarn"}
        hf = Ministral3NativeConfig.from_params_json(params_no_yarn).to_hf_config()
        expected = Ministral3Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=262144,
            tie_word_embeddings=True,
            quantization_config={"quant_method": "fp8", "activation_scheme": "static"},
        )
        self.assertEqual(hf, expected)

    def test_roundtrip(self):
        # FP8 quantization doesn't roundtrip via HF config, so exclude it
        params_no_quant = {k: v for k, v in _MINISTRAL3_PARAMS.items() if k != "quantization"}
        native = Ministral3NativeConfig.from_params_json(params_no_quant)
        restored = Ministral3NativeConfig.from_hf_config(native.to_hf_config())
        self.assertEqual(restored, native)


class TestMistral4NativeConfig(unittest.TestCase):
    r"""Tests for `Mistral4NativeConfig`."""

    def test_from_params_json(self):
        config = Mistral4NativeConfig.from_params_json(_MISTRAL4_PARAMS)
        expected = Mistral4NativeConfig(
            dim=4096,
            n_layers=32,
            hidden_dim=14336,
            n_heads=32,
            norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            n_kv_heads=32,
            rope_theta=10000.0,
            max_position_embeddings=1048576,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            kv_lora_rank=256,
            v_head_dim=128,
            moe=MoeNativeConfig(
                num_experts=128,
                num_experts_per_tok=4,
                expert_hidden_dim=2048,
                first_k_dense_replace=0,
                num_shared_experts=1,
                routed_scale=1.0,
                num_expert_groups=1,
                num_expert_groups_per_tok=1,
            ),
            yarn=YarnNativeConfig(factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0),
        )
        self.assertEqual(config, expected)

    def test_to_hf_config(self):
        hf = Mistral4NativeConfig.from_params_json(_MISTRAL4_PARAMS).to_hf_config()
        expected = Mistral4Config(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=32,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=1048576,
            q_lora_rank=1024,
            qk_rope_head_dim=64,
            qk_nope_head_dim=64,
            kv_lora_rank=256,
            v_head_dim=128,
            n_routed_experts=128,
            num_experts_per_tok=4,
            moe_intermediate_size=2048,
            first_k_dense_replace=0,
            n_shared_experts=1,
            routed_scaling_factor=1.0,
            n_group=1,
            topk_group=1,
            norm_topk_prob=True,
            rope_parameters={
                "type": "yarn",
                "rope_theta": 10000.0,
                "factor": 128.0,
                "original_max_position_embeddings": 8192,
                "beta_fast": 32.0,
                "beta_slow": 1.0,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
                "partial_rotary_factor": 0.5,
            },
        )
        self.assertEqual(hf, expected)

    def test_roundtrip(self):
        native = Mistral4NativeConfig.from_params_json(_MISTRAL4_PARAMS)
        restored = Mistral4NativeConfig.from_hf_config(native.to_hf_config())
        self.assertEqual(restored, native)


class TestMistral3NativeConfig(unittest.TestCase):
    r"""Tests for `Mistral3NativeConfig`."""

    def test_from_params_json(self):
        config = Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS)
        expected = Mistral3NativeConfig(
            text_config=MistralNativeConfig(
                dim=4096,
                n_layers=32,
                hidden_dim=14336,
                n_heads=32,
                norm_eps=1e-5,
                head_dim=128,
                vocab_size=32000,
                n_kv_heads=8,
                rope_theta=1000000000.0,
                max_position_embeddings=131072,
            ),
            vision_encoder=VisionEncoderNativeConfig(
                hidden_size=1024,
                num_hidden_layers=24,
                num_attention_heads=16,
                patch_size=14,
                image_size=1540,
                head_dim=64,
                intermediate_size=4096,
            ),
        )
        self.assertEqual(config, expected)

    def test_to_hf_config(self):
        hf = Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS).to_hf_config()
        text_config = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            rope_theta=1000000000.0,
            sliding_window=None,
            max_position_embeddings=131072,
        )
        vision_config = PixtralVisionConfig(
            hidden_size=1024,
            num_hidden_layers=24,
            num_attention_heads=16,
            patch_size=14,
            image_size=1540,
            intermediate_size=4096,
            hidden_act="silu",
        )
        expected = Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            multimodal_projector_bias=False,
            image_token_id=10,
            spatial_merge_size=2,
            vision_feature_layer=-1,
            tie_word_embeddings=False,
        )
        self.assertEqual(hf, expected)

    def test_roundtrip(self):
        native = Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS)
        restored = Mistral3NativeConfig.from_hf_config(native.to_hf_config())
        self.assertEqual(restored, native)

    def test_backbone_auto_detection_base(self):
        config = Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS)
        self.assertIsInstance(config.text_config, MistralNativeConfig)
        self.assertNotIsInstance(config.text_config, Ministral3NativeConfig)
        self.assertNotIsInstance(config.text_config, Mistral4NativeConfig)

    def test_backbone_auto_detection_yarn(self):
        params = {
            **_MISTRAL3_PARAMS,
            "yarn": {"factor": 16.0, "original_max_position_embeddings": 16384, "beta": 32.0, "alpha": 1.0},
        }
        config = Mistral3NativeConfig.from_params_json(params)
        self.assertIsInstance(config.text_config, Ministral3NativeConfig)

    def test_backbone_auto_detection_moe(self):
        params = {
            **_MISTRAL3_PARAMS,
            "q_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 64,
            "kv_lora_rank": 256,
            "v_head_dim": 128,
            "moe": {"num_experts": 128, "num_experts_per_tok": 4, "expert_hidden_dim": 2048},
            "yarn": {"factor": 128.0, "original_max_position_embeddings": 8192, "beta": 32.0, "alpha": 1.0},
        }
        config = Mistral3NativeConfig.from_params_json(params)
        self.assertIsInstance(config.text_config, Mistral4NativeConfig)


class TestDispatchers(unittest.TestCase):
    r"""Tests for `native_config_for_model_type` and `native_config_from_hf_config`."""

    def test_native_config_for_model_type_dispatches_correctly(self):
        cases = [
            ("mistral", _MISTRAL_PARAMS, MistralNativeConfig),
            ("ministral3", _MINISTRAL3_PARAMS, Ministral3NativeConfig),
            ("mistral4", _MISTRAL4_PARAMS, Mistral4NativeConfig),
            ("mistral3", _MISTRAL3_PARAMS, Mistral3NativeConfig),
        ]
        for model_type, params, expected_cls in cases:
            with self.subTest(model_type=model_type):
                result = native_config_for_model_type(model_type, params)
                self.assertIsInstance(result, expected_cls)

    def test_native_config_for_model_type_unknown_raises(self):
        with self.assertRaises(ValueError):
            native_config_for_model_type("unknown", {})

    def test_native_config_from_hf_config_unknown_raises(self):
        with self.assertRaises(ValueError):
            native_config_from_hf_config("unknown", object())


if __name__ == "__main__":
    unittest.main()
