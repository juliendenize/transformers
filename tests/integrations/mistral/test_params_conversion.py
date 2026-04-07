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

import pytest

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


class TestFP8NativeConfig:
    def test_valid_construction(self):
        config = FP8NativeConfig("fp8_e4m3", "TENSOR")
        assert config == FP8NativeConfig(qformat_weight="fp8_e4m3", qscheme_act="TENSOR")

    def test_unsupported_format_raises(self):
        with pytest.raises(ValueError, match="fp8_e4m3"):
            FP8NativeConfig("int8", "TENSOR")

    def test_dynamic_scheme(self):
        config = FP8NativeConfig("fp8_e4m3", "DYNAMIC")
        assert config == FP8NativeConfig(qformat_weight="fp8_e4m3", qscheme_act="DYNAMIC")


class TestMoeNativeConfig:
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
        assert config == expected


class TestMistralNativeConfig:
    def test_from_params_json(self, mistral_params):
        config = MistralNativeConfig.from_params_json(mistral_params)
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
        assert config == expected

    def test_from_params_json_minimal_required(self, mistral_base_fields):
        config = MistralNativeConfig.from_params_json(mistral_base_fields)
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            hidden_dim=14336,
            n_heads=32,
            norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
        )
        assert config == expected

    def test_from_params_json_missing_required_raises(self, mistral_base_fields):
        params = {k: v for k, v in mistral_base_fields.items() if k != "dim"}
        with pytest.raises(KeyError):
            MistralNativeConfig.from_params_json(params)

    def test_from_params_json_ignores_unknown_keys(self, mistral_base_fields):
        params = {**mistral_base_fields, "unknown_field": 999}
        config = MistralNativeConfig.from_params_json(params)
        assert not hasattr(config, "unknown_field")

    def test_to_hf_config(self, mistral_params):
        hf = MistralNativeConfig.from_params_json(mistral_params).to_hf_config()
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
        assert hf == expected

    def test_roundtrip(self, mistral_params):
        native = MistralNativeConfig.from_params_json(mistral_params)
        restored = MistralNativeConfig.from_hf_config(native.to_hf_config())
        assert restored == native


class TestMinistral3NativeConfig:
    def test_from_params_json(self, ministral3_params):
        config = Ministral3NativeConfig.from_params_json(ministral3_params)
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
        assert config == expected

    def test_from_params_json_no_quantization(self, ministral3_params):
        params = {k: v for k, v in ministral3_params.items() if k != "quantization"}
        config = Ministral3NativeConfig.from_params_json(params)
        assert config.quantization is None

    def test_to_hf_config_with_yarn(self, ministral3_params):
        hf = Ministral3NativeConfig.from_params_json(ministral3_params).to_hf_config()
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
        assert hf == expected

    def test_to_hf_config_with_fp8(self, ministral3_params):
        params_no_yarn = {k: v for k, v in ministral3_params.items() if k != "yarn"}
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
        assert hf == expected

    def test_roundtrip(self, ministral3_params):
        params_no_quant = {k: v for k, v in ministral3_params.items() if k != "quantization"}
        native = Ministral3NativeConfig.from_params_json(params_no_quant)
        restored = Ministral3NativeConfig.from_hf_config(native.to_hf_config())
        assert restored == native


class TestMistral4NativeConfig:
    def test_from_params_json(self, mistral4_params):
        config = Mistral4NativeConfig.from_params_json(mistral4_params)
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
        assert config == expected

    def test_to_hf_config(self, mistral4_params):
        hf = Mistral4NativeConfig.from_params_json(mistral4_params).to_hf_config()
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
        assert hf == expected

    def test_roundtrip(self, mistral4_params):
        native = Mistral4NativeConfig.from_params_json(mistral4_params)
        restored = Mistral4NativeConfig.from_hf_config(native.to_hf_config())
        assert restored == native


class TestMistral3NativeConfig:
    def test_from_params_json(self, mistral3_params):
        config = Mistral3NativeConfig.from_params_json(mistral3_params)
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
        assert config == expected

    def test_to_hf_config(self, mistral3_params):
        hf = Mistral3NativeConfig.from_params_json(mistral3_params).to_hf_config()
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
        assert hf == expected

    def test_roundtrip(self, mistral3_params):
        native = Mistral3NativeConfig.from_params_json(mistral3_params)
        restored = Mistral3NativeConfig.from_hf_config(native.to_hf_config())
        assert restored == native

    def test_backbone_auto_detection_base(self, mistral3_params):
        config = Mistral3NativeConfig.from_params_json(mistral3_params)
        assert isinstance(config.text_config, MistralNativeConfig)
        assert not isinstance(config.text_config, Ministral3NativeConfig)
        assert not isinstance(config.text_config, Mistral4NativeConfig)

    def test_backbone_auto_detection_yarn(self, mistral3_params):
        params = {
            **mistral3_params,
            "yarn": {"factor": 16.0, "original_max_position_embeddings": 16384, "beta": 32.0, "alpha": 1.0},
        }
        config = Mistral3NativeConfig.from_params_json(params)
        assert isinstance(config.text_config, Ministral3NativeConfig)

    def test_backbone_auto_detection_moe(self, mistral3_params):
        params = {
            **mistral3_params,
            "q_lora_rank": 1024,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 64,
            "kv_lora_rank": 256,
            "v_head_dim": 128,
            "moe": {"num_experts": 128, "num_experts_per_tok": 4, "expert_hidden_dim": 2048},
            "yarn": {"factor": 128.0, "original_max_position_embeddings": 8192, "beta": 32.0, "alpha": 1.0},
        }
        config = Mistral3NativeConfig.from_params_json(params)
        assert isinstance(config.text_config, Mistral4NativeConfig)


class TestDispatchers:
    @pytest.mark.parametrize(
        "model_type, params_fixture, expected_cls",
        [
            ("mistral", "mistral_params", MistralNativeConfig),
            ("ministral3", "ministral3_params", Ministral3NativeConfig),
            ("mistral4", "mistral4_params", Mistral4NativeConfig),
            ("mistral3", "mistral3_params", Mistral3NativeConfig),
        ],
        ids=["mistral", "ministral3", "mistral4", "mistral3"],
    )
    def test_native_config_for_model_type_dispatches_correctly(
        self, model_type, params_fixture, expected_cls, request
    ):
        params = request.getfixturevalue(params_fixture)
        result = native_config_for_model_type(model_type, params)
        assert isinstance(result, expected_cls)

    def test_native_config_for_model_type_unknown_raises(self):
        with pytest.raises(ValueError):
            native_config_for_model_type("unknown", {})

    def test_native_config_from_hf_config_unknown_raises(self):
        with pytest.raises(ValueError):
            native_config_from_hf_config("unknown", object())
