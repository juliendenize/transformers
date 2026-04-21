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

import pytest

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.params_conversion import (
    Llama4Scaling,
    MistralNativeConfig,
    MOEModelArgs,
    QuantizationArgs,
    VisionEncoderArgs,
    YarnArgs,
    _get_maybe_quant_config,
    _hf_config_to_native_config,
    _native_config_to_hf_config,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import QuantizationConfigMixin


@pytest.fixture()
def base_native_config() -> MistralNativeConfig:
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=32768,
    )


@pytest.fixture()
def yarn_args() -> YarnArgs:
    return YarnArgs(factor=16.0, original_max_position_embeddings=16384, beta=32.0, alpha=1.0)


@pytest.fixture()
def llama4_scaling() -> Llama4Scaling:
    return Llama4Scaling(original_max_position_embeddings=16384, beta=0.1)


@pytest.fixture()
def vision_encoder_args() -> VisionEncoderArgs:
    return VisionEncoderArgs(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=14,
        image_size=1540,
        intermediate_size=4096,
        num_channels=3,
        max_image_size=1540,
        rope_theta=10000.0,
        mm_projector_id="patch_merge",
        add_pre_mm_projector_layer_norm=True,
        adapter_bias=False,
        spatial_merge_size=2,
        image_token_id=10,
        image_break_token_id=12,
        image_end_token_id=13,
    )


@pytest.fixture()
def moe_args() -> MOEModelArgs:
    return MOEModelArgs(
        num_experts=128,
        num_experts_per_tok=4,
        expert_hidden_dim=2048,
        first_k_dense_replace=0,
        num_shared_experts=1,
        routed_scale=1.0,
        num_expert_groups=1,
        num_expert_groups_per_tok=1,
    )


@pytest.fixture()
def ministral3_native_config(yarn_args: YarnArgs, llama4_scaling: Llama4Scaling) -> MistralNativeConfig:
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=1000000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=262144,
        tied_embeddings=True,
        yarn=yarn_args,
        llama_4_scaling=llama4_scaling,
    )


@pytest.fixture()
def mistral3_native_config(vision_encoder_args: VisionEncoderArgs) -> MistralNativeConfig:
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=1000000000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=131072,
        vision_encoder=vision_encoder_args,
    )


@pytest.fixture()
def mistral4_native_config(moe_args: MOEModelArgs) -> MistralNativeConfig:
    return MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=32,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=1048576,
        q_lora_rank=1024,
        qk_rope_head_dim=64,
        qk_nope_head_dim=64,
        kv_lora_rank=256,
        v_head_dim=128,
        yarn=YarnArgs(factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0),
        llama_4_scaling=Llama4Scaling(original_max_position_embeddings=8192, beta=0.1),
        moe=moe_args,
    )


@pytest.fixture()
def expected_mistral_hf_config() -> MistralConfig:
    return MistralConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=32768,
        sliding_window=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 10000.0,
        },
        quantization_config=None,
    )


@pytest.fixture()
def expected_ministral3_hf_config() -> Ministral3Config:
    return Ministral3Config(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=262144,
        sliding_window=None,
        tie_word_embeddings=True,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 1000000.0,
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
        quantization_config=None,
    )


@pytest.fixture()
def expected_mistral4_hf_config() -> Mistral4Config:
    return Mistral4Config(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=32,
        rms_norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=1048576,
        sliding_window=None,
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
        quant_config=None,
        rope_parameters={
            "rope_type": "yarn",
            "rope_theta": 10000.0,
            "factor": 128.0,
            "original_max_position_embeddings": 8192,
            "beta_fast": 32.0,
            "beta_slow": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "partial_rotary_factor": 0.5,
        },
    )


@pytest.fixture()
def expected_mistral3_hf_config() -> Mistral3Config:
    text_config = MistralConfig(
        hidden_size=4096,
        num_hidden_layers=32,
        intermediate_size=14336,
        num_attention_heads=32,
        num_key_value_heads=8,
        rms_norm_eps=1e-5,
        head_dim=128,
        vocab_size=32000,
        max_position_embeddings=131072,
        sliding_window=None,
        rope_parameters={
            "rope_type": "default",
            "rope_theta": 1000000000.0,
        },
        quantization_config=None,
    )
    vision_config = PixtralVisionConfig(
        hidden_size=1024,
        num_hidden_layers=24,
        num_attention_heads=16,
        patch_size=14,
        image_size=1540,
        intermediate_size=4096,
        num_channels=3,
        hidden_act="silu",
        rope_theta=10000.0,
    )
    return Mistral3Config(
        text_config=text_config,
        vision_config=vision_config,
        multimodal_projector_bias=False,
        image_token_id=10,
        spatial_merge_size=2,
        vision_feature_layer=-1,
        quantization_config=None,
        tie_word_embeddings=False,
    )


def _make_hf_fp8_quant_config(activation_scheme: str = "static") -> QuantizationConfigMixin:
    return AutoQuantizationConfig.from_dict(
        {
            "quant_method": "fp8",
            "activation_scheme": activation_scheme,
            "modules_to_not_convert": ["lm_head"],
            "weight_block_size": None,
        }
    )


class TestQuantizationArgs:
    def test_valid_tensor_scheme(self) -> None:
        config = QuantizationArgs("fp8_e4m3", "TENSOR")
        assert config == QuantizationArgs("fp8_e4m3", "TENSOR")

    @pytest.mark.parametrize("scheme", ["DYNAMIC", "UNSUPPORTED", "static"])
    def test_invalid_scheme_raises(self, scheme: str) -> None:
        with pytest.raises(ValueError, match=scheme):
            QuantizationArgs("fp8_e4m3", scheme)

    def test_unsupported_format_raises(self) -> None:
        with pytest.raises(ValueError, match="fp8_e4m3"):
            QuantizationArgs("int8", "TENSOR")


class TestGetMaybeQuantConfig:
    def test_none_returns_none(self) -> None:
        assert _get_maybe_quant_config(is_vision_model=False, quantization_args=None) is None

    def test_tensor_produces_static(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=False,
            quantization_args=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        qc_dict = qc.to_dict()
        assert qc_dict["quant_method"] == "fp8"
        assert qc_dict["activation_scheme"] == "static"

    def test_vision_model_adds_modules_to_skip(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=True,
            quantization_args=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        qc_dict = qc.to_dict()
        assert "model.vision_tower" in qc_dict["modules_to_not_convert"]
        assert "model.multi_modal_projector" in qc_dict["modules_to_not_convert"]
        assert "lm_head" in qc_dict["modules_to_not_convert"]


class TestMistralNativeConfig:
    def test_mutual_exclusivity_raises(self) -> None:
        with pytest.raises(ValueError, match="Cannot set both"):
            MistralNativeConfig(
                dim=4096,
                n_layers=32,
                head_dim=128,
                hidden_dim=14336,
                n_heads=32,
                n_kv_heads=8,
                rope_theta=10000.0,
                norm_eps=1e-5,
                vocab_size=32000,
                max_position_embeddings=32768,
                quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
                quantization_config=_make_hf_fp8_quant_config(),
            )

    @pytest.mark.parametrize(
        "quant_kwargs",
        [
            pytest.param({"quantization": QuantizationArgs("fp8_e4m3", "TENSOR")}, id="native_only"),
            pytest.param({"quantization_config": _make_hf_fp8_quant_config()}, id="hf_only"),
            pytest.param({}, id="neither"),
        ],
    )
    def test_single_or_no_quantization_ok(self, quant_kwargs: dict) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
            **quant_kwargs,
        )
        has_native = "quantization" in quant_kwargs
        has_hf = "quantization_config" in quant_kwargs
        assert (native.quantization is not None) == has_native
        assert (native.quantization_config is not None) == has_hf


def test_native_to_hf_mistral(
    base_native_config: MistralNativeConfig, expected_mistral_hf_config: MistralConfig
) -> None:
    hf = _native_config_to_hf_config(base_native_config)
    assert isinstance(hf, MistralConfig)
    assert hf == expected_mistral_hf_config


def test_native_to_hf_ministral3(
    ministral3_native_config: MistralNativeConfig, expected_ministral3_hf_config: Ministral3Config
) -> None:
    hf = _native_config_to_hf_config(ministral3_native_config)
    assert isinstance(hf, Ministral3Config)
    assert hf == expected_ministral3_hf_config


def test_native_to_hf_mistral4(
    mistral4_native_config: MistralNativeConfig, expected_mistral4_hf_config: Mistral4Config
) -> None:
    hf = _native_config_to_hf_config(mistral4_native_config)
    assert isinstance(hf, Mistral4Config)
    assert hf == expected_mistral4_hf_config


class TestNativeToHFMistral3:
    def test_vision_model(
        self, mistral3_native_config: MistralNativeConfig, expected_mistral3_hf_config: Mistral3Config
    ) -> None:
        hf = _native_config_to_hf_config(mistral3_native_config)
        assert isinstance(hf, Mistral3Config)
        assert hf == expected_mistral3_hf_config

    def test_tensor_fp8_propagates_to_mistral3(self, vision_encoder_args: VisionEncoderArgs) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=131072,
            quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
            vision_encoder=vision_encoder_args,
        )
        hf = _native_config_to_hf_config(native)
        qc = hf.quantization_config
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        assert qc["activation_scheme"] == "static"


class TestQuantizationConfigPassthrough:
    def test_quantization_config_passthrough_mistral(self) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
            quantization_config=_make_hf_fp8_quant_config("dynamic"),
        )
        hf = _native_config_to_hf_config(native)
        qc_dict = hf.quantization_config.to_dict()
        assert qc_dict["activation_scheme"] == "dynamic"
        assert qc_dict["quant_method"] == "fp8"

    def test_native_quantization_forward_still_works(self) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
            quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        hf = _native_config_to_hf_config(native)
        qc_dict = hf.quantization_config.to_dict()
        assert qc_dict["activation_scheme"] == "static"


class TestHFToNativeMistral:
    def test_basic_reverse(self, base_native_config: MistralNativeConfig) -> None:
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=None,
        )
        native = _hf_config_to_native_config(hf)
        assert native == base_native_config

    def test_roundtrip(self, base_native_config: MistralNativeConfig) -> None:
        restored = _hf_config_to_native_config(_native_config_to_hf_config(base_native_config))
        assert restored == base_native_config

    def test_reverse_with_quantization_config(self, base_native_config: MistralNativeConfig) -> None:
        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            head_dim=128,
            vocab_size=32000,
            max_position_embeddings=32768,
            sliding_window=None,
            quantization_config=_make_hf_fp8_quant_config(),
        )
        native = _hf_config_to_native_config(hf)
        assert native.quantization is None
        assert native.quantization_config is not None
        assert isinstance(native.quantization_config, QuantizationConfigMixin)
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=10000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
            quantization_config=native.quantization_config,
        )
        assert native == expected


class TestHFToNativeMinistral3:
    def test_basic_reverse(self, ministral3_native_config: MistralNativeConfig) -> None:
        hf = Ministral3Config(
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
                "llama_4_scaling_beta": 0.1,
            },
        )
        native = _hf_config_to_native_config(hf)
        assert native == ministral3_native_config

    def test_roundtrip(self, ministral3_native_config: MistralNativeConfig) -> None:
        restored = _hf_config_to_native_config(_native_config_to_hf_config(ministral3_native_config))
        assert restored == ministral3_native_config

    def test_roundtrip_with_quantization(self, yarn_args: YarnArgs, llama4_scaling: Llama4Scaling) -> None:
        native = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=262144,
            tied_embeddings=True,
            yarn=yarn_args,
            llama_4_scaling=llama4_scaling,
            quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        hf = _native_config_to_hf_config(native)
        restored = _hf_config_to_native_config(hf)
        assert restored.quantization is None
        assert restored.quantization_config is not None
        qc_dict = restored.quantization_config.to_dict()
        assert qc_dict["quant_method"] == "fp8"
        assert qc_dict["activation_scheme"] == "static"
        expected = MistralNativeConfig(
            dim=4096,
            n_layers=32,
            head_dim=128,
            hidden_dim=14336,
            n_heads=32,
            n_kv_heads=8,
            rope_theta=1000000.0,
            norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=262144,
            tied_embeddings=True,
            yarn=yarn_args,
            llama_4_scaling=llama4_scaling,
            quantization_config=restored.quantization_config,
        )
        assert restored == expected


class TestHFToNativeMistral4:
    def test_basic_reverse(self, mistral4_native_config: MistralNativeConfig) -> None:
        hf = Mistral4Config(
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
        )
        native = _hf_config_to_native_config(hf)
        assert native == mistral4_native_config

    def test_roundtrip(self, mistral4_native_config: MistralNativeConfig) -> None:
        restored = _hf_config_to_native_config(_native_config_to_hf_config(mistral4_native_config))
        assert restored == mistral4_native_config


class TestHFToNativeMistral3:
    def test_basic_reverse(self, mistral3_native_config: MistralNativeConfig) -> None:
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
            rope_theta=10000.0,
        )
        hf = Mistral3Config(
            text_config=text_config,
            vision_config=vision_config,
            multimodal_projector_bias=False,
            image_token_id=10,
            spatial_merge_size=2,
            vision_feature_layer=-1,
        )
        native = _hf_config_to_native_config(hf)
        assert native == mistral3_native_config

    def test_roundtrip_ignores_non_roundtrippable_fields(self, mistral3_native_config: MistralNativeConfig) -> None:
        restored = _hf_config_to_native_config(_native_config_to_hf_config(mistral3_native_config))
        assert restored == mistral3_native_config


def test_unsupported_hf_config_type_raises() -> None:
    from transformers.configuration_utils import PreTrainedConfig

    with pytest.raises(ValueError, match="Unsupported HF config type"):
        _hf_config_to_native_config(PreTrainedConfig())


def test_forward_moe_without_mla_raises(moe_args: MOEModelArgs) -> None:
    native = MistralNativeConfig(
        dim=4096,
        n_layers=32,
        head_dim=128,
        hidden_dim=14336,
        n_heads=32,
        n_kv_heads=8,
        rope_theta=10000.0,
        norm_eps=1e-5,
        vocab_size=32000,
        max_position_embeddings=32768,
        moe=moe_args,
    )
    with pytest.raises(ValueError, match="MOE and MLA"):
        _native_config_to_hf_config(native)
