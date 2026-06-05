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

import unittest

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.params_conversion import (
    Llama4Scaling,
    MistralNativeConfig,
    QuantizationArgs,
    YarnArgs,
    _extract_rope_theta,
    _get_maybe_quant_config,
    _hf_config_to_native_config,
    _native_config_to_hf_config,
    native_config_for_model_type,
    native_config_from_hf_config,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig
from transformers.quantizers.auto import AutoQuantizationConfig
from transformers.utils.quantization_config import QuantizationConfigMixin

from .mistral_fixture_data import (
    base_native_config,
    expected_ministral3_hf_config,
    expected_mistral3_hf_config,
    expected_mistral4_hf_config,
    expected_mistral_hf_config,
    llama4_scaling,
    ministral3_native_config,
    mistral3_native_config,
    mistral4_native_config,
    moe_args,
    vision_encoder_args,
    yarn_args,
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


def _make_non_reversible_quant_config() -> QuantizationConfigMixin:
    return AutoQuantizationConfig.from_dict(
        {
            "quant_method": "gptq",
            "bits": 4,
            "group_size": 128,
        }
    )


class TestQuantizationArgs(unittest.TestCase):
    def test_valid_tensor_scheme(self) -> None:
        config = QuantizationArgs("fp8_e4m3", "TENSOR")
        self.assertEqual(config, QuantizationArgs("fp8_e4m3", "TENSOR"))

    def test_invalid_scheme_raises(self) -> None:
        for scheme in ["DYNAMIC", "UNSUPPORTED", "static"]:
            with self.subTest(scheme=scheme):
                with self.assertRaisesRegex(ValueError, scheme):
                    QuantizationArgs("fp8_e4m3", scheme)

    def test_unsupported_format_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "fp8_e4m3"):
            QuantizationArgs("int8", "TENSOR")


class TestGetMaybeQuantConfig(unittest.TestCase):
    def test_none_returns_none(self) -> None:
        self.assertIsNone(_get_maybe_quant_config(is_vision_model=False, quantization_args=None))

    def test_tensor_produces_static(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=False,
            quantization_args=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        qc_dict = qc.to_dict()
        self.assertEqual(qc_dict["quant_method"], "fp8")
        self.assertEqual(qc_dict["activation_scheme"], "static")

    def test_vision_model_adds_modules_to_skip(self) -> None:
        qc = _get_maybe_quant_config(
            is_vision_model=True,
            quantization_args=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        qc_dict = qc.to_dict()
        self.assertIn("model.vision_tower", qc_dict["modules_to_not_convert"])
        self.assertIn("model.multi_modal_projector", qc_dict["modules_to_not_convert"])
        self.assertIn("lm_head", qc_dict["modules_to_not_convert"])


class TestMistralNativeConfig(unittest.TestCase):
    def test_mutual_exclusivity_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Cannot set both"):
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

    def test_single_or_no_quantization_ok(self) -> None:
        test_cases = [
            ("native_only", {"quantization": QuantizationArgs("fp8_e4m3", "TENSOR")}),
            ("hf_only", {"quantization_config": _make_hf_fp8_quant_config()}),
            ("neither", {}),
        ]
        for test_id, quant_kwargs in test_cases:
            with self.subTest(test_id):
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
                self.assertEqual(native.quantization is not None, has_native)
                self.assertEqual(native.quantization_config is not None, has_hf)


class TestNativeToHF(unittest.TestCase):
    def test_native_to_hf(self) -> None:
        _CONFIG_MAP = {
            "base_native_config": base_native_config,
            "ministral3_native_config": ministral3_native_config,
            "mistral4_native_config": mistral4_native_config,
            "expected_mistral_hf_config": expected_mistral_hf_config,
            "expected_ministral3_hf_config": expected_ministral3_hf_config,
            "expected_mistral4_hf_config": expected_mistral4_hf_config,
        }
        test_cases = [
            ("mistral", "base_native_config", MistralConfig, "expected_mistral_hf_config"),
            ("ministral3", "ministral3_native_config", Ministral3Config, "expected_ministral3_hf_config"),
            ("mistral4", "mistral4_native_config", Mistral4Config, "expected_mistral4_hf_config"),
        ]
        for test_id, native_fixture, expected_type, expected_fixture in test_cases:
            with self.subTest(test_id):
                native_config_val = _CONFIG_MAP[native_fixture]()
                expected_hf_config_val = _CONFIG_MAP[expected_fixture]()
                hf = _native_config_to_hf_config(native_config_val)
                self.assertIsInstance(hf, expected_type)
                self.assertEqual(hf, expected_hf_config_val)


class TestNativeToHFMistral3(unittest.TestCase):
    def test_vision_model(self) -> None:
        native = mistral3_native_config()
        expected = expected_mistral3_hf_config()
        hf = _native_config_to_hf_config(native)
        self.assertIsInstance(hf, Mistral3Config)
        self.assertEqual(hf, expected)

    def test_tensor_fp8_propagates_to_mistral3(self) -> None:
        ve_args = vision_encoder_args()
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
            vision_encoder=ve_args,
        )
        hf = _native_config_to_hf_config(native)
        qc = hf.quantization_config
        if hasattr(qc, "to_dict"):
            qc = qc.to_dict()
        self.assertEqual(qc["activation_scheme"], "static")


class TestQuantizationConfigPassthrough(unittest.TestCase):
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
        self.assertEqual(qc_dict["activation_scheme"], "dynamic")
        self.assertEqual(qc_dict["quant_method"], "fp8")

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
        self.assertEqual(qc_dict["activation_scheme"], "static")


class TestHFToNativeMistral(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = base_native_config()
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
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        config = base_native_config()
        restored = _hf_config_to_native_config(_native_config_to_hf_config(config))
        self.assertEqual(restored, config)

    def test_reverse_with_quantization_config(self) -> None:
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
        self.assertIsNotNone(native.quantization)
        self.assertIsNone(native.quantization_config)
        self.assertEqual(native.quantization, QuantizationArgs("fp8_e4m3", "TENSOR"))
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
            quantization=native.quantization,
        )
        self.assertEqual(native, expected)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = _make_non_reversible_quant_config()
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
            quantization_config=hf_quant,
        )
        native = _hf_config_to_native_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMinistral3(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = ministral3_native_config()
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
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        """Roundtrip for both apply_scale values, checking mscale is preserved.

        Regression: mscale was missing from the forward conversion, causing
        _compute_yarn_parameters to compute a wrong attention_factor (~1.28
        instead of 1.0 when apply_scale=False).
        """
        for apply_scale, expected_mscale in [(False, 1.0), (True, 0.0)]:
            with self.subTest(apply_scale=apply_scale):
                ya = YarnArgs(
                    factor=16.0, original_max_position_embeddings=16384, beta=32, alpha=1, apply_scale=apply_scale
                )
                l4s = Llama4Scaling(original_max_position_embeddings=16384, beta=0.1)
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
                    yarn=ya,
                    llama_4_scaling=l4s,
                )
                hf = _native_config_to_hf_config(native)
                rope = hf.rope_parameters
                assert "mscale" in rope, f"mscale missing from rope_parameters when {apply_scale=}"
                assert rope["mscale"] == expected_mscale, (
                    f"Expected mscale={expected_mscale} for {apply_scale=}, got {rope['mscale']}"
                )
                assert rope["mscale_all_dim"] == expected_mscale, (
                    f"Expected mscale_all_dim={expected_mscale} for {apply_scale=}, got {rope['mscale_all_dim']}"
                )
                restored = _hf_config_to_native_config(hf)
                self.assertEqual(restored, native)

    def test_roundtrip_with_quantization(self) -> None:
        ya = yarn_args()
        l4s = llama4_scaling()
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
            yarn=ya,
            llama_4_scaling=l4s,
            quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        hf = _native_config_to_hf_config(native)
        restored = _hf_config_to_native_config(hf)
        self.assertIsNotNone(restored.quantization)
        self.assertIsNone(restored.quantization_config)
        self.assertEqual(restored.quantization, QuantizationArgs("fp8_e4m3", "TENSOR"))
        self.assertEqual(restored, native)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = _make_non_reversible_quant_config()
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
                "llama_4_scaling_beta": 0.1,
            },
            quantization_config=hf_quant,
        )
        native = _hf_config_to_native_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMistral4(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = mistral4_native_config()
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
        self.assertEqual(native, config)

    def test_roundtrip(self) -> None:
        config = mistral4_native_config()
        restored = _hf_config_to_native_config(_native_config_to_hf_config(config))
        self.assertEqual(restored, config)

    def test_roundtrip_with_quantization(self) -> None:
        _moe_args = moe_args()
        native = MistralNativeConfig(
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
            yarn=YarnArgs(
                factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0, apply_scale=False
            ),
            llama_4_scaling=Llama4Scaling(original_max_position_embeddings=8192, beta=0.1),
            moe=_moe_args,
            quantization=QuantizationArgs("fp8_e4m3", "TENSOR"),
        )
        hf = _native_config_to_hf_config(native)
        restored = _hf_config_to_native_config(hf)
        self.assertIsNotNone(restored.quantization)
        self.assertIsNone(restored.quantization_config)
        self.assertEqual(restored.quantization, QuantizationArgs("fp8_e4m3", "TENSOR"))
        self.assertEqual(restored, native)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = _make_non_reversible_quant_config()
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
            quantization_config=hf_quant,
        )
        native = _hf_config_to_native_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestHFToNativeMistral3(unittest.TestCase):
    def test_basic_reverse(self) -> None:
        config = mistral3_native_config()
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
        self.assertEqual(native, config)

    def test_roundtrip_ignores_non_roundtrippable_fields(self) -> None:
        config = mistral3_native_config()
        restored = _hf_config_to_native_config(_native_config_to_hf_config(config))
        self.assertEqual(restored, config)

    def test_reverse_with_non_reversible_quant(self) -> None:
        hf_quant = _make_non_reversible_quant_config()
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
            quantization_config=hf_quant,
        )
        native = _hf_config_to_native_config(hf)
        self.assertIsNone(native.quantization)
        self.assertIsNotNone(native.quantization_config)
        self.assertEqual(native.quantization_config.to_dict()["quant_method"], "gptq")


class TestErrorPaths(unittest.TestCase):
    """Error paths for params conversion functions."""

    # --- From original TestErrorPaths ---

    def test_native_config_for_unknown_model_type_raises(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown Mistral model type"):
            native_config_for_model_type("unknown_model", {"dim": 128})

    def test_native_config_from_hf_unknown_model_type_raises(self) -> None:
        config = MistralConfig()
        with self.assertRaisesRegex(ValueError, "Unknown Mistral model type"):
            native_config_from_hf_config("unknown_model", config)

    def test_unsupported_hf_config_type_raises(self) -> None:
        from transformers.configuration_utils import PreTrainedConfig

        with self.assertRaisesRegex(ValueError, "Unsupported HF config type"):
            _hf_config_to_native_config(PreTrainedConfig())

    def test_forward_moe_without_mla_raises(self) -> None:
        moe = moe_args()
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
            moe=moe,
        )
        with self.assertRaisesRegex(ValueError, "MOE and MLA"):
            _native_config_to_hf_config(native)

    # --- From original TestErrorPathsParamsConversion ---

    def test_base_mistral_with_yarn_raises(self) -> None:
        from transformers.integrations.mistral.params_conversion import _native_config_to_mistral

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
            yarn=YarnArgs(
                factor=16.0, original_max_position_embeddings=16384, beta=32.0, alpha=1.0, apply_scale=False
            ),
        )
        with self.assertRaisesRegex(ValueError, "must not have llama_4_scaling or yarn"):
            _native_config_to_mistral(native)

    def test_ministral3_without_yarn_raises(self) -> None:
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
        )
        with self.assertRaisesRegex(ValueError, "requires yarn"):
            from transformers.integrations.mistral.params_conversion import _native_config_to_ministral3

            _native_config_to_ministral3(native)

    def test_mistral3_without_vision_raises(self) -> None:
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
        )
        with self.assertRaisesRegex(ValueError, "requires vision_encoder"):
            from transformers.integrations.mistral.params_conversion import _native_config_to_mistral3

            _native_config_to_mistral3(native)

    def test_qk_rope_nope_mismatch_raises(self) -> None:
        from transformers.integrations.mistral.params_conversion import _get_rope_parameters

        with self.assertRaisesRegex(ValueError, "qk_rope and qk_nope must both be None or both set"):
            _get_rope_parameters(
                rope_theta=10000.0,
                yarn_args=None,
                llama4_scaling=None,
                qk_rope=64,
                qk_nope=None,
            )

    def test_hf_mistral_missing_head_dim_raises(self) -> None:
        from transformers.integrations.mistral.params_conversion import _hf_mistral_to_native

        hf = MistralConfig(
            hidden_size=4096,
            num_hidden_layers=32,
            intermediate_size=14336,
            num_attention_heads=32,
            num_key_value_heads=8,
            rms_norm_eps=1e-5,
            vocab_size=32000,
            max_position_embeddings=32768,
        )
        # MistralConfig auto-computes head_dim; force it to None to test the guard
        hf.head_dim = None
        with self.assertRaisesRegex(ValueError, "head_dim must be set"):
            _hf_mistral_to_native(hf)

    # --- From original TestErrorPathsAdditional ---

    def test_extract_rope_theta_missing_raises(self) -> None:
        from types import SimpleNamespace

        config = SimpleNamespace(rope_scaling=None)
        with self.assertRaisesRegex(ValueError, "rope_theta"):
            _extract_rope_theta(config)
