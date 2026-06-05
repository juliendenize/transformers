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

import fnmatch
import gc
import json
import re
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import Any

from huggingface_hub import snapshot_download

from transformers.testing_utils import (
    backend_empty_cache,
    cleanup,
    require_torch,
    require_torch_accelerator,
    require_torchvision,
    slow,
    torch_device,
)
from transformers.utils import is_torch_available


if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import (
        AutoConfig,
        AutoTokenizer,
        Ministral3Config,
        Ministral3ForCausalLM,
        Mistral3Config,
        Mistral3ForConditionalGeneration,
        Mistral4Config,
        Mistral4ForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, revert_weight_conversion
    from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig

_TINY_HIDDEN = 32
_TINY_HEADS = 2
_TINY_KV_HEADS = 2
_TINY_LAYERS = 2
_TINY_VOCAB = 64
_TINY_INTERMEDIATE = 64
_TINY_HEAD_DIM = _TINY_HIDDEN // _TINY_HEADS

_LAYER_REVERSE_BASE = {
    "input_layernorm.weight": "attention_norm.weight",
    "post_attention_layernorm.weight": "ffn_norm.weight",
    "self_attn.q_proj.weight": "attention.wq.weight",
    "self_attn.k_proj.weight": "attention.wk.weight",
    "self_attn.v_proj.weight": "attention.wv.weight",
    "self_attn.o_proj.weight": "attention.wo.weight",
    "mlp.gate_proj.weight": "feed_forward.w1.weight",
    "mlp.down_proj.weight": "feed_forward.w2.weight",
    "mlp.up_proj.weight": "feed_forward.w3.weight",
}


_CONFIG_INTERNAL_KEYS = {
    "_loaded_from_mistral_format",
    "transformers_weights",
    "transformers_version",
    "_name_or_path",
    "_commit_hash",
    "architectures",
    "dtype",
    "sliding_window",
    "quant_config",
}

# Extra keys that the HF config's rope processing may add to rope_parameters
# but are not produced by the native-format conversion functions.
_ROPE_INTERNAL_KEYS = {"max_position_embeddings", "mscale", "type"}

# Key patterns that are expected to be "unexpected" when loading a consolidated
# Mistral-native checkpoint into a text-only model (VLM vision keys) or any
# model (QAT training artifacts like fake_quantizer).
_EXPECTED_UNEXPECTED_KEY_PATTERNS = {
    r"^vision_encoder\.",
    r"^vision_language_adapter\.",
    r"^patch_merger\.",
    r"^pre_mm_projector_norm\.",
    r"fake_quantizer",
}


# params.json keys that are not recoverable after an HF round-trip.
# vision_encoder: lost when loading a VLM checkpoint as a text-only model.
# quantization: native format, converted to HF quantization_config in the round-trip.
_NON_ROUNDTRIPPABLE_PARAMS_KEYS = {"vision_encoder"}

_MISTRAL_MODEL_TYPES = ["mistral", "mistral3", "ministral", "ministral3", "mixtral"]


def _filter_expected_unexpected_keys(keys: list[str]) -> list[str]:
    return [k for k in keys if not any(re.search(pat, k) for pat in _EXPECTED_UNEXPECTED_KEY_PATTERNS)]


def _strip_internal_keys(d: dict) -> dict:
    cleaned = {}
    for k, v in d.items():
        if k in _CONFIG_INTERNAL_KEYS:
            continue
        if isinstance(v, dict):
            v = _strip_internal_keys(v)
        cleaned[k] = v
    return cleaned


def _assert_config_matches(test_case: unittest.TestCase, original, reloaded) -> None:
    original_dict = _strip_internal_keys(original.to_dict())
    reloaded_dict = _strip_internal_keys(reloaded.to_dict())

    for d in (original_dict, reloaded_dict):
        if "rope_parameters" in d and isinstance(d["rope_parameters"], dict):
            d["rope_parameters"] = {k: v for k, v in d["rope_parameters"].items() if k not in _ROPE_INTERNAL_KEYS}

    test_case.assertEqual(
        original_dict,
        reloaded_dict,
        f"Config mismatch after roundtrip.\n  Original: {original_dict}\n  Reloaded: {reloaded_dict}",
    )


def _tiny_mistral_config() -> "MistralConfig":
    return MistralConfig(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_KV_HEADS,
        intermediate_size=_TINY_INTERMEDIATE,
        head_dim=_TINY_HEAD_DIM,
        vocab_size=_TINY_VOCAB,
        max_position_embeddings=64,
    )


def _tiny_ministral3_config() -> "Ministral3Config":
    return Ministral3Config(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_KV_HEADS,
        intermediate_size=_TINY_INTERMEDIATE,
        head_dim=_TINY_HEAD_DIM,
        vocab_size=_TINY_VOCAB,
        max_position_embeddings=64,
    )


def _tiny_mistral3_config() -> "Mistral3Config":
    vision_config = PixtralVisionConfig(
        hidden_size=_TINY_HIDDEN,
        intermediate_size=_TINY_INTERMEDIATE,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_channels=3,
        image_size=16,
        patch_size=4,
        hidden_act="silu",
    )
    text_config = _tiny_mistral_config()
    return Mistral3Config(
        vision_config=vision_config,
        text_config=text_config,
        image_token_index=10,
        spatial_merge_size=2,
        multimodal_projector_bias=False,
        tie_word_embeddings=False,
    )


def _tiny_mistral4_config() -> "Mistral4Config":
    return Mistral4Config(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_HEADS,
        intermediate_size=_TINY_INTERMEDIATE,
        vocab_size=_TINY_VOCAB,
        max_position_embeddings=64,
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        kv_lora_rank=16,
        v_head_dim=16,
        n_routed_experts=4,
        num_experts_per_tok=2,
        moe_intermediate_size=16,
        first_k_dense_replace=0,
        n_shared_experts=1,
        routed_scaling_factor=1.0,
        n_group=1,
        topk_group=1,
    )


def _inverse_rope_permute(tensor: "torch.Tensor", n_heads: int) -> "torch.Tensor":
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _hf_to_native_key(hf_key: str) -> str:
    _REVERSE_MAP = {
        "lm_head.weight": "output.weight",
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
    }
    if hf_key in _REVERSE_MAP:
        return _REVERSE_MAP[hf_key]

    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".", 3)  # ["model", "layers", "{i}", "rest"]
        layer_idx = parts[2]
        suffix = parts[3]
        if suffix in _LAYER_REVERSE_BASE:
            return f"layers.{layer_idx}.{_LAYER_REVERSE_BASE[suffix]}"
    return hf_key


def _hf_to_native_key_mistral3(hf_key: str) -> str:
    _TOP_LEVEL_MAP = {
        "lm_head.weight": "output.weight",
        "model.language_model.embed_tokens.weight": "tok_embeddings.weight",
        "model.language_model.norm.weight": "norm.weight",
    }
    if hf_key in _TOP_LEVEL_MAP:
        return _TOP_LEVEL_MAP[hf_key]

    _PROJECTOR_MAP = {
        "model.multi_modal_projector.linear_1": "vision_language_adapter.w_in",
        "model.multi_modal_projector.linear_2": "vision_language_adapter.w_out",
        "model.multi_modal_projector.patch_merger": "patch_merger",
        "model.multi_modal_projector.norm": "pre_mm_projector_norm",
    }
    for hf_prefix, native_prefix in _PROJECTOR_MAP.items():
        if hf_key.startswith(hf_prefix):
            return hf_key.replace(hf_prefix, native_prefix)

    _ATTN_REVERSE = {
        "q_proj": "wq",
        "k_proj": "wk",
        "v_proj": "wv",
        "o_proj": "wo",
    }
    _FF_REVERSE = {
        "gate_proj": "w1",
        "down_proj": "w2",
        "up_proj": "w3",
    }

    if hf_key.startswith("model.language_model.layers."):
        parts = hf_key.split(".", 4)
        layer_idx = parts[3]
        suffix = parts[4]
        if suffix in _LAYER_REVERSE_BASE:
            return f"layers.{layer_idx}.{_LAYER_REVERSE_BASE[suffix]}"

    if hf_key.startswith("model.vision_tower."):
        native_key = hf_key.replace("model.vision_tower.", "vision_encoder.")
        for hf_name, native_name in _ATTN_REVERSE.items():
            native_key = native_key.replace(f"attention.{hf_name}", f"attention.{native_name}")
        for hf_name, native_name in _FF_REVERSE.items():
            native_key = native_key.replace(f"feed_forward.{hf_name}", f"feed_forward.{native_name}")
        return native_key

    return hf_key


def _build_native_ministral3_checkpoint(tmpdir: Path) -> tuple["Ministral3Config", dict[str, "torch.Tensor"]]:
    config = _tiny_ministral3_config()

    with torch.device("meta"):
        model = Ministral3ForCausalLM(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key(hf_key)
        if "attention.wq.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
        elif "attention.wk.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_KV_HEADS)
        native_sd[native_key] = tensor

    save_file(native_sd, str(tmpdir / "consolidated.safetensors"))

    params = {
        "dim": _TINY_HIDDEN,
        "n_layers": _TINY_LAYERS,
        "hidden_dim": _TINY_INTERMEDIATE,
        "n_heads": _TINY_HEADS,
        "n_kv_heads": _TINY_KV_HEADS,
        "norm_eps": 1e-5,
        "head_dim": _TINY_HEAD_DIM,
        "vocab_size": _TINY_VOCAB,
        "max_position_embeddings": 64,
        "rope_theta": 1000000.0,
        "yarn": {
            "factor": 16.0,
            "original_max_position_embeddings": 16384,
            "beta": 32.0,
            "alpha": 1.0,
        },
        "llama_4_scaling": {
            "original_max_position_embeddings": 16384,
            "beta": 0.1,
        },
    }
    with open(tmpdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    config.save_pretrained(str(tmpdir))

    return config, hf_sd


def _build_native_mistral3_checkpoint(tmpdir: Path) -> tuple["Mistral3Config", dict[str, "torch.Tensor"]]:
    config = _tiny_mistral3_config()

    with torch.device("meta"):
        model = Mistral3ForConditionalGeneration(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key_mistral3(hf_key)
        if native_key.startswith("layers.") and "attention.wq.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
        elif native_key.startswith("layers.") and "attention.wk.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_KV_HEADS)
        elif native_key.startswith("vision_encoder.") and "attention.wq.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
        elif native_key.startswith("vision_encoder.") and "attention.wk.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
        native_sd[native_key] = tensor

    save_file(native_sd, str(tmpdir / "consolidated.safetensors"))

    vision_cfg = config.vision_config
    params = {
        "dim": _TINY_HIDDEN,
        "n_layers": _TINY_LAYERS,
        "hidden_dim": _TINY_INTERMEDIATE,
        "n_heads": _TINY_HEADS,
        "n_kv_heads": _TINY_KV_HEADS,
        "norm_eps": 1e-5,
        "head_dim": _TINY_HEAD_DIM,
        "vocab_size": _TINY_VOCAB,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "vision_encoder": {
            "hidden_size": vision_cfg.hidden_size,
            "num_hidden_layers": vision_cfg.num_hidden_layers,
            "num_attention_heads": vision_cfg.num_attention_heads,
            "patch_size": vision_cfg.patch_size,
            "image_size": vision_cfg.image_size,
            "intermediate_size": vision_cfg.intermediate_size,
            "num_channels": vision_cfg.num_channels,
            "max_image_size": vision_cfg.image_size,
            "rope_theta": 10000.0,
            "mm_projector_id": "patch_merge",
            "add_pre_mm_projector_layer_norm": True,
            "adapter_bias": config.multimodal_projector_bias,
            "spatial_merge_size": config.spatial_merge_size,
            "image_token_id": config.image_token_index,
            "image_break_token_id": 12,
            "image_end_token_id": 13,
        },
    }
    with open(tmpdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    config.save_pretrained(str(tmpdir))

    return config, hf_sd


def _build_native_mistral_checkpoint(tmpdir: Path) -> tuple["MistralConfig", dict[str, "torch.Tensor"]]:
    config = _tiny_mistral_config()

    with torch.device("meta"):
        model = MistralForCausalLM(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key(hf_key)
        if "attention.wq.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
        elif "attention.wk.weight" in native_key:
            tensor = _inverse_rope_permute(tensor, _TINY_KV_HEADS)
        native_sd[native_key] = tensor

    save_file(native_sd, str(tmpdir / "consolidated.safetensors"))

    params = {
        "dim": _TINY_HIDDEN,
        "n_layers": _TINY_LAYERS,
        "hidden_dim": _TINY_INTERMEDIATE,
        "n_heads": _TINY_HEADS,
        "n_kv_heads": _TINY_KV_HEADS,
        "norm_eps": 1e-5,
        "head_dim": _TINY_HEAD_DIM,
        "vocab_size": _TINY_VOCAB,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
    }
    with open(tmpdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    config.save_pretrained(str(tmpdir))

    return config, hf_sd


def _hf_to_native_key_mistral4(hf_key: str) -> str | None:
    _REVERSE_MAP = {
        "lm_head.weight": "output.weight",
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
    }
    if hf_key in _REVERSE_MAP:
        return _REVERSE_MAP[hf_key]

    _LAYER_REVERSE = {
        "input_layernorm.weight": "attention_norm.weight",
        "post_attention_layernorm.weight": "ffn_norm.weight",
        "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a_with_mqa.weight",
        "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
        "self_attn.q_a_proj.weight": "attention.wq_a.weight",
        "self_attn.q_b_proj.weight": "attention.wq_b.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.q_a_layernorm.weight": "attention.q_a_norm.weight",
        "self_attn.kv_a_layernorm.weight": "attention.kv_a_norm.weight",
        "mlp.gate.weight": "gate.weight",
        "mlp.shared_experts.gate_proj.weight": "shared_experts.w1.weight",
        "mlp.shared_experts.down_proj.weight": "shared_experts.w2.weight",
        "mlp.shared_experts.up_proj.weight": "shared_experts.w3.weight",
    }

    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".", 3)
        layer_idx = parts[2]
        suffix = parts[3]

        if "mlp.experts.gate_up_proj" in suffix or "mlp.experts.down_proj" in suffix:
            return None

        if suffix in _LAYER_REVERSE:
            return f"layers.{layer_idx}.{_LAYER_REVERSE[suffix]}"

    return hf_key


def _build_native_mistral4_checkpoint(tmpdir: Path) -> tuple["Mistral4Config", dict[str, "torch.Tensor"]]:
    config = _tiny_mistral4_config()

    with torch.device("meta"):
        model = Mistral4ForCausalLM(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key_mistral4(hf_key)
        if native_key is None:
            parts = hf_key.split(".", 3)
            layer_idx = parts[2]
            suffix = parts[3]

            if "gate_up_proj" in suffix:
                n_experts = tensor.shape[0]
                half = tensor.shape[1] // 2
                for e in range(n_experts):
                    native_sd[f"layers.{layer_idx}.experts.{e}.w1.weight"] = tensor[e, :half, :]
                    native_sd[f"layers.{layer_idx}.experts.{e}.w3.weight"] = tensor[e, half:, :]
            elif "down_proj" in suffix:
                n_experts = tensor.shape[0]
                for e in range(n_experts):
                    native_sd[f"layers.{layer_idx}.experts.{e}.w2.weight"] = tensor[e]
            continue

        native_sd[native_key] = tensor

    save_file(native_sd, str(tmpdir / "consolidated.safetensors"))

    params = {
        "dim": _TINY_HIDDEN,
        "n_layers": _TINY_LAYERS,
        "hidden_dim": _TINY_INTERMEDIATE,
        "n_heads": _TINY_HEADS,
        "n_kv_heads": _TINY_HEADS,
        "norm_eps": 1e-5,
        "head_dim": config.qk_nope_head_dim + config.qk_rope_head_dim,
        "vocab_size": _TINY_VOCAB,
        "max_position_embeddings": 64,
        "rope_theta": 10000.0,
        "q_lora_rank": 16,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "kv_lora_rank": 16,
        "v_head_dim": 16,
        "moe": {
            "num_experts": 4,
            "num_experts_per_tok": 2,
            "expert_hidden_dim": 16,
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
    with open(tmpdir / "params.json", "w", encoding="utf-8") as f:
        json.dump(params, f, ensure_ascii=False)

    config.save_pretrained(str(tmpdir))

    return config, hf_sd


@require_torch
class TestConversionMappingRegistration(unittest.TestCase):
    def _assert_has_entries(self, model_type: str) -> None:
        mapping = get_checkpoint_conversion_mapping(model_type)
        self.assertIsNotNone(mapping, f"No conversion mapping for {model_type!r}")
        self.assertGreater(len(mapping), 0, f"Empty conversion mapping for {model_type!r}")
        has_renaming = any(isinstance(e, WeightRenaming) for e in mapping)
        self.assertTrue(has_renaming, f"No WeightRenaming entries for {model_type!r}")

    def test_mistral(self) -> None:
        self._assert_has_entries("mistral")

    def test_ministral3(self) -> None:
        self._assert_has_entries("ministral3")

    def test_mistral3(self) -> None:
        mapping = get_checkpoint_conversion_mapping("mistral3")
        self.assertIsNotNone(mapping)
        self.assertGreater(len(mapping), 0, "Empty conversion mapping for 'mistral3'")
        has_renaming = any(isinstance(e, WeightRenaming) for e in mapping)
        self.assertTrue(has_renaming, "mistral3 should have WeightRenaming entries")
        has_converter = any(isinstance(e, WeightConverter) for e in mapping)
        self.assertTrue(has_converter, "mistral3 should have WeightConverter entries for RoPE permutation")

    def test_mistral4(self) -> None:
        mapping = get_checkpoint_conversion_mapping("mistral4")
        self.assertIsNotNone(mapping)
        has_converter = any(isinstance(e, WeightConverter) for e in mapping)
        self.assertTrue(has_converter, "mistral4 should have WeightConverter entries for MoE expert fusion")


@require_torch
class TestFromPretrained(unittest.TestCase):
    """Test loading models from native Mistral checkpoint format."""

    # (variant_id, model_cls, config_fn, build_fn)
    _VARIANTS = [
        ("mistral", MistralForCausalLM, _tiny_mistral_config, _build_native_mistral_checkpoint),
        ("ministral3", Ministral3ForCausalLM, _tiny_ministral3_config, _build_native_ministral3_checkpoint),
        ("mistral3", Mistral3ForConditionalGeneration, _tiny_mistral3_config, _build_native_mistral3_checkpoint),
        ("mistral4", Mistral4ForCausalLM, _tiny_mistral4_config, _build_native_mistral4_checkpoint),
    ]

    # Variants that support test_native_weights_match (Mistral4 excluded — MoE expert
    # fusion makes direct weight comparison non-trivial).
    _WEIGHTS_MATCH_VARIANTS = [
        ("mistral", MistralForCausalLM, _build_native_mistral_checkpoint),
        ("ministral3", Ministral3ForCausalLM, _build_native_ministral3_checkpoint),
        ("mistral3", Mistral3ForConditionalGeneration, _build_native_mistral3_checkpoint),
    ]

    def test_native_format(self) -> None:
        for variant_id, model_cls, _, build_fn in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    build_fn(tmp_path)
                    model = model_cls.from_pretrained(tmp_path, mistral_format=True)
                    self.assertIsInstance(model, model_cls)

    def test_native_weights_match(self) -> None:
        for variant_id, model_cls, build_fn in self._WEIGHTS_MATCH_VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    _, ref_sd = build_fn(tmp_path)
                    model = model_cls.from_pretrained(str(tmp_path), mistral_format=True)
                    loaded_sd = model.state_dict()
                    for key in ref_sd:
                        self.assertTrue(
                            torch.allclose(loaded_sd[key], ref_sd[key], atol=1e-6),
                            f"Weight mismatch for {key}",
                        )

    def test_hf_save_reload_roundtrip(self) -> None:
        for variant_id, model_cls, _, build_fn in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    native_dir = tmp_path / "native"
                    native_dir.mkdir()
                    save_dir = tmp_path / "saved"

                    build_fn(native_dir)
                    model = model_cls.from_pretrained(str(native_dir), mistral_format=True)
                    original_sd = {k: v.cpu() for k, v in model.state_dict().items()}
                    original_config = model.config

                    model.save_pretrained(str(save_dir), save_format="hf")
                    reloaded = model_cls.from_pretrained(str(save_dir))
                    reloaded_sd = reloaded.state_dict()

                    _assert_config_matches(self, original_config, reloaded.config)
                    for key in original_sd:
                        self.assertTrue(
                            torch.equal(original_sd[key], reloaded_sd[key].cpu()),
                            f"Roundtrip mismatch for {key}",
                        )

    def test_native_weight_conversion_keys_correct(self) -> None:
        for variant_id, model_cls, config_fn, build_fn in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    config = config_fn()
                    with torch.device("meta"):
                        ref_model = model_cls(config)
                    expected_keys = set(ref_model.state_dict().keys())

                    build_fn(tmp_path)
                    model = model_cls.from_pretrained(tmp_path, mistral_format=True)

                    actual_keys = set(model.state_dict().keys())
                    self.assertEqual(expected_keys, actual_keys)


@require_torch
class TestFromPretrainedFormatSelection(unittest.TestCase):
    def test_prefers_hf_when_both_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _build_native_mistral_checkpoint(tmp_path)

            hf_config = _tiny_mistral_config()
            hf_config.save_pretrained(str(tmp_path))

            loaded_config = MistralConfig.from_pretrained(str(tmp_path))
            self.assertFalse(getattr(loaded_config, "_loaded_from_mistral_format", False))

    def test_mistral_format_true(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _build_native_mistral_checkpoint(tmp_path)

            hf_config = _tiny_mistral_config()
            hf_config.save_pretrained(str(tmp_path))

            loaded_config = MistralConfig.from_pretrained(str(tmp_path), mistral_format=True)
            self.assertTrue(getattr(loaded_config, "_loaded_from_mistral_format", False))

    def test_mistral_format_false_no_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _build_native_mistral_checkpoint(tmp_path)
            config_json = tmp_path / "config.json"
            if config_json.exists():
                config_json.unlink()

            with self.assertRaises(OSError):
                MistralConfig.from_pretrained(str(tmp_path), mistral_format=False)


@require_torch
class TestRevertWeightConversion(unittest.TestCase):
    def test_no_regex_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _build_native_mistral_checkpoint(tmp_path)
            model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)

            sd = model.state_dict()
            reverted = revert_weight_conversion(model, sd)

            for key in reverted:
                self.assertNotIn("\\", key, f"Reverted key {key!r} contains regex escapes")

    def test_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            native_dir = tmp_path / "native"
            native_dir.mkdir()

            _build_native_mistral_checkpoint(native_dir)
            model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)

            original_sd = {k: v.cpu() for k, v in model.state_dict().items()}
            reverted = revert_weight_conversion(model, model.state_dict())

            for key in reverted:
                self.assertFalse(key.startswith("model."), f"Reverted key {key!r} still has HF prefix")

            save_dir = tmp_path / "saved"
            save_dir.mkdir()
            save_file(reverted, str(save_dir / "model.safetensors"))
            model.config.save_pretrained(str(save_dir))
            reloaded = MistralForCausalLM.from_pretrained(str(save_dir))

            self.assertEqual(set(original_sd.keys()), set(reloaded.state_dict().keys()))
            for key in original_sd:
                self.assertTrue(
                    torch.equal(original_sd[key], reloaded.state_dict()[key].cpu()),
                    f"Roundtrip mismatch for {key}",
                )

    def test_returns_all_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            _build_native_mistral_checkpoint(tmp_path)
            model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)

            sd = model.state_dict()
            reverted = revert_weight_conversion(model, sd)
            self.assertEqual(len(sd), len(reverted))


@require_torch
class TestSavePretrained(unittest.TestCase):
    def test_default_hf_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _tiny_mistral_config()
            model = MistralForCausalLM(config)

            model.save_pretrained(tmp_path)
            self.assertTrue((tmp_path / "config.json").exists())
            self.assertFalse((tmp_path / "params.json").exists())

    def test_default_preserves_native_format(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            native_dir = tmp_path / "native"
            native_dir.mkdir()
            save_dir = tmp_path / "saved"

            _build_native_mistral_checkpoint(native_dir)
            model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
            model.save_pretrained(str(save_dir))

            from safetensors.torch import load_file

            saved_sd = load_file(str(save_dir / "model.safetensors"))

            self.assertIn("output.weight", saved_sd)
            self.assertNotIn("lm_head.weight", saved_sd)

    def test_force_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            native_dir = tmp_path / "native"
            native_dir.mkdir()
            save_dir = tmp_path / "saved"

            _build_native_mistral_checkpoint(native_dir)
            model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
            model.save_pretrained(str(save_dir), save_format="hf")

            from safetensors.torch import load_file

            saved_sd = load_file(str(save_dir / "model.safetensors"))
            self.assertIn("lm_head.weight", saved_sd)
            self.assertNotIn("output.weight", saved_sd)

    def test_force_mistral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _tiny_mistral_config()
            model = MistralForCausalLM(config)

            model.save_pretrained(tmp_path, save_format="mistral")
            self.assertTrue((tmp_path / "params.json").exists())
            self.assertTrue((tmp_path / "consolidated.safetensors").exists())
            self.assertFalse((tmp_path / "config.json").exists())
            self.assertFalse((tmp_path / "generation_config.json").exists())

    def test_invalid_save_format_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            config = _tiny_mistral_config()
            model = MistralForCausalLM(config)

            with self.assertRaises(ValueError):
                model.save_pretrained(tmp_path, save_format="invalid")


@require_torch
class TestSaveLoadRoundtrip(unittest.TestCase):
    """Test native <-> HF format save/load roundtrips."""

    # (variant_id, model_cls, config_fn, build_fn)
    _VARIANTS = [
        ("mistral", MistralForCausalLM, _tiny_mistral_config, _build_native_mistral_checkpoint),
        ("ministral3", Ministral3ForCausalLM, _tiny_ministral3_config, _build_native_ministral3_checkpoint),
        ("mistral3", Mistral3ForConditionalGeneration, _tiny_mistral3_config, _build_native_mistral3_checkpoint),
        ("mistral4", Mistral4ForCausalLM, _tiny_mistral4_config, _build_native_mistral4_checkpoint),
    ]

    def test_native_to_hf_to_native(self) -> None:
        for variant_id, model_cls, _, build_fn in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    native_dir = tmp_path / "native"
                    native_dir.mkdir()
                    hf_dir = tmp_path / "hf"
                    native2_dir = tmp_path / "native2"

                    build_fn(native_dir)
                    model = model_cls.from_pretrained(str(native_dir), mistral_format=True)
                    original_config = model.config
                    original_sd = {k: v.cpu() for k, v in model.state_dict().items()}

                    model.save_pretrained(str(hf_dir), save_format="hf")
                    model2 = model_cls.from_pretrained(str(hf_dir))
                    _assert_config_matches(self, original_config, model2.config)

                    # Intermediate HF weight check: verify HF save/reload preserves weights.
                    for key in original_sd:
                        self.assertTrue(
                            torch.equal(original_sd[key], model2.state_dict()[key].cpu()),
                            f"HF save/reload mismatch for {key}",
                        )

                    model2.save_pretrained(str(native2_dir), save_format="mistral")
                    model3 = model_cls.from_pretrained(str(native2_dir), mistral_format=True)

                    _assert_config_matches(self, original_config, model3.config)
                    for key in original_sd:
                        self.assertTrue(
                            torch.equal(original_sd[key], model3.state_dict()[key].cpu()),
                            f"Roundtrip mismatch for {key}",
                        )

    def test_hf_to_native_to_hf(self) -> None:
        for variant_id, model_cls, config_fn, _ in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    tmp_path = Path(tmp_dir)
                    config = config_fn()
                    with torch.device("meta"):
                        model = model_cls(config)
                    ref_sd = {k: torch.randn(v.shape) for k, v in model.named_parameters()}

                    hf_dir = tmp_path / "hf"
                    hf_dir.mkdir()
                    native_dir = tmp_path / "native"
                    hf2_dir = tmp_path / "hf2"

                    save_file(ref_sd, str(hf_dir / "model.safetensors"))
                    config.save_pretrained(str(hf_dir))

                    model = model_cls.from_pretrained(str(hf_dir))
                    original_config = model.config
                    model.save_pretrained(str(native_dir), save_format="mistral")

                    model2 = model_cls.from_pretrained(str(native_dir), mistral_format=True)
                    _assert_config_matches(self, original_config, model2.config)

                    # Intermediate native weight check: verify HF-to-native conversion preserves weights.
                    for key in ref_sd:
                        self.assertTrue(
                            torch.equal(ref_sd[key], model2.state_dict()[key].cpu()),
                            f"HF-to-native reload mismatch for {key}",
                        )

                    model2.save_pretrained(str(hf2_dir), save_format="hf")

                    model3 = model_cls.from_pretrained(str(hf2_dir))

                    _assert_config_matches(self, original_config, model3.config)
                    for key in ref_sd:
                        self.assertTrue(
                            torch.equal(ref_sd[key], model3.state_dict()[key]),
                            f"Roundtrip mismatch for {key}",
                        )

    # --- Mistral-specific tests (not parameterized) ---

    def test_config_native_to_hf_to_native(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            native_dir = tmp_path / "native"
            native_dir.mkdir()
            hf_dir = tmp_path / "hf"
            native2_dir = tmp_path / "native2"

            _build_native_mistral_checkpoint(native_dir)
            model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)

            model.save_pretrained(str(hf_dir), save_format="hf")
            model2 = MistralForCausalLM.from_pretrained(str(hf_dir))

            model2.save_pretrained(str(native2_dir), save_format="mistral")

            with open(native2_dir / "params.json", encoding="utf-8") as f:
                params = json.load(f)
            self.assertEqual(params["dim"], _TINY_HIDDEN)
            self.assertEqual(params["n_layers"], _TINY_LAYERS)
            self.assertEqual(params["n_heads"], _TINY_HEADS)

    def test_preserves_model_output(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            native_dir = tmp_path / "native"
            native_dir.mkdir()
            save_dir = tmp_path / "saved"

            _build_native_mistral_checkpoint(native_dir)
            model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
            model.eval()

            input_ids = torch.randint(0, _TINY_VOCAB, (1, 8))
            with torch.no_grad():
                original_output = model(input_ids).logits.clone()

            model.save_pretrained(str(save_dir), save_format="hf")
            reloaded = MistralForCausalLM.from_pretrained(str(save_dir))
            reloaded.eval()

            with torch.no_grad():
                reloaded_output = reloaded(input_ids).logits

            self.assertTrue(torch.equal(original_output, reloaded_output))


_FAKE_TEKKEN_SPECIAL_TOKENS = [
    {"rank": 0, "token_str": "<unk>", "is_control": True},
    {"rank": 1, "token_str": "<s>", "is_control": True},
    {"rank": 2, "token_str": "</s>", "is_control": True},
    {"rank": 3, "token_str": "[INST]", "is_control": True},
    {"rank": 4, "token_str": "[/INST]", "is_control": True},
    {"rank": 5, "token_str": "[AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 6, "token_str": "[/AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 7, "token_str": "[TOOL_RESULTS]", "is_control": True},
    {"rank": 8, "token_str": "[/TOOL_RESULTS]", "is_control": True},
    {"rank": 9, "token_str": "[TOOL_CALLS]", "is_control": True},
    {"rank": 10, "token_str": "[IMG]", "is_control": True},
    {"rank": 11, "token_str": "<pad>", "is_control": True},
    {"rank": 12, "token_str": "[IMG_BREAK]", "is_control": True},
    {"rank": 13, "token_str": "[IMG_END]", "is_control": True},
    {"rank": 14, "token_str": "[PREFIX]", "is_control": True},
    {"rank": 15, "token_str": "[MIDDLE]", "is_control": True},
    {"rank": 16, "token_str": "[SUFFIX]", "is_control": True},
    {"rank": 17, "token_str": "[SYSTEM_PROMPT]", "is_control": True},
    {"rank": 18, "token_str": "[/SYSTEM_PROMPT]", "is_control": True},
    {"rank": 19, "token_str": "[TOOL_CONTENT]", "is_control": True},
]

_FAKE_TEKKEN_CONFIG = {
    "pattern": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
    "num_vocab_tokens": _TINY_VOCAB - len(_FAKE_TEKKEN_SPECIAL_TOKENS),
    "default_vocab_size": _TINY_VOCAB,
    "default_num_special_tokens": len(_FAKE_TEKKEN_SPECIAL_TOKENS),
    "version": "v3",
}


def _build_fake_tekken_json(
    directory: Path,
    vocab_size: int = _TINY_VOCAB,
    image_config: dict | None = None,
) -> Path:
    import base64

    num_special = len(_FAKE_TEKKEN_SPECIAL_TOKENS)
    num_bpe = vocab_size - num_special

    vocab_list: list[dict] = []
    for rank in range(num_bpe):
        raw_byte = bytes([rank % 256])
        vocab_list.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(raw_byte).decode("ascii"),
                "token_str": None,
            }
        )

    tekken_data: dict = {
        "vocab": vocab_list,
        "special_tokens": _FAKE_TEKKEN_SPECIAL_TOKENS,
        "config": dict(_FAKE_TEKKEN_CONFIG),
        "version": 1,
        "type": "tekken",
    }

    if image_config is not None:
        tekken_data["image"] = image_config

    output_path = directory / "tekken.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tekken_data, f, ensure_ascii=False)

    return output_path


@require_torch
class TestTokenizerMistralFormatRoundtrip(unittest.TestCase):
    def test_tekken_metadata_preserved_in_hf(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_path = _build_fake_tekken_json(tmp_path)
            from transformers.integrations.mistral import convert_tekken_tokenizer

            tokenizer = convert_tekken_tokenizer(str(tekken_path))
            metadata = tokenizer.init_kwargs.get("tekken_metadata")

            self.assertIsNotNone(metadata, "tekken_metadata not stored in init_kwargs")
            self.assertEqual(metadata["config"]["version"], "v3")
            self.assertEqual(metadata["version"], 1)
            self.assertEqual(metadata["type"], "tekken")
            self.assertIn("pattern", metadata["config"])

    def test_tekken_to_hf_to_tekken(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_dir = tmp_path / "tekken"
            tekken_dir.mkdir()
            hf_dir = tmp_path / "hf"
            hf_dir.mkdir()
            reconstructed_dir = tmp_path / "reconstructed"
            reconstructed_dir.mkdir()

            tekken_path = _build_fake_tekken_json(tekken_dir)
            from transformers.integrations.mistral import convert_tekken_tokenizer, save_as_tekken

            original_tok = convert_tekken_tokenizer(str(tekken_path))
            original_tok.save_pretrained(str(hf_dir))

            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))
            save_as_tekken(reloaded_tok, reconstructed_dir)

            self.assertTrue((reconstructed_dir / "tekken.json").exists())

            with open(tekken_path, encoding="utf-8") as f:
                original = json.load(f)
            with open(reconstructed_dir / "tekken.json", encoding="utf-8") as f:
                reconstructed = json.load(f)

            self.assertEqual(original, reconstructed)

    def test_tekken_to_hf_to_tekken_tokenizers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_dir = tmp_path / "tekken"
            tekken_dir.mkdir()
            hf_dir = tmp_path / "hf"
            hf_dir.mkdir()
            reconstructed_dir = tmp_path / "reconstructed"
            reconstructed_dir.mkdir()

            tekken_path = _build_fake_tekken_json(tekken_dir)
            from transformers.integrations.mistral import convert_tekken_tokenizer, save_as_tekken

            original_tok = convert_tekken_tokenizer(str(tekken_path))
            original_tok.save_pretrained(str(hf_dir))

            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir), mistral_format=False)
            save_as_tekken(reloaded_tok, reconstructed_dir)

            self.assertTrue((reconstructed_dir / "tekken.json").exists())

            with open(tekken_path, encoding="utf-8") as f:
                original = json.load(f)
            with open(reconstructed_dir / "tekken.json", encoding="utf-8") as f:
                reconstructed = json.load(f)

            self.assertEqual(original, reconstructed)

    def test_tekken_multimodal_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            image_config = {
                "image_patch_size": 16,
                "max_image_size": 1024,
                "spatial_merge_size": 2,
            }
            tekken_path = _build_fake_tekken_json(tmp_path, image_config=image_config)

            from transformers.integrations.mistral import convert_tekken_tokenizer, save_as_tekken

            tok = convert_tekken_tokenizer(str(tekken_path))
            metadata = tok.init_kwargs["tekken_metadata"]
            self.assertEqual(metadata["image"], image_config)

            reconstructed_dir = tmp_path / "reconstructed"
            save_as_tekken(tok, reconstructed_dir)

            with open(reconstructed_dir / "tekken.json", encoding="utf-8") as f:
                reconstructed = json.load(f)

            self.assertEqual(reconstructed["image"], image_config)

    def test_save_format_mistral_on_hf_tokenizer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_dir = tmp_path / "tekken"
            tekken_dir.mkdir()
            save_dir = tmp_path / "saved"
            save_dir.mkdir()

            tekken_path = _build_fake_tekken_json(tekken_dir)
            from transformers.integrations.mistral import convert_tekken_tokenizer

            tok = convert_tekken_tokenizer(str(tekken_path))
            files = tok.save_pretrained(str(save_dir), save_format="mistral")

            self.assertTrue(any("tekken.json" in f for f in files))
            self.assertTrue((save_dir / "tekken.json").exists())
            self.assertFalse((save_dir / "tokenizer.json").exists())

    def test_save_format_invalid_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            tekken_dir = tmp_path / "tekken"
            tekken_dir.mkdir()
            tekken_path = _build_fake_tekken_json(tekken_dir)

            from transformers.integrations.mistral import convert_tekken_tokenizer

            tok = convert_tekken_tokenizer(str(tekken_path))
            with self.assertRaisesRegex(ValueError, "Unknown save_format"):
                tok.save_pretrained(str(tmp_path / "out"), save_format="bad")


class TestConvertTekkenProcessor(unittest.TestCase):
    def test_missing_vision_encoder_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from transformers.integrations.mistral.tokenizer import convert_tekken_image_processor

            tekken_path = _build_fake_tekken_json(tmp_path)
            params_path = tmp_path / "params.json"
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump({"dim": 128, "n_heads": 2}, f)

            with self.assertRaisesRegex(ValueError, "vision_encoder"):
                convert_tekken_image_processor(str(tekken_path), str(params_path))

    @require_torchvision
    def test_valid_processor_creation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from transformers.integrations.mistral.tokenizer import convert_tekken_image_processor

            tekken_path = _build_fake_tekken_json(tmp_path)
            params_path = tmp_path / "params.json"
            params = {
                "dim": 128,
                "n_heads": 2,
                "vision_encoder": {
                    "patch_size": 14,
                    "image_size": 1540,
                    "max_image_size": 1540,
                    "spatial_merge_size": 2,
                },
            }
            with open(params_path, "w", encoding="utf-8") as f:
                json.dump(params, f)

            processor = convert_tekken_image_processor(str(tekken_path), str(params_path))
            self.assertIsNotNone(processor)
            self.assertTrue(hasattr(processor, "tokenizer"))
            self.assertTrue(hasattr(processor, "image_processor"))


class TestSaveAsTekkenErrorPath(unittest.TestCase):
    def test_missing_metadata_raises(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            from unittest.mock import MagicMock

            from transformers.integrations.mistral.tokenizer import save_as_tekken

            tokenizer = MagicMock()
            tokenizer.tekken_metadata = None
            tokenizer.init_kwargs = {}

            with self.assertRaisesRegex(ValueError, "tekken_metadata"):
                save_as_tekken(tokenizer, str(tmp_path))


@require_torch
class TestSaveMistralFormatRecovery(unittest.TestCase):
    _MODEL_CLS_TO_TYPE = {
        "MistralForCausalLM": "mistral",
        "Ministral3ForCausalLM": "ministral3",
        "Mistral3ForConditionalGeneration": "mistral3",
        "Mistral4ForCausalLM": "mistral4",
    }

    def _run_roundtrip(
        self,
        tmp_path: Path,
        build_fn: Callable[[Path], Any],
        model_cls: type,
        tokenizer_backend: str | None = None,
    ) -> None:
        from safetensors.torch import load_file

        from transformers.integrations.mistral import convert_tekken_tokenizer

        native_dir = tmp_path / "native"
        native_dir.mkdir()
        hf_dir = tmp_path / "hf"
        native2_dir = tmp_path / "native2"

        _build_fake_tekken_json(native_dir)
        build_fn(native_dir)

        # Snapshot original native artifacts.
        original_native_sd = load_file(str(native_dir / "consolidated.safetensors"))
        with open(native_dir / "params.json", encoding="utf-8") as f:
            original_params = json.load(f)
        with open(native_dir / "tekken.json", encoding="utf-8") as f:
            original_tekken = json.load(f)

        # native → HF
        model = model_cls.from_pretrained(str(native_dir), mistral_format=True)
        tok = convert_tekken_tokenizer(str(native_dir / "tekken.json"))

        hf_sd = {k: v.cpu() for k, v in model.state_dict().items()}

        model.save_pretrained(str(hf_dir), save_format="hf")
        tok.save_pretrained(str(hf_dir))

        # Intermediate HF weight check: verify HF save/reload preserves weights.
        reloaded_hf = model_cls.from_pretrained(str(hf_dir))
        for key in hf_sd:
            self.assertTrue(
                torch.equal(hf_sd[key], reloaded_hf.state_dict()[key].cpu()),
                f"HF save/reload mismatch for {key}",
            )
        del reloaded_hf

        # HF → save as mistral
        reloaded = model_cls.from_pretrained(str(hf_dir))
        if tokenizer_backend is not None:
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir), mistral_format=False)
        else:
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))

        reloaded.save_pretrained(str(native2_dir), save_format="mistral")
        reloaded_tok.save_pretrained(str(native2_dir), save_format="mistral")

        # --- Direct file comparison, no from_pretrained on native2_dir ---

        # Weights: key names and tensor values must match original native checkpoint.
        recovered_sd = load_file(str(native2_dir / "consolidated.safetensors"))
        self.assertEqual(
            set(recovered_sd.keys()),
            set(original_native_sd.keys()),
            f"Key mismatch.\n  Missing: {set(original_native_sd) - set(recovered_sd)}"
            f"\n  Extra: {set(recovered_sd) - set(original_native_sd)}",
        )
        for key in original_native_sd:
            self.assertTrue(torch.equal(original_native_sd[key], recovered_sd[key]), f"Tensor mismatch for {key}")

        # params.json: recovered must contain all keys from original (may have extra defaults).
        with open(native2_dir / "params.json", encoding="utf-8") as f:
            recovered_params = json.load(f)
        for key, value in original_params.items():
            self.assertIn(key, recovered_params, f"params.json missing key {key!r}")
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    self.assertEqual(
                        recovered_params[key][sub_key], sub_value, f"params.json mismatch on {key!r}.{sub_key!r}"
                    )
            else:
                self.assertEqual(recovered_params[key], value, f"params.json mismatch on {key!r}")

        # tekken.json metadata + structure must match original.
        with open(native2_dir / "tekken.json", encoding="utf-8") as f:
            recovered_tekken = json.load(f)
        self.assertEqual(recovered_tekken["config"]["version"], original_tekken["config"]["version"])
        self.assertEqual(recovered_tekken["version"], original_tekken["version"])
        self.assertEqual(recovered_tekken["type"], original_tekken["type"])
        self.assertEqual(len(recovered_tekken["vocab"]), len(original_tekken["vocab"]))
        self.assertEqual(len(recovered_tekken["special_tokens"]), len(original_tekken["special_tokens"]))

        # HF artifacts must not be present.
        self.assertFalse((native2_dir / "config.json").exists())
        self.assertFalse((native2_dir / "generation_config.json").exists())
        self.assertFalse((native2_dir / "tokenizer.json").exists())

    def test_mistral(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(tmp_path, _build_native_mistral_checkpoint, MistralForCausalLM)

    def test_ministral3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(tmp_path, _build_native_ministral3_checkpoint, Ministral3ForCausalLM)

    def test_mistral3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(tmp_path, _build_native_mistral3_checkpoint, Mistral3ForConditionalGeneration)

    def test_mistral4(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(tmp_path, _build_native_mistral4_checkpoint, Mistral4ForCausalLM)

    def test_mistral_tokenizers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(
                tmp_path, _build_native_mistral_checkpoint, MistralForCausalLM, tokenizer_backend="TokenizersBackend"
            )

    def test_ministral3_tokenizers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(
                tmp_path,
                _build_native_ministral3_checkpoint,
                Ministral3ForCausalLM,
                tokenizer_backend="TokenizersBackend",
            )

    def test_mistral3_tokenizers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(
                tmp_path,
                _build_native_mistral3_checkpoint,
                Mistral3ForConditionalGeneration,
                tokenizer_backend="TokenizersBackend",
            )

    def test_mistral4_tokenizers_backend(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            self._run_roundtrip(
                tmp_path, _build_native_mistral4_checkpoint, Mistral4ForCausalLM, tokenizer_backend="TokenizersBackend"
            )


_MISTRAL_DOWNLOAD_PATTERNS = [
    "params.json",
    "tekken.json",
    "consolidated*.safetensors*",
]

_HF_OPTIONAL_FILES = {"generation_config.json"}


def _download_mistral_files(
    model_id: str,
    target_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    if allow_patterns is None:
        allow_patterns = list(_MISTRAL_DOWNLOAD_PATTERNS)
    snapshot_download(repo_id=model_id, local_dir=str(target_dir), allow_patterns=allow_patterns)
    return target_dir


def _assert_dir_contains_exactly(
    test_case: unittest.TestCase,
    directory: Path,
    expected_patterns: set[str],
    optional_patterns: set[str] | None = None,
) -> None:
    if optional_patterns is None:
        optional_patterns = set()

    actual_files = {f.name for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")}

    unmatched_files: set[str] = set()
    for filename in actual_files:
        is_expected = any(fnmatch.fnmatch(filename, pat) for pat in expected_patterns)
        is_optional = any(fnmatch.fnmatch(filename, pat) for pat in optional_patterns)
        if not is_expected and not is_optional:
            unmatched_files.add(filename)

    test_case.assertFalse(
        unmatched_files,
        f"Unexpected files in {directory}: {sorted(unmatched_files)}. "
        f"Expected patterns: {sorted(expected_patterns)}, optional: {sorted(optional_patterns)}",
    )

    for pattern in expected_patterns:
        has_match = any(fnmatch.fnmatch(f, pattern) for f in actual_files)
        test_case.assertTrue(
            has_match, f"Expected pattern {pattern!r} has no match in {directory}: {sorted(actual_files)}"
        )


def _assert_mistral_dir(test_case: unittest.TestCase, directory: Path) -> None:
    expected = {"params.json", "tekken.json", "consolidated*.safetensors*"}
    _assert_dir_contains_exactly(test_case, directory, expected)

    for forbidden in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        test_case.assertFalse(
            (directory / forbidden).exists(),
            f"HF artifact {forbidden!r} found in Mistral-native directory {directory}",
        )
    hf_weights = list(directory.glob("model*.safetensors*"))
    test_case.assertFalse(
        hf_weights,
        f"HF weight files found in Mistral-native directory {directory}: {sorted(f.name for f in hf_weights)}",
    )


def _assert_hf_dir(test_case: unittest.TestCase, directory: Path, weight_patterns: set[str] | None = None) -> None:
    if weight_patterns is None:
        weight_patterns = {"model*.safetensors*"}
    expected = {"config.json"} | weight_patterns

    optional = _HF_OPTIONAL_FILES | {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    _assert_dir_contains_exactly(test_case, directory, expected, optional)

    for forbidden in ("params.json", "tekken.json"):
        test_case.assertFalse(
            (directory / forbidden).exists(),
            f"Mistral artifact {forbidden!r} found in HF directory {directory}",
        )
    consolidated_files = list(directory.glob("consolidated*"))
    test_case.assertFalse(
        consolidated_files,
        f"Consolidated weight files found in HF directory {directory}: {sorted(f.name for f in consolidated_files)}",
    )


def _load_all_safetensors(directory: Path) -> dict:
    r"""Load all ``consolidated*.safetensors`` shards from a directory into a single state dict."""
    from safetensors.torch import load_file

    sd: dict = {}
    for sf_path in sorted(directory.glob("consolidated*.safetensors")):
        if sf_path.suffix == ".json":
            continue
        sd.update(load_file(str(sf_path)))
    return sd


def _assert_native_roundtrip(
    test_case: unittest.TestCase,
    model_cls: type,
    model_id: str,
    tmp_path: Path,
    *,
    unexpected_key_patterns: set[str] = _EXPECTED_UNEXPECTED_KEY_PATTERNS,
    tokenizer_backends: tuple[str | None, ...] = (None, "TokenizersBackend"),
) -> None:
    r"""Shared helper: download native → HF → save as native → compare files.

    Model conversion (expensive) runs once.  Tokenizer roundtrip (cheap) is
    tested for every entry in *tokenizer_backends*.
    """

    def _load_tokenizer(path: str, backend: str | None) -> "AutoTokenizer":
        if backend is not None:
            return AutoTokenizer.from_pretrained(path, mistral_format=False)
        return AutoTokenizer.from_pretrained(path)

    mistral_dir = tmp_path / "mistral"
    hf_dir = tmp_path / "hf"

    _download_mistral_files(model_id, mistral_dir)

    original_native_sd = _load_all_safetensors(mistral_dir)
    with open(mistral_dir / "params.json", encoding="utf-8") as f:
        original_params = json.load(f)
    with open(mistral_dir / "tekken.json", encoding="utf-8") as f:
        original_tekken = json.load(f)

    # native → HF (model weights — done once)
    model = model_cls.from_pretrained(
        str(mistral_dir), mistral_format=True, device_map=torch_device, torch_dtype=torch.bfloat16
    )
    model.save_pretrained(str(hf_dir), save_format="hf")
    del model
    backend_empty_cache(torch_device)
    gc.collect()

    # Test tokenizer roundtrip for each backend
    for backend in tokenizer_backends:
        backend_label = backend or "default"
        with test_case.subTest(tokenizer_backend=backend_label):
            native_dir = tmp_path / f"native_{backend_label}"

            # Clean up any tekken.json left by a previous backend iteration
            # to prevent cross-contamination via resolve_mistral_format.
            tekken_in_hf = hf_dir / "tekken.json"
            if tekken_in_hf.exists():
                tekken_in_hf.unlink()

            original_tok = _load_tokenizer(str(mistral_dir), backend)
            original_tok.save_pretrained(str(hf_dir))

            # HF → save as native
            model2 = model_cls.from_pretrained(str(hf_dir), device_map=torch_device, torch_dtype=torch.bfloat16)
            reloaded_tok = _load_tokenizer(str(hf_dir), backend)
            model2.save_pretrained(str(native_dir), save_format="mistral")
            reloaded_tok.save_pretrained(str(native_dir), save_format="mistral")
            del model2
            backend_empty_cache(torch_device)
            gc.collect()

            # --- Assertions ---
            _assert_mistral_dir(test_case, native_dir)

            recovered_sd = _load_all_safetensors(native_dir)
            expected_keys = {
                k for k in original_native_sd if not any(re.search(p, k) for p in unexpected_key_patterns)
            }
            test_case.assertEqual(
                set(recovered_sd.keys()),
                expected_keys,
                f"Key mismatch.\n  Missing: {expected_keys - set(recovered_sd)}"
                f"\n  Extra: {set(recovered_sd) - expected_keys}",
            )
            for key in expected_keys:
                test_case.assertTrue(
                    torch.equal(original_native_sd[key], recovered_sd[key]), f"Tensor mismatch for {key}"
                )

            with open(native_dir / "params.json", encoding="utf-8") as f:
                recovered_params = json.load(f)
            for key, value in original_params.items():
                if key not in recovered_params:
                    test_case.assertIn(
                        key, _NON_ROUNDTRIPPABLE_PARAMS_KEYS, f"params.json unexpectedly missing key {key!r}"
                    )
                    continue
                recovered_value = recovered_params[key]
                if isinstance(value, dict):
                    if recovered_value is None:
                        test_case.assertIn(
                            key,
                            _NON_ROUNDTRIPPABLE_PARAMS_KEYS,
                            f"params.json key {key!r} is null but original is a dict",
                        )
                        continue
                    for sub_key, sub_value in value.items():
                        test_case.assertEqual(
                            recovered_value[sub_key], sub_value, f"params.json mismatch on {key!r}.{sub_key!r}"
                        )
                else:
                    test_case.assertEqual(recovered_value, value, f"params.json mismatch on {key!r}")

            with open(native_dir / "tekken.json", encoding="utf-8") as f:
                recovered_tekken = json.load(f)
            test_case.assertEqual(recovered_tekken["config"]["version"], original_tekken["config"]["version"])
            test_case.assertEqual(recovered_tekken.get("type"), original_tekken.get("type"))
            test_case.assertEqual(len(recovered_tekken["vocab"]), len(original_tekken["vocab"]))
            test_case.assertEqual(len(recovered_tekken["special_tokens"]), len(original_tekken["special_tokens"]))


def _assert_native_format_roundtrip(
    test_case: unittest.TestCase,
    model_cls: type,
    model_id: str,
    tmp_path: Path,
    torch_dtype: "torch.dtype",
    tokenizer_backends: tuple[str | None, ...] = (None, "TokenizersBackend"),
) -> None:
    r"""Shared helper for slow native-format roundtrip tests.

    Model conversion (expensive) runs once.  Tokenizer roundtrip (cheap) is
    tested for every entry in *tokenizer_backends*.
    """

    def _load_tokenizer(path: str, backend: str | None) -> "AutoTokenizer":
        if backend is not None:
            return AutoTokenizer.from_pretrained(path, mistral_format=False)
        return AutoTokenizer.from_pretrained(path)

    mistral_dir = tmp_path / "mistral"
    hf_dir = tmp_path / "hf"

    _download_mistral_files(model_id, mistral_dir)
    _assert_mistral_dir(test_case, mistral_dir)

    model, loading_info = model_cls.from_pretrained(
        str(mistral_dir),
        mistral_format=True,
        device_map=torch_device,
        torch_dtype=torch_dtype,
        output_loading_info=True,
    )
    unexpected = _filter_expected_unexpected_keys(loading_info["unexpected_keys"])
    test_case.assertFalse(unexpected, f"Unexpected keys during native load: {unexpected}")
    test_case.assertFalse(
        loading_info["missing_keys"], f"Missing keys during native load: {loading_info['missing_keys']}"
    )
    original_config = model.config
    original_sd = {k: v.cpu() for k, v in model.state_dict().items()}

    model.save_pretrained(str(hf_dir), save_format="hf")
    del model
    backend_empty_cache(torch_device)
    gc.collect()

    _assert_hf_dir(test_case, hf_dir)

    reloaded_config = AutoConfig.from_pretrained(str(hf_dir))
    _assert_config_matches(test_case, original_config, reloaded_config)

    # Model weight roundtrip (done once, independent of tokenizer backend)
    reloaded, reload_info = model_cls.from_pretrained(
        str(hf_dir), device_map=torch_device, torch_dtype=torch_dtype, output_loading_info=True
    )
    test_case.assertFalse(
        reload_info["unexpected_keys"], f"Unexpected keys during HF reload: {reload_info['unexpected_keys']}"
    )
    test_case.assertFalse(reload_info["missing_keys"], f"Missing keys during HF reload: {reload_info['missing_keys']}")
    reloaded_sd = reloaded.state_dict()

    for key in original_sd:
        test_case.assertTrue(torch.equal(original_sd[key], reloaded_sd[key].cpu()), f"Roundtrip mismatch for {key}")

    del reloaded
    backend_empty_cache(torch_device)
    gc.collect()

    # Tokenizer roundtrip for each backend
    for backend in tokenizer_backends:
        backend_label = backend or "default"
        with test_case.subTest(tokenizer_backend=backend_label):
            # Clean up any tekken.json left by a previous backend iteration
            # to prevent cross-contamination via resolve_mistral_format.
            tekken_in_hf = hf_dir / "tekken.json"
            if tekken_in_hf.exists():
                tekken_in_hf.unlink()

            original_tok = _load_tokenizer(str(mistral_dir), backend)
            original_tok.save_pretrained(str(hf_dir))
            reloaded_tok = _load_tokenizer(str(hf_dir), backend)
            test_case.assertEqual(reloaded_tok.vocab_size, original_tok.vocab_size)
            test_text = "Hello, world!"
            test_case.assertEqual(reloaded_tok.encode(test_text), original_tok.encode(test_text))
            test_case.assertEqual(
                reloaded_tok.decode(reloaded_tok.encode(test_text), skip_special_tokens=True), test_text
            )


@slow
@require_torch_accelerator
class TestRealModelIntegration(unittest.TestCase):
    """End-to-end tests with real Mistral models from the Hub."""

    # (variant_id, model_cls, model_id, torch_dtype, unexpected_key_patterns)
    _VARIANTS = [
        (
            "mistral",
            MistralForCausalLM,
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            torch.float16,
            _EXPECTED_UNEXPECTED_KEY_PATTERNS,
        ),
        (
            "ministral3",
            Ministral3ForCausalLM,
            "mistralai/Ministral-3-3B-Instruct-2512",
            torch.bfloat16,
            _EXPECTED_UNEXPECTED_KEY_PATTERNS,
        ),
        (
            "mistral3",
            Mistral3ForConditionalGeneration,
            "mistralai/Mistral-Small-3.2-24B-Instruct-2506",
            torch.bfloat16,
            {r"fake_quantizer"},
        ),
        (
            "mistral4",
            Mistral4ForCausalLM,
            "mistralai/Mistral-Small-4-119B-2603",
            torch.bfloat16,
            _EXPECTED_UNEXPECTED_KEY_PATTERNS,
        ),
    ]

    def setUp(self) -> None:
        cleanup(torch_device, gc_collect=True)

    def tearDown(self) -> None:
        cleanup(torch_device, gc_collect=True)

    def test_native_format_roundtrip(self) -> None:
        for variant_id, model_cls, model_id, dtype, _ in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _assert_native_format_roundtrip(self, model_cls, model_id, Path(tmp_dir), dtype)
                cleanup(torch_device, gc_collect=True)

    def test_save_recovers_native_format(self) -> None:
        for variant_id, model_cls, model_id, _, unexpected_patterns in self._VARIANTS:
            with self.subTest(variant=variant_id):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    _assert_native_roundtrip(
                        self, model_cls, model_id, Path(tmp_dir), unexpected_key_patterns=unexpected_patterns
                    )
                cleanup(torch_device, gc_collect=True)
