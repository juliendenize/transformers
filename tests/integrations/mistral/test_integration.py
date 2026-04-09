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
r"""Tests for Phase 5+6: from_pretrained pipeline, save_pretrained native format, and roundtrips."""

import fnmatch
import gc
import json
import re
import tempfile
import unittest
from pathlib import Path

import pytest
from huggingface_hub import snapshot_download

from transformers.testing_utils import (
    backend_empty_cache,
    cleanup,
    require_torch,
    require_torch_accelerator,
    slow,
    torch_device,
)


if True:  # guarded import block for test discovery
    from transformers.utils import is_torch_available

if is_torch_available():
    import torch
    from safetensors.torch import save_file

    from transformers import (
        AutoConfig,
        AutoTokenizer,
        Ministral3Config,
        Ministral3ForCausalLM,
        Mistral3ForConditionalGeneration,
        Mistral4Config,
        Mistral4ForCausalLM,
        MistralConfig,
        MistralForCausalLM,
    )
    from transformers.conversion_mapping import get_checkpoint_conversion_mapping
    from transformers.core_model_loading import WeightConverter, WeightRenaming, revert_weight_conversion


# ---------------------------------------------------------------------------
# Tiny config helpers
# ---------------------------------------------------------------------------

_TINY_HIDDEN = 32
_TINY_HEADS = 2
_TINY_KV_HEADS = 2
_TINY_LAYERS = 2
_TINY_VOCAB = 64
_TINY_INTERMEDIATE = 64
_TINY_HEAD_DIM = _TINY_HIDDEN // _TINY_HEADS  # 16


_CONFIG_INTERNAL_KEYS = {
    "_loaded_from_mistral_format",
    "transformers_weights",
    "transformers_version",
    "_name_or_path",
    "_commit_hash",
    # architectures is set by save_pretrained based on the model class, not by config loading.
    "architectures",
    # dtype is set at model-load time (from weights), not config-load time.
    "dtype",
    # sliding_window / quant_config may be absent in configs created directly
    # but present after a native-format roundtrip.
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


def _filter_expected_unexpected_keys(keys: list[str]) -> list[str]:
    r"""Remove keys matching known-ignorable patterns from an unexpected-keys list."""
    return [k for k in keys if not any(re.search(pat, k) for pat in _EXPECTED_UNEXPECTED_KEY_PATTERNS)]


def _assert_config_matches(original, reloaded) -> None:
    r"""Assert that a config survived a save/reload roundtrip.

    Compares all config fields except internal metadata that may legitimately
    differ between a freshly-constructed config and one loaded from disk.
    Rope parameters are compared after stripping keys that the HF config
    processing adds but the native conversion does not produce.
    """
    original_dict = {k: v for k, v in original.to_dict().items() if k not in _CONFIG_INTERNAL_KEYS}
    reloaded_dict = {k: v for k, v in reloaded.to_dict().items() if k not in _CONFIG_INTERNAL_KEYS}

    # Normalise rope_parameters before comparison: strip HF-only internal keys
    for d in (original_dict, reloaded_dict):
        if "rope_parameters" in d and isinstance(d["rope_parameters"], dict):
            d["rope_parameters"] = {k: v for k, v in d["rope_parameters"].items() if k not in _ROPE_INTERNAL_KEYS}

    assert original_dict == reloaded_dict, (
        f"Config mismatch after roundtrip.\n  Original: {original_dict}\n  Reloaded: {reloaded_dict}"
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


def _tiny_mistral4_config() -> "Mistral4Config":
    return Mistral4Config(
        hidden_size=_TINY_HIDDEN,
        num_hidden_layers=_TINY_LAYERS,
        num_attention_heads=_TINY_HEADS,
        num_key_value_heads=_TINY_HEADS,
        intermediate_size=_TINY_INTERMEDIATE,
        vocab_size=_TINY_VOCAB,
        max_position_embeddings=64,
        # MLA fields
        q_lora_rank=16,
        qk_rope_head_dim=8,
        qk_nope_head_dim=8,
        kv_lora_rank=16,
        v_head_dim=16,
        # MoE fields (tiny)
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
    r"""Inverse of PermuteForRope._apply.

    Forward is `view(n, h/2, 2, d).transpose(1,2).reshape(...)`.
    Inverse is `view(n, 2, h/2, d).transpose(1,2).reshape(...)`.
    """
    dim1, dim2 = tensor.shape
    return tensor.view(n_heads, 2, dim1 // n_heads // 2, dim2).transpose(1, 2).reshape(dim1, dim2)


def _hf_to_native_key(hf_key: str) -> str:
    r"""Map an HF state dict key back to native Mistral format (base model only)."""
    _REVERSE_MAP = {
        "lm_head.weight": "output.weight",
        "model.embed_tokens.weight": "tok_embeddings.weight",
        "model.norm.weight": "norm.weight",
    }
    if hf_key in _REVERSE_MAP:
        return _REVERSE_MAP[hf_key]

    # Layer-level mappings
    _LAYER_REVERSE = {
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
    # model.layers.{i}.XXX -> layers.{i}.YYY
    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".", 3)  # ["model", "layers", "{i}", "rest"]
        layer_idx = parts[2]
        suffix = parts[3]
        if suffix in _LAYER_REVERSE:
            return f"layers.{layer_idx}.{_LAYER_REVERSE[suffix]}"
    return hf_key  # fallback: keep as-is


def _build_native_mistral_checkpoint(tmpdir: Path) -> "MistralConfig":
    r"""Create a tiny native-format Mistral checkpoint.

    Writes `params.json` + `consolidated.safetensors` (native format) and also
    `config.json` (needed by `adjust_generation_fn` fallback). Use
    `mistral_format=True` with `from_pretrained` to force native weight loading
    when `config.json` is present.
    """
    config = _tiny_mistral_config()

    with torch.device("meta"):
        model = MistralForCausalLM(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key(hf_key)
        # Q/K weights need inverse RoPE permutation
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

    # Write config.json so adjust_generation_fn can fall back to it
    config.save_pretrained(str(tmpdir))

    return config


def _hf_to_native_key_mistral4(hf_key: str) -> str | None:
    r"""Map an HF state dict key back to native Mistral4 format.

    Returns `None` for keys that are part of fused tensors (experts) which
    need special handling.
    """
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
        # MLA attention keys
        "self_attn.kv_a_proj_with_mqa.weight": "attention.wkv_a_with_mqa.weight",
        "self_attn.kv_b_proj.weight": "attention.wkv_b.weight",
        "self_attn.q_a_proj.weight": "attention.wq_a.weight",
        "self_attn.q_b_proj.weight": "attention.wq_b.weight",
        "self_attn.o_proj.weight": "attention.wo.weight",
        "self_attn.q_a_layernorm.weight": "attention.q_a_norm.weight",
        "self_attn.kv_a_layernorm.weight": "attention.kv_a_norm.weight",
        # Router
        "mlp.gate.weight": "gate.weight",
        # Shared experts
        "mlp.shared_experts.gate_proj.weight": "shared_experts.w1.weight",
        "mlp.shared_experts.down_proj.weight": "shared_experts.w2.weight",
        "mlp.shared_experts.up_proj.weight": "shared_experts.w3.weight",
    }

    if hf_key.startswith("model.layers."):
        parts = hf_key.split(".", 3)
        layer_idx = parts[2]
        suffix = parts[3]

        # Skip fused expert tensors — handled separately
        if "mlp.experts.gate_up_proj" in suffix or "mlp.experts.down_proj" in suffix:
            return None

        if suffix in _LAYER_REVERSE:
            return f"layers.{layer_idx}.{_LAYER_REVERSE[suffix]}"

    return hf_key


def _build_native_mistral4_checkpoint(tmpdir: Path) -> "Mistral4Config":
    r"""Create a tiny native-format Mistral4 checkpoint."""
    config = _tiny_mistral4_config()

    with torch.device("meta"):
        model = Mistral4ForCausalLM(config)

    hf_sd = {name: torch.randn(param.shape) for name, param in model.named_parameters()}

    native_sd: dict[str, torch.Tensor] = {}
    for hf_key, tensor in hf_sd.items():
        native_key = _hf_to_native_key_mistral4(hf_key)
        if native_key is None:
            # Fused expert tensor — split back to per-expert individual weights
            parts = hf_key.split(".", 3)
            layer_idx = parts[2]
            suffix = parts[3]

            if "gate_up_proj" in suffix:
                # Shape: (n_experts, 2*moe_intermediate, hidden_size)
                n_experts = tensor.shape[0]
                half = tensor.shape[1] // 2
                for e in range(n_experts):
                    native_sd[f"layers.{layer_idx}.experts.{e}.w1.weight"] = tensor[e, :half, :]
                    native_sd[f"layers.{layer_idx}.experts.{e}.w3.weight"] = tensor[e, half:, :]
            elif "down_proj" in suffix:
                # Shape: (n_experts, hidden_size, moe_intermediate)
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

    # Write config.json for adjust_generation_fn fallback
    config.save_pretrained(str(tmpdir))

    return config


# ---------------------------------------------------------------------------
# Test classes
# ---------------------------------------------------------------------------


@require_torch
class TestConversionMappingRegistration:
    r"""Verify that all Mistral model types are registered in the conversion mapping."""

    def _assert_has_entries(self, model_type: str) -> None:
        mapping = get_checkpoint_conversion_mapping(model_type)
        assert mapping is not None, f"No conversion mapping for {model_type!r}"
        assert len(mapping) > 0, f"Empty conversion mapping for {model_type!r}"
        has_renaming = any(isinstance(e, WeightRenaming) for e in mapping)
        assert has_renaming, f"No WeightRenaming entries for {model_type!r}"

    def test_mistral(self):
        self._assert_has_entries("mistral")

    def test_ministral3(self):
        self._assert_has_entries("ministral3")

    def test_mistral4(self):
        mapping = get_checkpoint_conversion_mapping("mistral4")
        assert mapping is not None
        has_converter = any(isinstance(e, WeightConverter) for e in mapping)
        assert has_converter, "mistral4 should have WeightConverter entries for MoE expert fusion"


@require_torch
class TestMistralFromPretrained:
    r"""End-to-end from_pretrained tests with tiny native Mistral checkpoints."""

    def test_native_format(self, tmp_path):
        r"""Loading from native format produces a valid model."""
        _build_native_mistral_checkpoint(tmp_path)
        model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)
        assert isinstance(model, MistralForCausalLM)

    def test_native_weights_match(self, tmp_path):
        r"""Weights loaded from native format match the original HF model weights."""
        config = _tiny_mistral_config()
        with torch.device("meta"):
            ref_model = MistralForCausalLM(config)
        ref_sd = {name: torch.randn(param.shape) for name, param in ref_model.named_parameters()}

        # Build native checkpoint from the reference state dict
        native_sd: dict[str, torch.Tensor] = {}
        for hf_key, tensor in ref_sd.items():
            native_key = _hf_to_native_key(hf_key)
            if "attention.wq.weight" in native_key:
                tensor = _inverse_rope_permute(tensor, _TINY_HEADS)
            elif "attention.wk.weight" in native_key:
                tensor = _inverse_rope_permute(tensor, _TINY_KV_HEADS)
            native_sd[native_key] = tensor.clone()

        save_file(native_sd, str(tmp_path / "consolidated.safetensors"))
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
        with open(tmp_path / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)

        # Write config.json for adjust_generation_fn fallback
        config.save_pretrained(str(tmp_path))

        model = MistralForCausalLM.from_pretrained(str(tmp_path), mistral_format=True)

        loaded_sd = model.state_dict()
        for key in ref_sd:
            assert torch.allclose(loaded_sd[key], ref_sd[key], atol=1e-6), f"Weight mismatch for {key}"

    def test_hf_save_reload_roundtrip(self, tmp_path):
        r"""Load native → save → reload produces identical weights and config.

        `save_pretrained` applies `revert_weight_conversion`, saving weights
        with native keys. Reloading re-applies the forward conversion mapping
        (registered for model type `"mistral"`), restoring HF key names.
        """
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        save_dir = tmp_path / "saved"

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        original_config = model.config

        model.save_pretrained(str(save_dir))
        # Saved dir has config.json + model.safetensors (with native keys).
        # Reloading applies the conversion mapping automatically.
        reloaded = MistralForCausalLM.from_pretrained(str(save_dir))
        reloaded_sd = reloaded.state_dict()

        _assert_config_matches(original_config, reloaded.config)
        assert set(original_sd.keys()) == set(reloaded_sd.keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], reloaded_sd[key]), f"Roundtrip mismatch for {key}"

    def test_native_weight_conversion_keys_correct(self, tmp_path):
        r"""All expected HF state dict keys are present after loading from native format."""
        config = _tiny_mistral_config()
        with torch.device("meta"):
            ref_model = MistralForCausalLM(config)
        expected_keys = set(ref_model.state_dict().keys())

        _build_native_mistral_checkpoint(tmp_path)
        model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)

        actual_keys = set(model.state_dict().keys())
        assert expected_keys == actual_keys


@require_torch
class TestMinistral3FromPretrained:
    r"""Ministral3 uses the same weight layout as base Mistral (with FP8 scales for quantized models)."""

    def test_native_format(self, tmp_path):
        r"""Loading a tiny Ministral3 native checkpoint succeeds with correct config."""
        config = Ministral3Config(
            hidden_size=_TINY_HIDDEN,
            num_hidden_layers=_TINY_LAYERS,
            num_attention_heads=_TINY_HEADS,
            num_key_value_heads=_TINY_KV_HEADS,
            intermediate_size=_TINY_INTERMEDIATE,
            head_dim=_TINY_HEAD_DIM,
            vocab_size=_TINY_VOCAB,
            max_position_embeddings=64,
        )
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

        save_file(native_sd, str(tmp_path / "consolidated.safetensors"))
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
        with open(tmp_path / "params.json", "w", encoding="utf-8") as f:
            json.dump(params, f, ensure_ascii=False)

        # Write config.json for adjust_generation_fn fallback
        config.save_pretrained(str(tmp_path))

        loaded = Ministral3ForCausalLM.from_pretrained(str(tmp_path), mistral_format=True)

        assert isinstance(loaded, Ministral3ForCausalLM)
        loaded_sd = loaded.state_dict()
        for key in hf_sd:
            assert torch.allclose(loaded_sd[key], hf_sd[key], atol=1e-6), f"Weight mismatch for {key}"


@require_torch
class TestMistral4FromPretrained:
    r"""End-to-end from_pretrained tests with tiny native Mistral4 (MoE/MLA) checkpoints."""

    def test_native_format(self, tmp_path):
        r"""Loading from native Mistral4 format produces a valid model."""
        _build_native_mistral4_checkpoint(tmp_path)
        model = Mistral4ForCausalLM.from_pretrained(tmp_path, mistral_format=True)
        assert isinstance(model, Mistral4ForCausalLM)

    def test_hf_save_reload_roundtrip(self, tmp_path):
        r"""Load native Mistral4 → save HF → reload produces identical weights and config."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        hf_dir = tmp_path / "hf"

        _build_native_mistral4_checkpoint(native_dir)
        model = Mistral4ForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        original_config = model.config

        model.save_pretrained(str(hf_dir))
        reloaded = Mistral4ForCausalLM.from_pretrained(str(hf_dir))
        reloaded_sd = reloaded.state_dict()

        _assert_config_matches(original_config, reloaded.config)
        for key in original_sd:
            assert torch.equal(original_sd[key], reloaded_sd[key]), f"Roundtrip mismatch for {key}"


@require_torch
class TestFromPretrainedFormatSelection:
    r"""Verify format selection logic: HF preferred, mistral_format overrides."""

    def test_prefers_hf_when_both_exist(self, tmp_path):
        r"""When both config.json and params.json exist, HF format is preferred."""
        _build_native_mistral_checkpoint(tmp_path)

        # Also save HF format — the HF one should be preferred
        hf_config = _tiny_mistral_config()
        hf_config.save_pretrained(str(tmp_path))

        loaded_config = MistralConfig.from_pretrained(str(tmp_path))
        # When loaded from HF, _loaded_from_mistral_format should not be set
        assert not getattr(loaded_config, "_loaded_from_mistral_format", False)

    def test_mistral_format_true(self, tmp_path):
        r"""mistral_format=True forces native format loading."""
        _build_native_mistral_checkpoint(tmp_path)

        # Also save HF format
        hf_config = _tiny_mistral_config()
        hf_config.save_pretrained(str(tmp_path))

        loaded_config = MistralConfig.from_pretrained(str(tmp_path), mistral_format=True)
        assert getattr(loaded_config, "_loaded_from_mistral_format", False)

    def test_mistral_format_false_no_hf(self, tmp_path):
        r"""mistral_format=False without HF config raises."""
        _build_native_mistral_checkpoint(tmp_path)
        # Remove config.json if it was created
        config_json = tmp_path / "config.json"
        if config_json.exists():
            config_json.unlink()

        with pytest.raises(OSError):
            MistralConfig.from_pretrained(str(tmp_path), mistral_format=False)


# ---------------------------------------------------------------------------
# Phase 6: save_pretrained native format + roundtrip
# ---------------------------------------------------------------------------


@require_torch
class TestRevertWeightConversion:
    r"""Verify revert_weight_conversion produces correct concrete key names."""

    def test_no_regex_keys(self, tmp_path):
        r"""Reverted state dict keys must not contain regex escapes like backslash-dot."""
        _build_native_mistral_checkpoint(tmp_path)
        model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)

        sd = model.state_dict()
        reverted = revert_weight_conversion(model, sd)

        for key in reverted:
            assert "\\" not in key, f"Reverted key {key!r} contains regex escapes"

    def test_roundtrip(self, tmp_path):
        r"""Revert then forward conversion produces identical weights."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)

        original_sd = {k: v.clone() for k, v in model.state_dict().items()}
        reverted = revert_weight_conversion(model, model.state_dict())

        # All reverted keys should be native format
        for key in reverted:
            assert not key.startswith("model."), f"Reverted key {key!r} still has HF prefix"

        # Now save with reverted keys and reload
        save_dir = tmp_path / "saved"
        save_dir.mkdir()
        save_file(reverted, str(save_dir / "model.safetensors"))
        model.config.save_pretrained(str(save_dir))
        reloaded = MistralForCausalLM.from_pretrained(str(save_dir))

        assert set(original_sd.keys()) == set(reloaded.state_dict().keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], reloaded.state_dict()[key]), f"Roundtrip mismatch for {key}"

    def test_returns_all_keys(self, tmp_path):
        r"""Reverted state dict has the same number of keys as the original."""
        _build_native_mistral_checkpoint(tmp_path)
        model = MistralForCausalLM.from_pretrained(tmp_path, mistral_format=True)

        sd = model.state_dict()
        reverted = revert_weight_conversion(model, sd)
        assert len(sd) == len(reverted)


@require_torch
class TestSavePretrained:
    r"""Verify save_pretrained save_format parameter behavior."""

    def test_default_hf_format(self, tmp_path):
        r"""Model created directly (not from native) saves as HF format by default."""
        config = _tiny_mistral_config()
        model = MistralForCausalLM(config)

        model.save_pretrained(tmp_path)
        assert (tmp_path / "config.json").exists()
        assert not (tmp_path / "params.json").exists()

    def test_default_preserves_native_format(self, tmp_path):
        r"""Model loaded from native format saves with native keys by default."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        save_dir = tmp_path / "saved"

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        model.save_pretrained(str(save_dir))

        from safetensors.torch import load_file

        saved_sd = load_file(str(save_dir / "model.safetensors"))

        # Native format keys should be present (e.g. "output.weight", not "lm_head.weight")
        assert "output.weight" in saved_sd
        assert "lm_head.weight" not in saved_sd

    def test_force_hf(self, tmp_path):
        r"""save_format='hf' saves with HF keys even if loaded from native."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        save_dir = tmp_path / "saved"

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        model.save_pretrained(str(save_dir), save_format="hf")

        from safetensors.torch import load_file

        saved_sd = load_file(str(save_dir / "model.safetensors"))
        assert "lm_head.weight" in saved_sd
        assert "output.weight" not in saved_sd

    def test_force_mistral(self, tmp_path):
        r"""save_format='mistral' saves params.json alongside native weights."""
        config = _tiny_mistral_config()
        model = MistralForCausalLM(config)

        model.save_pretrained(tmp_path, save_format="mistral")
        assert (tmp_path / "params.json").exists()
        assert (tmp_path / "consolidated.safetensors").exists()

    def test_invalid_save_format_raises(self, tmp_path):
        r"""Unknown save_format raises ValueError."""
        config = _tiny_mistral_config()
        model = MistralForCausalLM(config)

        with pytest.raises(ValueError):
            model.save_pretrained(tmp_path, save_format="invalid")


@require_torch
class TestMistralSaveLoadRoundtrip:
    r"""Full roundtrip tests: native → HF → native and HF → native → HF."""

    def test_native_to_hf_to_native(self, tmp_path):
        r"""Native → load → save HF → reload → save native → reload preserves config and weights."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        hf_dir = tmp_path / "hf"
        native2_dir = tmp_path / "native2"

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        original_config = model.config
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        model.save_pretrained(str(hf_dir), save_format="hf")
        model2 = MistralForCausalLM.from_pretrained(str(hf_dir))
        _assert_config_matches(original_config, model2.config)

        model2.save_pretrained(str(native2_dir), save_format="mistral")
        model3 = MistralForCausalLM.from_pretrained(str(native2_dir), mistral_format=True)

        _assert_config_matches(original_config, model3.config)
        assert set(original_sd.keys()) == set(model3.state_dict().keys())
        for key in original_sd:
            assert torch.equal(original_sd[key], model3.state_dict()[key]), f"Roundtrip mismatch for {key}"

    def test_hf_to_native_to_hf(self, tmp_path):
        r"""HF → save native → reload → save HF preserves config and weights."""
        config = _tiny_mistral_config()
        with torch.device("meta"):
            model = MistralForCausalLM(config)
        ref_sd = {k: torch.randn(v.shape) for k, v in model.named_parameters()}

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        native_dir = tmp_path / "native"
        hf2_dir = tmp_path / "hf2"

        # Save as HF
        save_file(ref_sd, str(hf_dir / "model.safetensors"))
        config.save_pretrained(str(hf_dir))

        # Load HF → save native
        model = MistralForCausalLM.from_pretrained(str(hf_dir))
        original_config = model.config
        model.save_pretrained(str(native_dir), save_format="mistral")

        # Load native → save HF
        model2 = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        _assert_config_matches(original_config, model2.config)
        model2.save_pretrained(str(hf2_dir), save_format="hf")

        # Reload HF
        model3 = MistralForCausalLM.from_pretrained(str(hf2_dir))

        _assert_config_matches(original_config, model3.config)
        assert set(ref_sd.keys()) == set(model3.state_dict().keys())
        for key in ref_sd:
            assert torch.equal(ref_sd[key], model3.state_dict()[key]), f"Roundtrip mismatch for {key}"

    def test_config_native_to_hf_to_native(self, tmp_path):
        r"""Config roundtrips correctly through native → HF → native."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        hf_dir = tmp_path / "hf"
        native2_dir = tmp_path / "native2"

        _build_native_mistral_checkpoint(native_dir)
        model = MistralForCausalLM.from_pretrained(str(native_dir), mistral_format=True)

        model.save_pretrained(str(hf_dir), save_format="hf")
        model2 = MistralForCausalLM.from_pretrained(str(hf_dir))

        model2.save_pretrained(str(native2_dir), save_format="mistral")

        # Verify params.json was written correctly
        with open(native2_dir / "params.json", encoding="utf-8") as f:
            params = json.load(f)
        assert params["dim"] == _TINY_HIDDEN
        assert params["n_layers"] == _TINY_LAYERS
        assert params["n_heads"] == _TINY_HEADS

    def test_preserves_model_output(self, tmp_path):
        r"""Roundtripped model produces identical forward pass output."""
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

        assert torch.equal(original_output, reloaded_output)


@require_torch
class TestMistral4SaveLoadRoundtrip:
    r"""Mistral4 roundtrip tests with MoE expert fusion."""

    def test_native_to_hf_to_native(self, tmp_path):
        r"""Mistral4 native → HF → native roundtrip preserves weights and config."""
        native_dir = tmp_path / "native"
        native_dir.mkdir()
        hf_dir = tmp_path / "hf"
        native2_dir = tmp_path / "native2"

        _build_native_mistral4_checkpoint(native_dir)
        model = Mistral4ForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        original_config = model.config
        original_sd = {k: v.clone() for k, v in model.state_dict().items()}

        model.save_pretrained(str(hf_dir), save_format="hf")
        model2 = Mistral4ForCausalLM.from_pretrained(str(hf_dir))
        _assert_config_matches(original_config, model2.config)

        model2.save_pretrained(str(native2_dir), save_format="mistral")
        model3 = Mistral4ForCausalLM.from_pretrained(str(native2_dir), mistral_format=True)

        _assert_config_matches(original_config, model3.config)
        for key in original_sd:
            assert torch.equal(original_sd[key], model3.state_dict()[key]), f"Roundtrip mismatch for {key}"

    def test_hf_to_native_to_hf(self, tmp_path):
        r"""Mistral4 HF → native → HF roundtrip preserves weights and config."""
        config = _tiny_mistral4_config()
        with torch.device("meta"):
            model = Mistral4ForCausalLM(config)
        ref_sd = {k: torch.randn(v.shape) for k, v in model.named_parameters()}

        hf_dir = tmp_path / "hf"
        hf_dir.mkdir()
        native_dir = tmp_path / "native"
        hf2_dir = tmp_path / "hf2"

        save_file(ref_sd, str(hf_dir / "model.safetensors"))
        config.save_pretrained(str(hf_dir))

        model = Mistral4ForCausalLM.from_pretrained(str(hf_dir))
        original_config = model.config
        model.save_pretrained(str(native_dir), save_format="mistral")

        model2 = Mistral4ForCausalLM.from_pretrained(str(native_dir), mistral_format=True)
        _assert_config_matches(original_config, model2.config)
        model2.save_pretrained(str(hf2_dir), save_format="hf")

        model3 = Mistral4ForCausalLM.from_pretrained(str(hf2_dir))

        _assert_config_matches(original_config, model3.config)
        for key in ref_sd:
            assert torch.equal(ref_sd[key], model3.state_dict()[key]), f"Roundtrip mismatch for {key}"


# ---------------------------------------------------------------------------
# Slow-test download and assertion helpers
# ---------------------------------------------------------------------------

# Mistral-native checkpoint files: config (``params.json``), tokenizer
# (``tekken.json``), and consolidated weights. Only these are downloaded so
# that HF-format files (``config.json``, ``model*.safetensors``) never enter
# the native directory — any such file would indicate config poisoning.
_MISTRAL_DOWNLOAD_PATTERNS = [
    "params.json",
    "tekken.json",
    "consolidated*.safetensors*",
]

# Files that ``save_pretrained`` may auto-generate beyond the core model artifacts.
_HF_OPTIONAL_FILES = {"generation_config.json"}


def _download_mistral_files(
    model_id: str,
    target_dir: Path,
    allow_patterns: list[str] | None = None,
) -> Path:
    r"""Download Mistral-native checkpoint files into `target_dir`.

    Uses `snapshot_download` with `local_dir` so that files land directly
    in `target_dir` (no nested cache layout). Only files matching
    `allow_patterns` are fetched.

    Args:
        model_id: Hub repository identifier (e.g. ``mistralai/Mistral-Small-3.2-24B-Instruct-2506``).
        target_dir: Local directory to place the downloaded files in.
        allow_patterns: Glob patterns passed to `snapshot_download`. Defaults
            to `_MISTRAL_DOWNLOAD_PATTERNS`.

    Returns:
        The `target_dir` path (same as the input, for chaining convenience).
    """
    if allow_patterns is None:
        allow_patterns = list(_MISTRAL_DOWNLOAD_PATTERNS)
    snapshot_download(repo_id=model_id, local_dir=str(target_dir), allow_patterns=allow_patterns)
    return target_dir


def _assert_dir_contains_exactly(
    directory: Path,
    expected_patterns: set[str],
    optional_patterns: set[str] | None = None,
) -> None:
    r"""Assert that `directory` contains exactly the expected files, nothing more.

    Hidden entries (e.g. `.cache/`) created by `snapshot_download` metadata
    are silently ignored. Every non-hidden file must match one of
    `expected_patterns` or `optional_patterns`; all `expected_patterns` must
    have at least one match.

    Args:
        directory: Directory to inspect (non-recursively for top-level files).
        expected_patterns: Glob patterns that *must* each match at least one file.
        optional_patterns: Glob patterns that *may* be present but not required.

    Raises:
        AssertionError: If an expected pattern has no match or an unexpected
            file is found.
    """
    if optional_patterns is None:
        optional_patterns = set()

    actual_files = {f.name for f in directory.iterdir() if f.is_file() and not f.name.startswith(".")}

    unmatched_files: set[str] = set()
    for filename in actual_files:
        is_expected = any(fnmatch.fnmatch(filename, pat) for pat in expected_patterns)
        is_optional = any(fnmatch.fnmatch(filename, pat) for pat in optional_patterns)
        if not is_expected and not is_optional:
            unmatched_files.add(filename)

    assert not unmatched_files, (
        f"Unexpected files in {directory}: {sorted(unmatched_files)}. "
        f"Expected patterns: {sorted(expected_patterns)}, optional: {sorted(optional_patterns)}"
    )

    for pattern in expected_patterns:
        has_match = any(fnmatch.fnmatch(f, pattern) for f in actual_files)
        assert has_match, f"Expected pattern {pattern!r} has no match in {directory}: {sorted(actual_files)}"


def _assert_mistral_dir(directory: Path) -> None:
    r"""Assert a directory contains only Mistral-native files and no HF artifacts.

    Verifies that ``params.json``, ``tekken.json``, and consolidated weight
    files are present, and that HF config files (``config.json``,
    ``tokenizer.json``, ``tokenizer_config.json``) and HF weight files
    (``model*.safetensors``) are absent — their presence would indicate
    data poisoning compromising the config.
    """
    expected = {"params.json", "tekken.json", "consolidated*.safetensors*"}
    _assert_dir_contains_exactly(directory, expected)

    # Verify no HF artifacts leaked in
    for forbidden in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        assert not (directory / forbidden).exists(), (
            f"HF artifact {forbidden!r} found in Mistral-native directory {directory}"
        )
    hf_weights = list(directory.glob("model*.safetensors*"))
    assert not hf_weights, (
        f"HF weight files found in Mistral-native directory {directory}: {sorted(f.name for f in hf_weights)}"
    )


def _assert_hf_dir(directory: Path, weight_patterns: set[str] | None = None) -> None:
    r"""Assert a directory contains only HF-format files and no Mistral-native artifacts.

    Verifies that ``config.json`` and HF weight files are present, and that
    native metadata (``params.json``, ``tekken.json``) and consolidated
    weight files are absent.

    Args:
        directory: Directory to verify.
        weight_patterns: Glob patterns for the weight files. Defaults to
            ``{"model*.safetensors*"}``.
    """
    if weight_patterns is None:
        weight_patterns = {"model*.safetensors*"}
    expected = {"config.json"} | weight_patterns
    # Tokenizer files are optional (only present after tokenizer.save_pretrained)
    optional = _HF_OPTIONAL_FILES | {"tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"}
    _assert_dir_contains_exactly(directory, expected, optional)

    # Verify no Mistral-native artifacts leaked in
    for forbidden in ("params.json", "tekken.json"):
        assert not (directory / forbidden).exists(), (
            f"Mistral artifact {forbidden!r} found in HF directory {directory}"
        )
    consolidated_files = list(directory.glob("consolidated*"))
    assert not consolidated_files, (
        f"Consolidated weight files found in HF directory {directory}: {sorted(f.name for f in consolidated_files)}"
    )


# ---------------------------------------------------------------------------
# Slow integration tests
# ---------------------------------------------------------------------------


@slow
@require_torch_accelerator
class TestMistralRealModelIntegration(unittest.TestCase):
    r"""Slow tests that load real Mistral models from the Hub via native format conversion."""

    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_native_format_roundtrip(self):
        r"""Load real Mistral model from native format, save as HF, reload and verify config, tokenizer, and weights.

        Downloads mistral-native files into an isolated directory, verifies
        no HF artifacts are present, then saves to a separate HF directory
        and verifies no mistral artifacts leaked.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mistral_dir = Path(tmpdir) / "mistral"
            hf_dir = Path(tmpdir) / "hf"

            # Download only the Mistral-native files
            _download_mistral_files(self.model_id, mistral_dir)
            _assert_mistral_dir(mistral_dir)

            # Load from isolated mistral directory
            model, loading_info = MistralForCausalLM.from_pretrained(
                str(mistral_dir),
                mistral_format=True,
                device_map="auto",
                torch_dtype=torch.float16,
                output_loading_info=True,
            )
            unexpected = _filter_expected_unexpected_keys(loading_info["unexpected_keys"])
            assert not unexpected, f"Unexpected keys during native load: {unexpected}"
            assert not loading_info["missing_keys"], f"Missing keys during native load: {loading_info['missing_keys']}"
            original_config = model.config
            original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            # Save as HF format into a separate directory
            model.save_pretrained(str(hf_dir), save_format="hf")
            del model
            backend_empty_cache(torch_device)
            gc.collect()

            # Verify HF dir contains only HF files (no mistral artifacts)
            _assert_hf_dir(hf_dir)

            # Verify config roundtrip
            reloaded_config = AutoConfig.from_pretrained(str(hf_dir))
            _assert_config_matches(original_config, reloaded_config)

            # Verify tokenizer roundtrip: load from mistral dir, save to hf dir
            original_tok = AutoTokenizer.from_pretrained(str(mistral_dir))
            original_tok.save_pretrained(str(hf_dir))
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))
            assert reloaded_tok.vocab_size == original_tok.vocab_size
            test_text = "Hello, world!"
            assert reloaded_tok.encode(test_text) == original_tok.encode(test_text)
            assert reloaded_tok.decode(reloaded_tok.encode(test_text), skip_special_tokens=True) == test_text

            # Verify weight roundtrip
            reloaded, reload_info = MistralForCausalLM.from_pretrained(
                str(hf_dir), device_map="auto", torch_dtype=torch.float16, output_loading_info=True
            )
            assert not reload_info["unexpected_keys"], (
                f"Unexpected keys during HF reload: {reload_info['unexpected_keys']}"
            )
            assert not reload_info["missing_keys"], f"Missing keys during HF reload: {reload_info['missing_keys']}"
            reloaded_sd = reloaded.state_dict()

            for key in original_sd:
                assert torch.equal(original_sd[key], reloaded_sd[key].cpu()), f"Roundtrip mismatch for {key}"

            del reloaded
            backend_empty_cache(torch_device)
            gc.collect()


@slow
@require_torch_accelerator
class TestMinistral3RealModelIntegration(unittest.TestCase):
    r"""Slow tests that load real Ministral3 models from the Hub via native format conversion."""

    model_id = "mistralai/Ministral-3-3B-Instruct-2512"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_native_format_roundtrip(self):
        r"""Load real Ministral3 from native format, save as HF, reload and verify config, tokenizer, and weights.

        Downloads mistral-native files into an isolated directory, verifies
        no HF artifacts are present, then saves to a separate HF directory
        and verifies no mistral artifacts leaked.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mistral_dir = Path(tmpdir) / "mistral"
            hf_dir = Path(tmpdir) / "hf"

            _download_mistral_files(self.model_id, mistral_dir)
            _assert_mistral_dir(mistral_dir)

            model, loading_info = Ministral3ForCausalLM.from_pretrained(
                str(mistral_dir),
                mistral_format=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                output_loading_info=True,
            )
            unexpected = _filter_expected_unexpected_keys(loading_info["unexpected_keys"])
            assert not unexpected, f"Unexpected keys during native load: {unexpected}"
            assert not loading_info["missing_keys"], f"Missing keys during native load: {loading_info['missing_keys']}"
            original_config = model.config
            original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            model.save_pretrained(str(hf_dir), save_format="hf")
            del model
            backend_empty_cache(torch_device)
            gc.collect()

            _assert_hf_dir(hf_dir)

            reloaded_config = AutoConfig.from_pretrained(str(hf_dir))
            _assert_config_matches(original_config, reloaded_config)

            original_tok = AutoTokenizer.from_pretrained(str(mistral_dir))
            original_tok.save_pretrained(str(hf_dir))
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))
            assert reloaded_tok.vocab_size == original_tok.vocab_size
            test_text = "Hello, world!"
            assert reloaded_tok.encode(test_text) == original_tok.encode(test_text)
            assert reloaded_tok.decode(reloaded_tok.encode(test_text), skip_special_tokens=True) == test_text

            reloaded, reload_info = Ministral3ForCausalLM.from_pretrained(
                str(hf_dir), device_map="auto", torch_dtype=torch.bfloat16, output_loading_info=True
            )
            assert not reload_info["unexpected_keys"], (
                f"Unexpected keys during HF reload: {reload_info['unexpected_keys']}"
            )
            assert not reload_info["missing_keys"], f"Missing keys during HF reload: {reload_info['missing_keys']}"
            reloaded_sd = reloaded.state_dict()

            for key in original_sd:
                assert torch.equal(original_sd[key], reloaded_sd[key].cpu()), f"Roundtrip mismatch for {key}"

            del reloaded
            backend_empty_cache(torch_device)
            gc.collect()


@slow
@require_torch_accelerator
class TestMistral3RealModelIntegration(unittest.TestCase):
    r"""Slow tests that load real Mistral3 VLM models from the Hub via native format conversion."""

    model_id = "mistralai/Mistral-Small-3.2-24B-Instruct-2506"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_native_format_roundtrip(self):
        r"""Load real Mistral3 from native format, save as HF, reload and verify config, tokenizer, and weights.

        Downloads mistral-native files into an isolated directory, verifies
        no HF artifacts are present, then saves to a separate HF directory
        and verifies no mistral artifacts leaked.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mistral_dir = Path(tmpdir) / "mistral"
            hf_dir = Path(tmpdir) / "hf"

            _download_mistral_files(self.model_id, mistral_dir)
            _assert_mistral_dir(mistral_dir)

            model, loading_info = Mistral3ForConditionalGeneration.from_pretrained(
                str(mistral_dir),
                mistral_format=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                output_loading_info=True,
            )
            unexpected = _filter_expected_unexpected_keys(loading_info["unexpected_keys"])
            assert not unexpected, f"Unexpected keys during native load: {unexpected}"
            assert not loading_info["missing_keys"], f"Missing keys during native load: {loading_info['missing_keys']}"
            original_config = model.config
            original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            model.save_pretrained(str(hf_dir), save_format="hf")
            del model
            backend_empty_cache(torch_device)
            gc.collect()

            _assert_hf_dir(hf_dir)

            reloaded_config = AutoConfig.from_pretrained(str(hf_dir))
            _assert_config_matches(original_config, reloaded_config)

            original_tok = AutoTokenizer.from_pretrained(str(mistral_dir))
            original_tok.save_pretrained(str(hf_dir))
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))
            assert reloaded_tok.vocab_size == original_tok.vocab_size
            test_text = "Hello, world!"
            assert reloaded_tok.encode(test_text) == original_tok.encode(test_text)
            assert reloaded_tok.decode(reloaded_tok.encode(test_text), skip_special_tokens=True) == test_text

            reloaded, reload_info = Mistral3ForConditionalGeneration.from_pretrained(
                str(hf_dir), device_map="auto", torch_dtype=torch.bfloat16, output_loading_info=True
            )
            assert not reload_info["unexpected_keys"], (
                f"Unexpected keys during HF reload: {reload_info['unexpected_keys']}"
            )
            assert not reload_info["missing_keys"], f"Missing keys during HF reload: {reload_info['missing_keys']}"
            reloaded_sd = reloaded.state_dict()

            for key in original_sd:
                assert torch.equal(original_sd[key], reloaded_sd[key].cpu()), f"Roundtrip mismatch for {key}"

            del reloaded
            backend_empty_cache(torch_device)
            gc.collect()


@slow
@require_torch_accelerator
class TestMistral4RealModelIntegration(unittest.TestCase):
    r"""Slow tests that load real Mistral4 models from the Hub via native format conversion."""

    model_id = "mistralai/Mistral-Small-4-119B-2603"

    def setUp(self):
        cleanup(torch_device, gc_collect=True)

    def tearDown(self):
        cleanup(torch_device, gc_collect=True)

    def test_native_format_roundtrip(self):
        r"""Load real Mistral4 from native format, save as HF, reload and verify config, tokenizer, and weights.

        Downloads mistral-native files into an isolated directory, verifies
        no HF artifacts are present, then saves to a separate HF directory
        and verifies no mistral artifacts leaked.
        """
        with tempfile.TemporaryDirectory() as tmpdir:
            mistral_dir = Path(tmpdir) / "mistral"
            hf_dir = Path(tmpdir) / "hf"

            _download_mistral_files(self.model_id, mistral_dir)
            _assert_mistral_dir(mistral_dir)

            model, loading_info = Mistral4ForCausalLM.from_pretrained(
                str(mistral_dir),
                mistral_format=True,
                device_map="auto",
                torch_dtype=torch.bfloat16,
                output_loading_info=True,
            )
            unexpected = _filter_expected_unexpected_keys(loading_info["unexpected_keys"])
            assert not unexpected, f"Unexpected keys during native load: {unexpected}"
            assert not loading_info["missing_keys"], f"Missing keys during native load: {loading_info['missing_keys']}"
            original_config = model.config
            original_sd = {k: v.clone().cpu() for k, v in model.state_dict().items()}

            model.save_pretrained(str(hf_dir), save_format="hf")
            del model
            backend_empty_cache(torch_device)
            gc.collect()

            _assert_hf_dir(hf_dir)

            reloaded_config = AutoConfig.from_pretrained(str(hf_dir))
            _assert_config_matches(original_config, reloaded_config)

            original_tok = AutoTokenizer.from_pretrained(str(mistral_dir))
            original_tok.save_pretrained(str(hf_dir))
            reloaded_tok = AutoTokenizer.from_pretrained(str(hf_dir))
            assert reloaded_tok.vocab_size == original_tok.vocab_size
            test_text = "Hello, world!"
            assert reloaded_tok.encode(test_text) == original_tok.encode(test_text)
            assert reloaded_tok.decode(reloaded_tok.encode(test_text), skip_special_tokens=True) == test_text

            reloaded, reload_info = Mistral4ForCausalLM.from_pretrained(
                str(hf_dir), device_map="auto", torch_dtype=torch.bfloat16, output_loading_info=True
            )
            assert not reload_info["unexpected_keys"], (
                f"Unexpected keys during HF reload: {reload_info['unexpected_keys']}"
            )
            assert not reload_info["missing_keys"], f"Missing keys during HF reload: {reload_info['missing_keys']}"
            reloaded_sd = reloaded.state_dict()

            for key in original_sd:
                assert torch.equal(original_sd[key], reloaded_sd[key].cpu()), f"Roundtrip mismatch for {key}"

            del reloaded
            backend_empty_cache(torch_device)
            gc.collect()
