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
r"""Weight key conversion between Mistral native and HuggingFace formats.

Provides factory functions returning lists of `WeightRenaming` and `WeightConverter`
entries for each model type, plus `FP8AwareMergeAndConcatenate` for expert fusion.
"""

from __future__ import annotations

import json
import os
from typing import Any

import torch

from ...core_model_loading import (
    Concatenate,
    ConversionOps,
    MergeModulelist,
    PermuteForRope,
    WeightConverter,
    WeightRenaming,
    register_many_to_many_conversion,
)
from ...utils import SAFE_WEIGHTS_INDEX_NAME, SAFE_WEIGHTS_NAME


_FP8_DTYPE = torch.float8_e4m3fn
_FP8_MIN = torch.finfo(_FP8_DTYPE).min
_FP8_MAX = torch.finfo(_FP8_DTYPE).max


def _rescale_fp8(
    tensor: torch.Tensor,
    original_scale_inv: torch.Tensor,
    target_scale_inv: torch.Tensor,
) -> torch.Tensor:
    r"""Rescale an FP8 tensor from `original_scale_inv` to `target_scale_inv`."""
    ratio = original_scale_inv / target_scale_inv
    return (tensor.to(torch.bfloat16) * ratio).clamp(min=_FP8_MIN, max=_FP8_MAX).to(_FP8_DTYPE)


class FP8AwareMergeAndConcatenate(ConversionOps):
    r"""FP8-aware gate+up expert fusion with per-expert scale rescaling.

    Takes per-expert gate (w1) and up (w3) weight tensors, concatenates gate+up
    along dim=0 per expert, and stacks across experts along a new leading dimension.

    Handles three cases:
    - BF16: simple cat + stack, no scale handling.
    - Per-tensor FP8 (scalar scales): rescale to common max scale, then cat + stack.
    - Block-wise FP8 (multi-dim scales): cat scales independently, then cat + stack weights.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        w1_weights = input_dict["w1.weight"]
        w3_weights = input_dict["w3.weight"]
        n_experts = len(w1_weights)

        if len(w3_weights) != n_experts:
            raise ValueError(f"Mismatched expert counts: w1 has {n_experts} experts but w3 has {len(w3_weights)}.")

        has_scales = "w1.qscale_weight" in input_dict and "w3.qscale_weight" in input_dict

        if not has_scales:
            return self._merge_bf16(w1_weights, w3_weights, n_experts)

        w1_scales = input_dict["w1.qscale_weight"]
        w3_scales = input_dict["w3.qscale_weight"]

        is_per_tensor = w1_scales[0].ndim == 0
        if is_per_tensor:
            return self._merge_per_tensor_fp8(w1_weights, w3_weights, w1_scales, w3_scales, n_experts)

        return self._merge_blockwise_fp8(w1_weights, w3_weights, w1_scales, w3_scales, n_experts)

    def _merge_bf16(
        self,
        w1_weights: list[torch.Tensor],
        w3_weights: list[torch.Tensor],
        n_experts: int,
    ) -> dict[str, torch.Tensor]:
        gate_up_list = [torch.cat([w1_weights[e], w3_weights[e]], dim=0) for e in range(n_experts)]
        return {"gate_up_proj": torch.stack(gate_up_list, dim=0)}

    def _merge_per_tensor_fp8(
        self,
        w1_weights: list[torch.Tensor],
        w3_weights: list[torch.Tensor],
        w1_scales: list[torch.Tensor],
        w3_scales: list[torch.Tensor],
        n_experts: int,
    ) -> dict[str, torch.Tensor]:
        gate_up_list: list[torch.Tensor] = []
        scale_inv_list: list[torch.Tensor] = []

        for e in range(n_experts):
            fused_scale_inv = torch.max(w1_scales[e], w3_scales[e])
            gate = _rescale_fp8(w1_weights[e], w1_scales[e], fused_scale_inv)
            up = _rescale_fp8(w3_weights[e], w3_scales[e], fused_scale_inv)
            gate_up_list.append(torch.cat([gate, up], dim=0))
            scale_inv_list.append(fused_scale_inv)

        gate_up_proj_scale_inv = torch.stack(scale_inv_list)
        while gate_up_proj_scale_inv.ndim < 3:
            gate_up_proj_scale_inv = gate_up_proj_scale_inv.unsqueeze(-1)

        return {
            "gate_up_proj": torch.stack(gate_up_list, dim=0),
            "gate_up_proj_scale_inv": gate_up_proj_scale_inv,
        }

    def _merge_blockwise_fp8(
        self,
        w1_weights: list[torch.Tensor],
        w3_weights: list[torch.Tensor],
        w1_scales: list[torch.Tensor],
        w3_scales: list[torch.Tensor],
        n_experts: int,
    ) -> dict[str, torch.Tensor]:
        gate_up_list = [torch.cat([w1_weights[e], w3_weights[e]], dim=0) for e in range(n_experts)]
        scale_list = [torch.cat([w1_scales[e], w3_scales[e]], dim=0) for e in range(n_experts)]

        return {
            "gate_up_proj": torch.stack(gate_up_list, dim=0),
            "gate_up_proj_scale_inv": torch.stack(scale_list, dim=0),
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8AwareSplitAndUnstack()


class FP8AwareSplitAndUnstack(ConversionOps):
    r"""Reverse of `FP8AwareMergeAndConcatenate`.

    Splits a fused `gate_up_proj` tensor along dim=1 into gate and up halves,
    then unstacks along dim=0 back to per-expert tensors.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        fused_tensor = input_dict["gate_up_proj"]
        if isinstance(fused_tensor, list):
            fused_tensor = fused_tensor[0]

        half_dim = fused_tensor.shape[1] // 2

        w1_list = list(fused_tensor[:, :half_dim, :].unbind(dim=0))
        w3_list = list(fused_tensor[:, half_dim:, :].unbind(dim=0))

        result: dict[str, list[torch.Tensor]] = {
            "w1.weight": w1_list,
            "w3.weight": w3_list,
        }

        if "gate_up_proj_scale_inv" in input_dict:
            fused_scales = input_dict["gate_up_proj_scale_inv"]
            if isinstance(fused_scales, list):
                fused_scales = fused_scales[0]
            if fused_scales.ndim == 3 and fused_scales.shape[1] == 1 and fused_scales.shape[2] == 1:
                # Per-tensor scales: just duplicate the same scale for w1 and w3
                scale_list = list(fused_scales.squeeze(-1).squeeze(-1).unbind(dim=0))
                result["w1.qscale_weight"] = scale_list
                result["w3.qscale_weight"] = scale_list
            else:
                # Block-wise scales: split along dim=1
                w1_scales = list(fused_scales[:, :half_dim].unbind(dim=0))
                w3_scales = list(fused_scales[:, half_dim:].unbind(dim=0))
                result["w1.qscale_weight"] = w1_scales
                result["w3.qscale_weight"] = w3_scales

        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8AwareMergeAndConcatenate()


def mistral_base_native_renamings() -> list[WeightRenaming]:
    r"""Renamings for base Mistral text models (shared by mistral/ministral3)."""
    return [
        WeightRenaming("^output", "lm_head"),
        WeightRenaming("^tok_embeddings", "model.embed_tokens"),
        WeightRenaming("^norm", "model.norm"),
        WeightRenaming("^layers", "model.layers"),
        WeightRenaming("attention_norm", "input_layernorm"),
        WeightRenaming("ffn_norm", "post_attention_layernorm"),
        WeightRenaming(r"attention\.wv", "self_attn.v_proj"),
        WeightRenaming(r"attention\.wo", "self_attn.o_proj"),
        WeightRenaming(r"feed_forward\.w1", "mlp.gate_proj"),
        WeightRenaming(r"feed_forward\.w2", "mlp.down_proj"),
        WeightRenaming(r"feed_forward\.w3", "mlp.up_proj"),
    ]


def mistral_base_native_converters() -> list[WeightConverter]:
    r"""Converters for base Mistral text models (Q/K RoPE permutation)."""
    return [
        WeightConverter(
            source_patterns=r"attention\.wq",
            target_patterns="self_attn.q_proj",
            operations=[PermuteForRope()],
        ),
        WeightConverter(
            source_patterns=r"attention\.wk",
            target_patterns="self_attn.k_proj",
            operations=[PermuteForRope(n_heads_attr="num_key_value_heads")],
        ),
    ]


def fp8_scale_renamings() -> list[WeightRenaming]:
    r"""Renamings for FP8 quantization scale keys."""
    return [
        WeightRenaming(r"\.qscale_weight", ".weight_scale_inv"),
        WeightRenaming(r"\.qscale_act", ".activation_scale"),
    ]


def mistral3_native_text_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral3 text backbone (prefixed with `language_model.`)."""
    return [
        WeightRenaming("^output", "language_model.lm_head"),
        WeightRenaming("^tok_embeddings", "language_model.model.embed_tokens"),
        WeightRenaming("^norm", "language_model.model.norm"),
        WeightRenaming("^layers", "language_model.model.layers"),
        WeightRenaming("attention_norm", "input_layernorm"),
        WeightRenaming("ffn_norm", "post_attention_layernorm"),
        WeightRenaming(r"attention\.wv", "self_attn.v_proj"),
        WeightRenaming(r"attention\.wo", "self_attn.o_proj"),
        WeightRenaming(r"feed_forward\.w1", "mlp.gate_proj"),
        WeightRenaming(r"feed_forward\.w2", "mlp.down_proj"),
        WeightRenaming(r"feed_forward\.w3", "mlp.up_proj"),
    ]


def mistral3_native_text_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3 text backbone (Q/K RoPE permutation)."""
    return [
        WeightConverter(
            source_patterns=r"attention\.wq",
            target_patterns="self_attn.q_proj",
            operations=[PermuteForRope()],
        ),
        WeightConverter(
            source_patterns=r"attention\.wk",
            target_patterns="self_attn.k_proj",
            operations=[PermuteForRope(n_heads_attr="num_key_value_heads")],
        ),
    ]


def mistral3_native_vision_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral3 vision encoder keys."""
    return [
        WeightRenaming("^vision_encoder", "vision_tower"),
        WeightRenaming(r"^vision_language_adapter\.w_in", "multi_modal_projector.linear_1"),
        WeightRenaming(r"^vision_language_adapter\.w_out", "multi_modal_projector.linear_2"),
        WeightRenaming("^patch_merger", "multi_modal_projector.patch_merger"),
        WeightRenaming("^pre_mm_projector_norm", "multi_modal_projector.norm"),
        WeightRenaming(r"attention\.wv\.", "attention.v_proj."),
        WeightRenaming(r"attention\.wo\.", "attention.o_proj."),
        WeightRenaming(r"feed_forward\.w1", "feed_forward.gate_proj"),
        WeightRenaming(r"feed_forward\.w2", "feed_forward.down_proj"),
        WeightRenaming(r"feed_forward\.w3", "feed_forward.up_proj"),
    ]


def mistral3_native_vision_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3 vision Q/K with `PermuteForRope` using dotted attr."""
    return [
        WeightConverter(
            source_patterns=r"attention\.wq\.",
            target_patterns="attention.q_proj.",
            operations=[PermuteForRope(n_heads_attr="vision_config.num_attention_heads")],
        ),
        WeightConverter(
            source_patterns=r"attention\.wk\.",
            target_patterns="attention.k_proj.",
            operations=[PermuteForRope(n_heads_attr="vision_config.num_attention_heads")],
        ),
    ]


def mistral4_native_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral4 MLA/MoE model keys."""
    return [
        # Top-level structure
        WeightRenaming("^output", "lm_head"),
        WeightRenaming("^tok_embeddings", "model.embed_tokens"),
        WeightRenaming("^norm", "model.norm"),
        WeightRenaming("^layers", "model.layers"),
        # Layer norms
        WeightRenaming("attention_norm", "input_layernorm"),
        WeightRenaming("ffn_norm", "post_attention_layernorm"),
        # MLA attention keys
        WeightRenaming(r"attention\.wkv_a_with_mqa", "self_attn.kv_a_proj_with_mqa"),
        WeightRenaming(r"attention\.wkv_b", "self_attn.kv_b_proj"),
        WeightRenaming(r"attention\.wq_a", "self_attn.q_a_proj"),
        WeightRenaming(r"attention\.wq_b", "self_attn.q_b_proj"),
        WeightRenaming(r"attention\.wo", "self_attn.o_proj"),
        WeightRenaming(r"attention\.q_a_norm", "self_attn.q_a_layernorm"),
        WeightRenaming(r"attention\.kv_a_norm", "self_attn.kv_a_layernorm"),
        # Router
        WeightRenaming(r"\.gate\.weight", ".mlp.gate.weight"),
        # Shared experts
        WeightRenaming(r"shared_experts\.w1", "mlp.shared_experts.gate_proj"),
        WeightRenaming(r"shared_experts\.w2", "mlp.shared_experts.down_proj"),
        WeightRenaming(r"shared_experts\.w3", "mlp.shared_experts.up_proj"),
    ]


def mistral4_native_converters() -> list[WeightConverter]:
    r"""Converters for Mistral4 MoE expert fusion (gate+up → gate_up_proj, down → down_proj)."""
    return [
        WeightConverter(
            source_patterns=[r"experts\..*\.w1", r"experts\..*\.w3"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=r"experts\..*\.w2",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


register_many_to_many_conversion(FP8AwareMergeAndConcatenate)
register_many_to_many_conversion(FP8AwareSplitAndUnstack)


def _add_variant(weights_name: str, variant: str | None = None) -> str:
    r"""Insert a variant suffix into `weights_name` (e.g. ``model.fp16.safetensors``)."""
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def _save_native_mistral_format(
    save_directory: str | os.PathLike,
    config,
    index: dict | None,
    variant: str | None,
) -> None:
    r"""Rename HF weight files to native Mistral names and write ``params.json``."""
    save_directory = str(save_directory)

    # Rename model.safetensors → consolidated.safetensors (single shard)
    hf_single = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_NAME, variant))
    consolidated_single = os.path.join(save_directory, "consolidated.safetensors")
    if os.path.isfile(hf_single):
        os.rename(hf_single, consolidated_single)

    # Rename sharded files: model-00001-of-00005.safetensors → consolidated-00001-of-00005.safetensors
    if index is not None:
        new_weight_map = {}
        for param_name, shard_file in index["weight_map"].items():
            new_shard = shard_file.replace("model", "consolidated")
            new_weight_map[param_name] = new_shard
            src = os.path.join(save_directory, shard_file)
            dst = os.path.join(save_directory, new_shard)
            if os.path.isfile(src) and src != dst:
                os.rename(src, dst)
        index["weight_map"] = new_weight_map

        # Rewrite index file
        hf_index_name = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
        consolidated_index = os.path.join(save_directory, "consolidated.safetensors.index.json")
        if os.path.isfile(hf_index_name):
            os.remove(hf_index_name)
        with open(consolidated_index, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(content)

    # Write params.json if the config supports it
    if hasattr(config, "_config_to_params_json"):
        params = config._config_to_params_json()
        with open(os.path.join(save_directory, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
