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

import json
import os
from typing import Any

import torch

from ...configuration_utils import PretrainedConfig
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
from .config_format import MistralFormatConfig


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


class MaxMergeModulelist(ConversionOps):
    r"""Per-expert element-wise max of two groups, then stack across experts.

    Used for fusing activation scales and per-tensor FP8 weight scales:
    for each expert, takes the element-wise max of the gate (w1) and up
    (w3) values, then stacks the results into a single tensor along dim 0.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        keys = list(input_dict.keys())
        if len(keys) != 2:
            raise ValueError(f"MaxMergeModulelist expects exactly 2 input groups, got {len(keys)}: {keys}")

        group_a = input_dict[keys[0]]
        group_b = input_dict[keys[1]]
        n_experts = len(group_a)
        if len(group_b) != n_experts:
            raise ValueError(f"Mismatched expert counts: {len(group_a)} vs {len(group_b)}.")

        target = target_patterns[0] if len(target_patterns) == 1 else keys[0]
        fused = torch.stack([torch.max(group_a[e], group_b[e]) for e in range(n_experts)])
        return {target: fused}

    @property
    def reverse_op(self) -> ConversionOps:
        return DuplicateAndSplit()


class DuplicateAndSplit(ConversionOps):
    r"""Reverse of `MaxMergeModulelist`: duplicate the fused tensor for both targets.

    Since the forward max-merge is lossy, the reverse simply duplicates the
    fused tensor for each target pattern.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        tensor = next(iter(input_dict.values()))
        if isinstance(tensor, list):
            tensor = tensor[0]

        expert_list = list(tensor.unbind(dim=0))
        result: dict[str, list[torch.Tensor]] = {}
        for target in target_patterns:
            result[target] = [t.clone() for t in expert_list]
        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return MaxMergeModulelist()


class FP8ScaleFusionMerge(ConversionOps):
    r"""Fuse gate (w1) and up (w3) FP8 scale tensors across experts.

    Handles two cases based on the input tensor dimensionality:

    - **Per-tensor FP8** (scalar scales, ndim=0): takes the per-expert max
      of w1/w3 scales, stacks across experts, and unsqueezes to 3-D
      `[num_experts, 1, 1]` to match the `FP8Experts` parameter shape.
    - **Block-wise FP8** (multi-dim scales, ndim≥1): stacks per-expert
      scales along a new leading dim, then concatenates w1/w3 blocks
      along dim 1 (the output-feature dimension).
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        keys = list(input_dict.keys())
        if len(keys) != 2:
            raise ValueError(f"FP8ScaleFusionMerge expects exactly 2 input groups, got {len(keys)}: {keys}")

        group_a = input_dict[keys[0]]
        group_b = input_dict[keys[1]]
        n_experts = len(group_a)
        if len(group_b) != n_experts:
            raise ValueError(f"Mismatched expert counts: {len(group_a)} vs {len(group_b)}.")

        target = target_patterns[0] if len(target_patterns) == 1 else keys[0]
        is_per_tensor = group_a[0].ndim == 0

        if is_per_tensor:
            fused = torch.stack([torch.max(group_a[e], group_b[e]) for e in range(n_experts)])
            # Unsqueeze to [num_experts, 1, 1] to match FP8Experts parameter shape
            fused = fused.view(-1, 1, 1)
        else:
            fused = torch.stack(
                [torch.cat([group_a[e], group_b[e]], dim=0) for e in range(n_experts)],
                dim=0,
            )

        return {target: fused}

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8ScaleFusionSplit()


class FP8ScaleFusionSplit(ConversionOps):
    r"""Reverse of `FP8ScaleFusionMerge`.

    Splits fused gate_up scale tensors back into per-expert w1/w3 scales:

    - **Per-tensor** (`shape[1]==1, shape[2]==1`): squeeze and duplicate
      for both w1 and w3.
    - **Block-wise**: split along dim 1 into equal halves for w1 and w3.
    """

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        tensor = next(iter(input_dict.values()))
        if isinstance(tensor, list):
            tensor = tensor[0]

        if len(target_patterns) < 2:
            raise ValueError(f"FP8ScaleFusionSplit needs ≥2 target patterns, got {target_patterns}")

        is_per_tensor = tensor.ndim == 3 and tensor.shape[1] == 1 and tensor.shape[2] == 1
        if is_per_tensor:
            scale_list = list(tensor.flatten(-2, -1).unbind(dim=0))
            return {
                target_patterns[0]: scale_list,
                target_patterns[1]: [s.clone() for s in scale_list],
            }

        half = tensor.shape[1] // 2
        a_scales = list(tensor[:, :half].unbind(dim=0))
        b_scales = list(tensor[:, half:].unbind(dim=0))
        return {
            target_patterns[0]: a_scales,
            target_patterns[1]: b_scales,
        }

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8ScaleFusionMerge()


# Default suffix → semantic role mappings for FP8AwareMergeAndConcatenate.
# Order matters: first match wins.
_MERGE_WEIGHT_SUFFIXES: dict[str, str] = {
    "w1.weight": "w1_weight",
    "w3.weight": "w3_weight",
}
_MERGE_SCALE_SUFFIXES: dict[str, str] = {
    "w1.qscale_weight": "w1_scale",
    "w3.qscale_weight": "w3_scale",
    "w1.weight_scale_inv": "w1_scale",
    "w3.weight_scale_inv": "w3_scale",
}
_MERGE_ACT_SCALE_SUFFIXES: dict[str, str] = {
    "w1.qscale_act": "w1_act",
    "w3.qscale_act": "w3_act",
    "w1.activation_scale": "w1_act",
    "w3.activation_scale": "w3_act",
}


class FP8AwareMergeAndConcatenate(ConversionOps):
    r"""FP8-aware gate+up expert fusion with per-expert scale rescaling.

    Takes per-expert gate (w1) and up (w3) weight tensors, concatenates gate+up
    along dim=0 per expert, and stacks across experts along a new leading dimension.

    Input dict keys are matched by suffix so that both short names (`w1.weight`)
    and full glob-style names (`experts.*.w1.weight`) are supported.  Recognised
    suffixes for scale keys include both native names (`qscale_weight`,
    `qscale_act`) and post-renamed names (`weight_scale_inv`,
    `activation_scale`).

    Handles four cases:
    - BF16: simple cat + stack, no scale handling.
    - Per-tensor FP8 (scalar scales): rescale to common max scale, then cat + stack.
    - Block-wise FP8 (multi-dim scales): cat scales independently, then cat + stack weights.
    - Activation scales: per-expert max of w1/w3 scales, then stack.

    Attributes:
        _weight_suffixes: Suffix → role mapping for weight keys.
        _scale_suffixes: Suffix → role mapping for FP8 weight-scale keys.
        _act_scale_suffixes: Suffix → role mapping for FP8 activation-scale keys.
    """

    def __init__(
        self,
        weight_suffixes: dict[str, str] | None = None,
        scale_suffixes: dict[str, str] | None = None,
        act_scale_suffixes: dict[str, str] | None = None,
    ) -> None:
        self._weight_suffixes = weight_suffixes if weight_suffixes is not None else _MERGE_WEIGHT_SUFFIXES
        self._scale_suffixes = scale_suffixes if scale_suffixes is not None else _MERGE_SCALE_SUFFIXES
        self._act_scale_suffixes = act_scale_suffixes if act_scale_suffixes is not None else _MERGE_ACT_SCALE_SUFFIXES

    @staticmethod
    def _find_key(input_dict: dict[str, list[torch.Tensor]], suffixes: dict[str, str], role: str) -> str | None:
        r"""Find the first key in `input_dict` whose suffix maps to `role`."""
        for key in input_dict:
            for suffix, mapped_role in suffixes.items():
                if mapped_role == role and (key == suffix or key.endswith(f".{suffix}")):
                    return key
        return None

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor]],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, torch.Tensor]:
        w1_key = self._find_key(input_dict, self._weight_suffixes, "w1_weight")
        w3_key = self._find_key(input_dict, self._weight_suffixes, "w3_weight")
        if w1_key is None or w3_key is None:
            raise ValueError(f"Cannot find w1/w3 weight keys in input_dict: {list(input_dict.keys())}")

        w1_weights = input_dict[w1_key]
        w3_weights = input_dict[w3_key]
        n_experts = len(w1_weights)

        if len(w3_weights) != n_experts:
            raise ValueError(f"Mismatched expert counts: w1 has {n_experts} experts but w3 has {len(w3_weights)}.")

        w1_scale_key = self._find_key(input_dict, self._scale_suffixes, "w1_scale")
        w3_scale_key = self._find_key(input_dict, self._scale_suffixes, "w3_scale")
        has_scales = w1_scale_key is not None and w3_scale_key is not None

        w1_act_key = self._find_key(input_dict, self._act_scale_suffixes, "w1_act")
        w3_act_key = self._find_key(input_dict, self._act_scale_suffixes, "w3_act")
        has_act_scales = w1_act_key is not None and w3_act_key is not None

        if not has_scales:
            result = self._merge_bf16(w1_weights, w3_weights, n_experts)
        else:
            w1_scales = input_dict[w1_scale_key]
            w3_scales = input_dict[w3_scale_key]
            is_per_tensor = w1_scales[0].ndim == 0
            if is_per_tensor:
                result = self._merge_per_tensor_fp8(w1_weights, w3_weights, w1_scales, w3_scales, n_experts)
            else:
                result = self._merge_blockwise_fp8(w1_weights, w3_weights, w1_scales, w3_scales, n_experts)

        if has_act_scales:
            w1_act = input_dict[w1_act_key]
            w3_act = input_dict[w3_act_key]
            result["gate_up_proj_activation_scale"] = torch.stack(
                [torch.max(w1_act[e], w3_act[e]) for e in range(n_experts)]
            )

        return result

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


_SPLIT_INPUT_SUFFIXES: dict[str, str] = {
    "gate_up_proj": "fused",
    "gate_up_proj_scale_inv": "scale",
    "gate_up_proj_activation_scale": "act_scale",
}
_SPLIT_OUTPUT_SUFFIXES: dict[str, str] = {
    "w1.weight": "w1_weight",
    "w3.weight": "w3_weight",
    "w1.qscale_weight": "w1_scale",
    "w3.qscale_weight": "w3_scale",
    "w1.weight_scale_inv": "w1_scale",
    "w3.weight_scale_inv": "w3_scale",
    "w1.qscale_act": "w1_act",
    "w3.qscale_act": "w3_act",
    "w1.activation_scale": "w1_act",
    "w3.activation_scale": "w3_act",
}


class FP8AwareSplitAndUnstack(ConversionOps):
    r"""Reverse of `FP8AwareMergeAndConcatenate`.

    Splits a fused `gate_up_proj` tensor along dim=1 into gate and up halves,
    then unstacks along dim=0 back to per-expert tensors.

    Input dict keys are matched by suffix so that both short names
    (`gate_up_proj`) and full prefixed names work.  Output keys are
    derived from `target_patterns` via suffix matching so that the
    reverse-transform pipeline can expand them to concrete key names.
    """

    def __init__(
        self,
        input_suffixes: dict[str, str] | None = None,
        output_suffixes: dict[str, str] | None = None,
    ) -> None:
        self._input_suffixes = input_suffixes if input_suffixes is not None else _SPLIT_INPUT_SUFFIXES
        self._output_suffixes = output_suffixes if output_suffixes is not None else _SPLIT_OUTPUT_SUFFIXES

    @staticmethod
    def _find_key(
        keys: dict[str, Any] | list[str],
        suffixes: dict[str, str],
        role: str,
    ) -> str | None:
        r"""Find the first key whose suffix maps to `role`."""
        items = keys if isinstance(keys, list) else keys.keys()
        for key in items:
            for suffix, mapped_role in suffixes.items():
                if mapped_role == role and (key == suffix or key.endswith(f".{suffix}")):
                    return key
        return None

    @torch.no_grad()
    def convert(
        self,
        input_dict: dict[str, list[torch.Tensor] | torch.Tensor],
        source_patterns: list[str],
        target_patterns: list[str],
        **kwargs: Any,
    ) -> dict[str, list[torch.Tensor]]:
        fused_key = self._find_key(input_dict, self._input_suffixes, "fused")
        if fused_key is None:
            raise ValueError(f"Cannot find gate_up_proj key in input_dict: {list(input_dict.keys())}")

        fused_tensor = input_dict[fused_key]
        if isinstance(fused_tensor, list):
            fused_tensor = fused_tensor[0]

        half_dim = fused_tensor.shape[1] // 2

        w1_list = list(fused_tensor[:, :half_dim, :].unbind(dim=0))
        w3_list = list(fused_tensor[:, half_dim:, :].unbind(dim=0))

        # Resolve output key names from target_patterns when available
        w1_key = self._find_key(target_patterns, self._output_suffixes, "w1_weight") or "w1.weight"
        w3_key = self._find_key(target_patterns, self._output_suffixes, "w3_weight") or "w3.weight"

        result: dict[str, list[torch.Tensor]] = {
            w1_key: w1_list,
            w3_key: w3_list,
        }

        scale_key = self._find_key(input_dict, self._input_suffixes, "scale")
        if scale_key is not None:
            fused_scales = input_dict[scale_key]
            if isinstance(fused_scales, list):
                fused_scales = fused_scales[0]

            w1_scale_key = self._find_key(target_patterns, self._output_suffixes, "w1_scale") or "w1.qscale_weight"
            w3_scale_key = self._find_key(target_patterns, self._output_suffixes, "w3_scale") or "w3.qscale_weight"

            if fused_scales.ndim == 3 and fused_scales.shape[1] == 1 and fused_scales.shape[2] == 1:
                # Per-tensor scales: duplicate the same scale for w1 and w3
                scale_list = list(fused_scales.squeeze(-1).squeeze(-1).unbind(dim=0))
                result[w1_scale_key] = scale_list
                result[w3_scale_key] = scale_list
            else:
                # Block-wise scales: split along dim=1
                w1_scales = list(fused_scales[:, :half_dim].unbind(dim=0))
                w3_scales = list(fused_scales[:, half_dim:].unbind(dim=0))
                result[w1_scale_key] = w1_scales
                result[w3_scale_key] = w3_scales

        act_key = self._find_key(input_dict, self._input_suffixes, "act_scale")
        if act_key is not None:
            act_scales = input_dict[act_key]
            if isinstance(act_scales, list):
                act_scales = act_scales[0]

            w1_act_key = self._find_key(target_patterns, self._output_suffixes, "w1_act") or "w1.qscale_act"
            w3_act_key = self._find_key(target_patterns, self._output_suffixes, "w3_act") or "w3.qscale_act"

            act_list = list(act_scales.unbind(dim=0))
            result[w1_act_key] = act_list
            result[w3_act_key] = act_list

        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8AwareMergeAndConcatenate()


def _fp8_scale_renamings() -> list[WeightRenaming]:
    return [
        WeightRenaming(r"\.qscale_weight", ".weight_scale_inv"),
        WeightRenaming(r"\.qscale_act", ".activation_scale"),
    ]


def _qk_fp8_scale_renamings(
    prefix_src: str,
    prefix_tgt: str,
) -> list[WeightRenaming]:
    entries: list[WeightRenaming] = []
    for src_attn, tgt_attn in (("wq", "q_proj"), ("wk", "k_proj")):
        for src_suffix, tgt_suffix in (("qscale_weight", "weight_scale_inv"), ("qscale_act", "activation_scale")):
            entries.append(
                WeightRenaming(
                    rf"{prefix_src}attention\.{src_attn}\.{src_suffix}",
                    rf"{prefix_tgt}self_attn.{tgt_attn}.{tgt_suffix}",
                )
            )
    return entries


def mistral_base_native_renamings() -> list[WeightRenaming]:
    r"""Renamings for base Mistral text models (shared by mistral/ministral3)."""
    return [
        WeightRenaming("^output", "lm_head"),
        WeightRenaming("^tok_embeddings", "model.embed_tokens"),
        WeightRenaming("^norm", "model.norm"),
        WeightRenaming(r"^layers(.*?)attention_norm", r"model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"model.layers\1post_attention_layernorm"),
        *_qk_fp8_scale_renamings(r"^layers(.*?)", r"model.layers\1"),
        WeightRenaming(r"^layers(.*?)attention\.wv", r"model.layers\1self_attn.v_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w1", r"model.layers\1mlp.gate_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w2", r"model.layers\1mlp.down_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w3", r"model.layers\1mlp.up_proj"),
        WeightRenaming("^layers", "model.layers"),
    ] + _fp8_scale_renamings()


def mistral_base_native_converters() -> list[WeightConverter]:
    r"""Converters for base Mistral text models."""
    return [
        WeightConverter(
            source_patterns=r"attention\.wq\.weight$",
            target_patterns="self_attn.q_proj.weight",
            operations=[PermuteForRope(n_heads_attr="num_attention_heads")],
        ),
        WeightConverter(
            source_patterns=r"attention\.wk\.weight$",
            target_patterns="self_attn.k_proj.weight",
            operations=[PermuteForRope(n_heads_attr="num_key_value_heads")],
        ),
    ]


def mistral3_native_text_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral3 text backbone."""

    return [
        WeightRenaming(r"^output", "lm_head"),
        WeightRenaming(r"^tok_embeddings", "language_model.embed_tokens"),
        WeightRenaming(r"^norm", "language_model.norm"),
        WeightRenaming(r"^layers(.*?)attention_norm", r"language_model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"language_model.layers\1post_attention_layernorm"),
        *_qk_fp8_scale_renamings(r"^layers(.*?)", r"language_model.layers\1"),
        WeightRenaming(r"^layers(.*?)attention\.wv", r"language_model.layers\1self_attn.v_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"language_model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w1", r"language_model.layers\1mlp.gate_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w2", r"language_model.layers\1mlp.down_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w3", r"language_model.layers\1mlp.up_proj"),
        WeightRenaming(r"^layers", "language_model.layers"),
    ] + _fp8_scale_renamings()


def mistral3_native_text_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3 text backbone."""
    return [
        WeightConverter(
            source_patterns=r"attention\.wq\.weight$",
            target_patterns="self_attn.q_proj.weight",
            operations=[PermuteForRope(n_heads_attr="text_config.num_attention_heads")],
        ),
        WeightConverter(
            source_patterns=r"attention\.wk\.weight$",
            target_patterns="self_attn.k_proj.weight",
            operations=[PermuteForRope(n_heads_attr="text_config.num_key_value_heads")],
        ),
        WeightConverter(
            source_patterns=[r"experts.*\.w1", r"experts.*\.w3"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=r"experts.*\.w2",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


def mistral3_native_vision_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral3 vision encoder keys."""
    return [
        WeightRenaming(r"^vision_language_adapter\.w_in", "multi_modal_projector.linear_1"),
        WeightRenaming(r"^vision_language_adapter\.w_out", "multi_modal_projector.linear_2"),
        WeightRenaming("^patch_merger", "multi_modal_projector.patch_merger"),
        WeightRenaming("^pre_mm_projector_norm", "multi_modal_projector.norm"),
        WeightRenaming(r"^vision_encoder(.*?)attention\.wv", r"vision_tower\1attention.v_proj"),
        WeightRenaming(r"^vision_encoder(.*?)attention\.wo", r"vision_tower\1attention.o_proj"),
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w1", r"vision_tower\1feed_forward.gate_proj"),
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w2", r"vision_tower\1feed_forward.down_proj"),
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w3", r"vision_tower\1feed_forward.up_proj"),
        WeightRenaming("^vision_encoder", "vision_tower"),
    ]


def mistral3_native_vision_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3."""
    return [
        WeightConverter(
            source_patterns=r"(vision_tower\..*\.)attention\.wq\.weight$",
            target_patterns=r"\1attention.q_proj.weight",
            operations=[PermuteForRope(n_heads_attr="vision_config.num_attention_heads")],
        ),
        WeightConverter(
            source_patterns=r"(vision_tower\..*\.)attention\.wk\.weight$",
            target_patterns=r"\1attention.k_proj.weight",
            operations=[PermuteForRope(n_heads_attr="vision_config.num_attention_heads")],
        ),
    ]


def mistral4_native_renamings() -> list[WeightRenaming]:
    r"""Renamings for Mistral4 model keys."""
    return [
        WeightRenaming(r"^output", "lm_head"),
        WeightRenaming(r"^tok_embeddings", "model.embed_tokens"),
        WeightRenaming(r"^norm", "model.norm"),
        WeightRenaming(r"^layers(.*?)attention_norm", r"model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"model.layers\1post_attention_layernorm"),
        WeightRenaming(r"^layers(.*?)attention\.wkv_a_with_mqa", r"model.layers\1self_attn.kv_a_proj_with_mqa"),
        WeightRenaming(r"^layers(.*?)attention\.wkv_b", r"model.layers\1self_attn.kv_b_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wq_a", r"model.layers\1self_attn.q_a_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wq_b", r"model.layers\1self_attn.q_b_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)attention\.q_a_norm", r"model.layers\1self_attn.q_a_layernorm"),
        WeightRenaming(r"^layers(.*?)attention\.kv_a_norm", r"model.layers\1self_attn.kv_a_layernorm"),
        WeightRenaming(r"^layers(.*?)\.gate\.weight", r"model.layers\1.mlp.gate.weight"),
        WeightRenaming(r"^layers(.*?)shared_experts\.w1", r"model.layers\1mlp.shared_experts.gate_proj"),
        WeightRenaming(r"^layers(.*?)shared_experts\.w2", r"model.layers\1mlp.shared_experts.down_proj"),
        WeightRenaming(r"^layers(.*?)shared_experts\.w3", r"model.layers\1mlp.shared_experts.up_proj"),
        WeightRenaming(r"^layers(.*?experts)", r"model.layers\1"),
    ] + _fp8_scale_renamings()


def mistral4_native_converters() -> list[WeightConverter]:
    r"""Converters for Mistral4."""
    return [
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.weight$", r"experts.*\.w3\.weight$"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.weight_scale_inv", r"experts.*\.w3\.weight_scale_inv"],
            target_patterns="mlp.experts.gate_up_proj_scale_inv",
            operations=[FP8ScaleFusionMerge()],
        ),
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.activation_scale", r"experts.*\.w3\.activation_scale"],
            target_patterns="mlp.experts.gate_up_proj_activation_scale",
            operations=[MaxMergeModulelist()],
        ),
        WeightConverter(
            source_patterns=r"experts.*\.w2\.weight$",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns=r"experts.*\.w2\.weight_scale_inv",
            target_patterns="mlp.experts.down_proj_scale_inv",
            operations=[MergeModulelist(dim=0)],
        ),
        WeightConverter(
            source_patterns=r"experts.*\.w2\.activation_scale",
            target_patterns="mlp.experts.down_proj_activation_scale",
            operations=[MergeModulelist(dim=0)],
        ),
    ]


register_many_to_many_conversion(FP8AwareMergeAndConcatenate)
register_many_to_many_conversion(FP8AwareSplitAndUnstack)


def _add_variant(weights_name: str, variant: str | None = None) -> str:
    if variant is not None:
        path, name = weights_name.rsplit(".", 1)
        weights_name = f"{path}.{variant}.{name}"
    return weights_name


def save_native_mistral_format(
    save_directory: str | os.PathLike,
    config: PretrainedConfig | MistralFormatConfig,
    index: dict | None,
    variant: str | None,
) -> None:
    r"""Rename HF weight files to native Mistral names and write `params.json`."""
    save_directory = str(save_directory)

    hf_single = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_NAME, variant))
    consolidated_single = os.path.join(save_directory, "consolidated.safetensors")
    # single shard
    if os.path.isfile(hf_single):
        os.rename(hf_single, consolidated_single)

    # multiple shards
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

        hf_index_name = os.path.join(save_directory, _add_variant(SAFE_WEIGHTS_INDEX_NAME, variant))
        consolidated_index = os.path.join(save_directory, "consolidated.safetensors.index.json")
        if os.path.isfile(hf_index_name):
            os.remove(hf_index_name)
        with open(consolidated_index, "w", encoding="utf-8") as f:
            content = json.dumps(index, indent=2, sort_keys=True, ensure_ascii=False) + "\n"
            f.write(content)

    if isinstance(config, MistralFormatConfig):
        params = config._config_to_params_json()
        with open(os.path.join(save_directory, "params.json"), "w", encoding="utf-8") as f:
            json.dump(params, f, indent=2, ensure_ascii=False)
