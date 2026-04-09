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
      ``[num_experts, 1, 1]`` to match the `FP8Experts` parameter shape.
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
            while fused.ndim < 3:
                fused = fused.unsqueeze(-1)
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

    - **Per-tensor** (``shape[1]==1, shape[2]==1``): squeeze and duplicate
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
            scale_list = list(tensor.squeeze(-1).squeeze(-1).unbind(dim=0))
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


class FP8AwareMergeAndConcatenate(ConversionOps):
    r"""FP8-aware gate+up expert fusion with per-expert scale rescaling.

    Takes per-expert gate (w1) and up (w3) weight tensors, concatenates gate+up
    along dim=0 per expert, and stacks across experts along a new leading dimension.

    Input dict keys are matched by suffix so that both short names (``w1.weight``)
    and full glob-style names (``experts.*.w1.weight``) are supported.  Recognised
    suffixes for scale keys include both native names (``qscale_weight``,
    ``qscale_act``) and post-renamed names (``weight_scale_inv``,
    ``activation_scale``).

    Handles four cases:
    - BF16: simple cat + stack, no scale handling.
    - Per-tensor FP8 (scalar scales): rescale to common max scale, then cat + stack.
    - Block-wise FP8 (multi-dim scales): cat scales independently, then cat + stack weights.
    - Activation scales: per-expert max of w1/w3 scales, then stack.
    """

    # Suffix → semantic role mapping (order matters: first match wins)
    _WEIGHT_SUFFIXES: dict[str, str] = {
        "w1.weight": "w1_weight",
        "w3.weight": "w3_weight",
    }
    _SCALE_SUFFIXES: dict[str, str] = {
        "w1.qscale_weight": "w1_scale",
        "w3.qscale_weight": "w3_scale",
        "w1.weight_scale_inv": "w1_scale",
        "w3.weight_scale_inv": "w3_scale",
    }
    _ACT_SCALE_SUFFIXES: dict[str, str] = {
        "w1.qscale_act": "w1_act",
        "w3.qscale_act": "w3_act",
        "w1.activation_scale": "w1_act",
        "w3.activation_scale": "w3_act",
    }

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
        w1_key = self._find_key(input_dict, self._WEIGHT_SUFFIXES, "w1_weight")
        w3_key = self._find_key(input_dict, self._WEIGHT_SUFFIXES, "w3_weight")
        if w1_key is None or w3_key is None:
            raise ValueError(f"Cannot find w1/w3 weight keys in input_dict: {list(input_dict.keys())}")

        w1_weights = input_dict[w1_key]
        w3_weights = input_dict[w3_key]
        n_experts = len(w1_weights)

        if len(w3_weights) != n_experts:
            raise ValueError(f"Mismatched expert counts: w1 has {n_experts} experts but w3 has {len(w3_weights)}.")

        w1_scale_key = self._find_key(input_dict, self._SCALE_SUFFIXES, "w1_scale")
        w3_scale_key = self._find_key(input_dict, self._SCALE_SUFFIXES, "w3_scale")
        has_scales = w1_scale_key is not None and w3_scale_key is not None

        w1_act_key = self._find_key(input_dict, self._ACT_SCALE_SUFFIXES, "w1_act")
        w3_act_key = self._find_key(input_dict, self._ACT_SCALE_SUFFIXES, "w3_act")
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


class FP8AwareSplitAndUnstack(ConversionOps):
    r"""Reverse of `FP8AwareMergeAndConcatenate`.

    Splits a fused `gate_up_proj` tensor along dim=1 into gate and up halves,
    then unstacks along dim=0 back to per-expert tensors.

    Input dict keys are matched by suffix so that both short names
    (``gate_up_proj``) and full prefixed names work.  Output keys are
    derived from ``target_patterns`` via suffix matching so that the
    reverse-transform pipeline can expand them to concrete key names.
    """

    # Suffix → semantic role for input keys (fused HF tensors)
    _INPUT_SUFFIXES: dict[str, str] = {
        "gate_up_proj": "fused",
        "gate_up_proj_scale_inv": "scale",
        "gate_up_proj_activation_scale": "act_scale",
    }
    # Suffix → semantic role for output keys (per-expert native tensors)
    _OUTPUT_SUFFIXES: dict[str, str] = {
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
        fused_key = self._find_key(input_dict, self._INPUT_SUFFIXES, "fused")
        if fused_key is None:
            raise ValueError(f"Cannot find gate_up_proj key in input_dict: {list(input_dict.keys())}")

        fused_tensor = input_dict[fused_key]
        if isinstance(fused_tensor, list):
            fused_tensor = fused_tensor[0]

        half_dim = fused_tensor.shape[1] // 2

        w1_list = list(fused_tensor[:, :half_dim, :].unbind(dim=0))
        w3_list = list(fused_tensor[:, half_dim:, :].unbind(dim=0))

        # Resolve output key names from target_patterns when available
        w1_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w1_weight") or "w1.weight"
        w3_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w3_weight") or "w3.weight"

        result: dict[str, list[torch.Tensor]] = {
            w1_key: w1_list,
            w3_key: w3_list,
        }

        scale_key = self._find_key(input_dict, self._INPUT_SUFFIXES, "scale")
        if scale_key is not None:
            fused_scales = input_dict[scale_key]
            if isinstance(fused_scales, list):
                fused_scales = fused_scales[0]

            w1_scale_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w1_scale") or "w1.qscale_weight"
            w3_scale_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w3_scale") or "w3.qscale_weight"

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

        act_key = self._find_key(input_dict, self._INPUT_SUFFIXES, "act_scale")
        if act_key is not None:
            act_scales = input_dict[act_key]
            if isinstance(act_scales, list):
                act_scales = act_scales[0]

            w1_act_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w1_act") or "w1.qscale_act"
            w3_act_key = self._find_key(target_patterns, self._OUTPUT_SUFFIXES, "w3_act") or "w3.qscale_act"

            act_list = list(act_scales.unbind(dim=0))
            result[w1_act_key] = act_list
            result[w3_act_key] = act_list

        return result

    @property
    def reverse_op(self) -> ConversionOps:
        return FP8AwareMergeAndConcatenate()


def fp8_scale_renamings() -> list[WeightRenaming]:
    r"""Renamings for FP8 quantization scale keys."""
    return [
        WeightRenaming(r"\.qscale_weight", ".weight_scale_inv"),
        WeightRenaming(r"\.qscale_act", ".activation_scale"),
    ]


def _qk_fp8_scale_renamings(
    prefix_src: str,
    prefix_tgt: str,
) -> list[WeightRenaming]:
    r"""Generate FP8 scale renamings for Q/K attention weights.

    For Q/K weight keys, the ``.weight`` suffix is handled by a
    ``WeightConverter`` (which also applies ``PermuteForRope``).  FP8
    scale keys (``.qscale_weight`` / ``.qscale_act``) still need renaming
    from native ``attention.wq`` / ``attention.wk`` to HF
    ``self_attn.q_proj`` / ``self_attn.k_proj``, plus the suffix
    transform.  These combined renamings handle that in one step.

    Args:
        prefix_src: Anchored prefix for native keys, with a capturing
            group for the layer index (e.g.
            ``r"^layers(.*?)"``, ``r"^vision_encoder(.*?)"``).
        prefix_tgt: Replacement prefix for HF keys, using ``\\1``
            for the captured layer index (e.g.
            ``r"model.layers\\1"``, ``r"vision_tower\\1"``).
    """
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
    r"""Renamings for base Mistral text models (shared by mistral/ministral3).

    Sub-component renamings (layer norms, attention, FFN) are anchored to
    ``^layers`` via a ``\1`` capturing group so they only match layer-scoped
    keys.  The catch-all ``^layers`` → ``model.layers`` must come **after**
    the specific renamings to avoid short-circuiting them.

    Q/K ``.weight`` keys are **not** renamed here — they are handled by the
    ``WeightConverter`` (which applies both renaming and ``PermuteForRope``
    in one step, preventing double-permutation on HF-format reloads).
    Q/K FP8 scale keys (``.qscale_weight`` / ``.qscale_act``) are renamed
    via ``_qk_fp8_scale_renamings`` which combines the ``attention.wq`` →
    ``self_attn.q_proj`` mapping with the suffix transform.
    """
    return [
        WeightRenaming("^output", "lm_head"),
        WeightRenaming("^tok_embeddings", "model.embed_tokens"),
        WeightRenaming("^norm", "model.norm"),
        # Layer sub-component renamings (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)attention_norm", r"model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"model.layers\1post_attention_layernorm"),
        # Q/K FP8 scale renamings (combined prefix + suffix in one step)
        *_qk_fp8_scale_renamings(r"^layers(.*?)", r"model.layers\1"),
        # Q/K .weight is NOT renamed here — handled by WeightConverter
        WeightRenaming(r"^layers(.*?)attention\.wv", r"model.layers\1self_attn.v_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w1", r"model.layers\1mlp.gate_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w2", r"model.layers\1mlp.down_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w3", r"model.layers\1mlp.up_proj"),
        # Catch-all for remaining layers keys (including Q/K .weight)
        WeightRenaming("^layers", "model.layers"),
    ] + fp8_scale_renamings()


def mistral_base_native_converters() -> list[WeightConverter]:
    r"""Converters for base Mistral text models (Q/K RoPE permutation).

    Converter source patterns use **native** (pre-renaming) names so they
    only match native-format checkpoints.  The Q/K renamings use a negative
    lookahead to skip ``.weight`` keys, so after renamings the native Q weight
    key still contains ``attention.wq.weight`` and is matched here.  HF-format
    keys (``self_attn.q_proj.weight``) never contain ``attention.wq`` and
    therefore do not trigger the converter, preventing double-permutation on
    roundtrip reloads.

    Source patterns are anchored with ``\.weight$`` so they only match the
    weight tensor and not FP8 scale parameters.

    Note: Q uses ``num_attention_heads`` but K uses ``num_key_value_heads``
    because Mistral uses GQA (grouped-query attention).
    """
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
    r"""Renamings for Mistral3 text backbone (prefixed with `language_model.`).

    These produce keys that match the HF model structure directly:
    ``language_model.layers.{N}.self_attn.v_proj.weight`` (the ``model.``
    prefix is added by the base-model-prefix handling in ``rename_source_key``).

    Sub-component renamings (layer norms, attention, FFN) are anchored to
    ``^layers`` via a ``\1`` capturing group so they never accidentally
    match vision-encoder keys.  The catch-all ``^layers`` →
    ``language_model.layers`` must come **after** the specific renamings.

    Q/K ``.weight`` keys are **not** renamed here — they are handled by
    the ``WeightConverter`` to prevent double-permutation.  Q/K FP8 scale
    keys are handled by ``_qk_fp8_scale_renamings``.
    """

    return [
        WeightRenaming(r"^output", "lm_head"),
        WeightRenaming(r"^tok_embeddings", "language_model.embed_tokens"),
        WeightRenaming(r"^norm", "language_model.norm"),
        # Layer sub-component renamings (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)attention_norm", r"language_model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"language_model.layers\1post_attention_layernorm"),
        # Q/K FP8 scale renamings (combined prefix + suffix in one step)
        *_qk_fp8_scale_renamings(r"^layers(.*?)", r"language_model.layers\1"),
        # Q/K .weight is NOT renamed here — handled by WeightConverter
        WeightRenaming(r"^layers(.*?)attention\.wv", r"language_model.layers\1self_attn.v_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"language_model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w1", r"language_model.layers\1mlp.gate_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w2", r"language_model.layers\1mlp.down_proj"),
        WeightRenaming(r"^layers(.*?)feed_forward\.w3", r"language_model.layers\1mlp.up_proj"),
        # Catch-all for remaining layers keys (including Q/K .weight)
        WeightRenaming(r"^layers", "language_model.layers"),
    ] + fp8_scale_renamings()


def mistral3_native_text_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3 text backbone (Q/K RoPE permutation).

    Converter source patterns use **native** (pre-renaming) names so they
    only match native-format checkpoints.  After renamings (which skip
    ``.weight`` via negative lookahead), text Q/K weight keys still contain
    ``attention.wq.weight`` and are matched here.  Vision Q/K weight keys
    have already been renamed to ``attention.q_proj.weight`` by the vision
    renamings, so ``attention.wq`` only matches text keys — no ambiguity.

    HF-format keys (``self_attn.q_proj.weight``) never contain
    ``attention.wq`` and therefore do not trigger the converter, preventing
    double-permutation on roundtrip reloads.

    Source patterns are anchored with ``\.weight$`` so they only match the
    weight tensor and not FP8 scale parameters.

    Note: Q uses ``num_attention_heads`` but K uses ``num_key_value_heads``
    because the text backbone uses GQA (grouped-query attention).
    """
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
    r"""Renamings for Mistral3 vision encoder keys.

    Each renaming is anchored to ``^vision_encoder`` (or another top-level
    native prefix) so that text-backbone keys are never accidentally matched.
    Sub-component renamings (attention, feed_forward) are combined with the
    prefix in a single pattern using a ``\\1`` capturing group, avoiding
    collision with the text renamings that use the same native sub-component
    names but map to different HF names.

    Q/K ``.weight`` keys are **not** renamed here — they are handled by
    the ``WeightConverter`` to prevent double-permutation.
    """
    return [
        # Projector / adapter keys (anchored, no collision possible)
        WeightRenaming(r"^vision_language_adapter\.w_in", "multi_modal_projector.linear_1"),
        WeightRenaming(r"^vision_language_adapter\.w_out", "multi_modal_projector.linear_2"),
        WeightRenaming("^patch_merger", "multi_modal_projector.patch_merger"),
        WeightRenaming("^pre_mm_projector_norm", "multi_modal_projector.norm"),
        # Vision attention sub-components (anchored to ^vision_encoder with capturing group)
        # Q/K .weight is NOT renamed here — handled by WeightConverter
        WeightRenaming(r"^vision_encoder(.*?)attention\.wv", r"vision_tower\1attention.v_proj"),
        WeightRenaming(r"^vision_encoder(.*?)attention\.wo", r"vision_tower\1attention.o_proj"),
        # Vision feed-forward sub-components
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w1", r"vision_tower\1feed_forward.gate_proj"),
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w2", r"vision_tower\1feed_forward.down_proj"),
        WeightRenaming(r"^vision_encoder(.*?)feed_forward\.w3", r"vision_tower\1feed_forward.up_proj"),
        # Catch-all for remaining vision_encoder keys (patch_conv, ln_pre, etc.)
        WeightRenaming("^vision_encoder", "vision_tower"),
    ]


def mistral3_native_vision_converters() -> list[WeightConverter]:
    r"""Converters for Mistral3 vision Q/K with `PermuteForRope`.

    Source patterns are anchored to ``vision_tower.`` via a capturing group
    ``(vision_tower\..*\.)`` so they only match vision keys after the
    catch-all renaming ``^vision_encoder`` → ``vision_tower`` and never
    collide with text Q/K converters (which match the bare suffix
    ``attention.wq.weight$``).

    The ``\1`` back-reference in ``target_patterns`` preserves the captured
    ``vision_tower.…layers.{N}.`` prefix when producing the renamed key.

    HF-format vision keys (``attention.q_proj.weight``) never contain
    ``attention.wq`` and therefore do not trigger the converter, preventing
    double-permutation on roundtrip reloads.

    Source patterns are anchored with ``\.weight$`` to prevent accidental
    matching of scale parameters.
    """
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
    r"""Renamings for Mistral4 MLA/MoE model keys.

    Sub-component renamings (layer norms, MLA attention, router, shared
    experts) are anchored to ``^layers`` via a ``\1`` capturing group so
    they only match layer-scoped keys.  The catch-all ``^layers`` →
    ``model.layers`` must come **after** the specific renamings.
    """
    return [
        # Top-level structure
        WeightRenaming(r"^output", "lm_head"),
        WeightRenaming(r"^tok_embeddings", "model.embed_tokens"),
        WeightRenaming(r"^norm", "model.norm"),
        # Layer norms (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)attention_norm", r"model.layers\1input_layernorm"),
        WeightRenaming(r"^layers(.*?)ffn_norm", r"model.layers\1post_attention_layernorm"),
        # MLA attention keys (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)attention\.wkv_a_with_mqa", r"model.layers\1self_attn.kv_a_proj_with_mqa"),
        WeightRenaming(r"^layers(.*?)attention\.wkv_b", r"model.layers\1self_attn.kv_b_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wq_a", r"model.layers\1self_attn.q_a_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wq_b", r"model.layers\1self_attn.q_b_proj"),
        WeightRenaming(r"^layers(.*?)attention\.wo", r"model.layers\1self_attn.o_proj"),
        WeightRenaming(r"^layers(.*?)attention\.q_a_norm", r"model.layers\1self_attn.q_a_layernorm"),
        WeightRenaming(r"^layers(.*?)attention\.kv_a_norm", r"model.layers\1self_attn.kv_a_layernorm"),
        # Router (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)\.gate\.weight", r"model.layers\1.mlp.gate.weight"),
        # Shared experts (anchored to ^layers with capturing group)
        WeightRenaming(r"^layers(.*?)shared_experts\.w1", r"model.layers\1mlp.shared_experts.gate_proj"),
        WeightRenaming(r"^layers(.*?)shared_experts\.w2", r"model.layers\1mlp.shared_experts.down_proj"),
        WeightRenaming(r"^layers(.*?)shared_experts\.w3", r"model.layers\1mlp.shared_experts.up_proj"),
        # Catch-all for remaining layers keys
        WeightRenaming("^layers", "model.layers"),
    ] + fp8_scale_renamings()


def mistral4_native_converters() -> list[WeightConverter]:
    r"""Converters for Mistral4 MoE expert fusion (gate+up → gate_up_proj, down → down_proj).

    Source patterns are raw regexes with escaped dots (``\.``) so they bypass
    the glob-to-regex ``".*."`` processing.  The ``model.layers.{N}.`` prefix
    placed by renamings is preserved because the patterns only match the
    sub-component portion.

    Weight patterns use a ``$`` anchor so they do not accidentally match
    ``weight_scale_inv`` keys.  Scale and activation-scale keys have their
    own dedicated converters.
    """
    return [
        # gate+up weight fusion (w1 = gate, w3 = up)
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.weight$", r"experts.*\.w3\.weight$"],
            target_patterns="mlp.experts.gate_up_proj",
            operations=[MergeModulelist(dim=0), Concatenate(dim=1)],
        ),
        # gate+up FP8 weight_scale_inv fusion (max for per-tensor, cat for block-wise)
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.weight_scale_inv", r"experts.*\.w3\.weight_scale_inv"],
            target_patterns="mlp.experts.gate_up_proj_scale_inv",
            operations=[FP8ScaleFusionMerge()],
        ),
        # gate+up activation_scale fusion (per-expert max of w1/w3)
        WeightConverter(
            source_patterns=[r"experts.*\.w1\.activation_scale", r"experts.*\.w3\.activation_scale"],
            target_patterns="mlp.experts.gate_up_proj_activation_scale",
            operations=[MaxMergeModulelist()],
        ),
        # down_proj weight
        WeightConverter(
            source_patterns=r"experts.*\.w2\.weight$",
            target_patterns="mlp.experts.down_proj",
            operations=[MergeModulelist(dim=0)],
        ),
        # down_proj FP8 weight_scale_inv
        WeightConverter(
            source_patterns=r"experts.*\.w2\.weight_scale_inv",
            target_patterns="mlp.experts.down_proj_scale_inv",
            operations=[MergeModulelist(dim=0)],
        ),
        # down_proj activation_scale
        WeightConverter(
            source_patterns=r"experts.*\.w2\.activation_scale",
            target_patterns="mlp.experts.down_proj_activation_scale",
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


def save_native_mistral_format(
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
