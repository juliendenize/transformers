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

from transformers.testing_utils import require_torch
from transformers.utils import is_torch_available


if is_torch_available():
    import torch

    from transformers.core_model_loading import PermuteForRope, WeightConverter, WeightRenaming
    from transformers.integrations.mistral.weight_conversion import (
        FP8AwareMergeAndConcatenate,
        FP8AwareSplitAndUnstack,
        FP8ScaleFusionMerge,
        FP8ScaleFusionSplit,
        _fp8_scale_renamings,
        mistral3_native_text_renamings,
        mistral3_native_vision_converters,
        mistral3_native_vision_renamings,
        mistral4_native_renamings,
        mistral_base_native_converters,
        mistral_base_native_renamings,
    )


def _source_target_pairs(
    entries: list,
) -> list[tuple[str, str]]:
    pairs = []
    for entry in entries:
        for src in entry.source_patterns:
            for tgt in entry.target_patterns:
                pairs.append((src, tgt))
    return pairs


def _apply_renamings(key: str, renamings: list) -> str:
    for renaming in renamings:
        key, _ = renaming.rename_source_key(key)
    return key


@require_torch
class TestMistralBaseRenamings:
    def test_renamings(self):
        renamings = mistral_base_native_renamings()
        native_keys = [
            "output.weight",
            "tok_embeddings.weight",
            "norm.weight",
            "layers.0.attention_norm.weight",
            "layers.0.ffn_norm.weight",
            "layers.0.attention.wv.weight",
            "layers.0.attention.wo.weight",
            "layers.0.feed_forward.w1.weight",
            "layers.0.feed_forward.w2.weight",
            "layers.0.feed_forward.w3.weight",
        ]
        expected_hf_keys = [
            "lm_head.weight",
            "model.embed_tokens.weight",
            "model.norm.weight",
            "model.layers.0.input_layernorm.weight",
            "model.layers.0.post_attention_layernorm.weight",
            "model.layers.0.self_attn.v_proj.weight",
            "model.layers.0.self_attn.o_proj.weight",
            "model.layers.0.mlp.gate_proj.weight",
            "model.layers.0.mlp.down_proj.weight",
            "model.layers.0.mlp.up_proj.weight",
        ]
        for native_key, expected in zip(native_keys, expected_hf_keys):
            result = _apply_renamings(native_key, renamings)
            assert result == expected, f"Renaming failed for {native_key!r}"

    def test_rope_permutation_applied(self):
        converters = mistral_base_native_converters()
        n_heads, head_dim, hidden = 2, 4, 8
        q_dim = n_heads * head_dim

        class FakeConfig:
            num_attention_heads = n_heads
            num_key_value_heads = n_heads

        original = torch.arange(q_dim * hidden, dtype=torch.float32).reshape(q_dim, hidden)

        q_conv, k_conv = converters
        q_result = q_conv.operations[0].convert(
            input_dict={"attention.wq": [original.clone()]},
            source_patterns=["attention.wq"],
            target_patterns=["self_attn.q_proj"],
            config=FakeConfig(),
        )
        permuted = q_result["self_attn.q_proj"][0]
        assert permuted.shape == original.shape
        assert not torch.equal(permuted, original)

        inv_result = q_conv.operations[0].reverse_op.convert(
            input_dict={"self_attn.q_proj": [permuted]},
            source_patterns=["self_attn.q_proj"],
            target_patterns=["attention.wq"],
            config=FakeConfig(),
        )["attention.wq"][0]
        torch.testing.assert_close(inv_result, original)

        # K converter uses "num_key_value_heads"
        k_result = k_conv.operations[0].convert(
            input_dict={"attention.wk": [original.clone()]},
            source_patterns=["attention.wk"],
            target_patterns=["self_attn.k_proj"],
            config=FakeConfig(),
        )
        k_permuted = k_result["self_attn.k_proj"][0]
        assert not torch.equal(k_permuted, original)


@require_torch
class TestFP8ScaleRenamings:
    def test_renamings(self):
        renamings = _fp8_scale_renamings()
        assert isinstance(renamings, list)
        assert all(isinstance(r, WeightRenaming) for r in renamings)
        pairs = _source_target_pairs(renamings)
        sources = {src for src, _ in pairs}
        targets = {tgt for _, tgt in pairs}
        assert any("qscale_weight" in s for s in sources)
        assert any("qscale_act" in s for s in sources)
        assert any("weight_scale_inv" in t for t in targets)
        assert any("activation_scale" in t for t in targets)


@require_torch
class TestMistral3Renamings:
    def test_text_renamings_prefixed(self):
        renamings = mistral3_native_text_renamings()
        result = _apply_renamings("output.weight", renamings)
        assert result == "lm_head.weight"
        result = _apply_renamings("tok_embeddings.weight", renamings)
        assert result == "model.language_model.embed_tokens.weight"
        result = _apply_renamings("layers.0.attention.wo.weight", renamings)
        assert result == "model.language_model.layers.0.self_attn.o_proj.weight"

    def test_vision_renamings(self):
        renamings = mistral3_native_vision_renamings()
        test_cases = [
            (
                "vision_encoder.transformer.layers.0.attention.wv.weight",
                "model.vision_tower.transformer.layers.0.attention.v_proj.weight",
            ),
            ("vision_encoder.ln_pre.weight", "model.vision_tower.ln_pre.weight"),
            ("vision_encoder.patch_conv.weight", "model.vision_tower.patch_conv.weight"),
            ("vision_language_adapter.w_in.weight", "model.multi_modal_projector.linear_1.weight"),
            ("vision_language_adapter.w_out.weight", "model.multi_modal_projector.linear_2.weight"),
            ("patch_merger.merging_layer.weight", "model.multi_modal_projector.patch_merger.merging_layer.weight"),
            ("pre_mm_projector_norm.weight", "model.multi_modal_projector.norm.weight"),
        ]
        for native_key, expected in test_cases:
            result = _apply_renamings(native_key, renamings)
            assert result == expected, f"Vision renaming failed for {native_key!r}"

    def test_vision_converters_dotted_heads(self):
        converters = mistral3_native_vision_converters()
        assert isinstance(converters, list)
        assert all(isinstance(c, WeightConverter) for c in converters)

        assert len(converters) == 2
        for converter in converters:
            assert any(isinstance(op, PermuteForRope) for op in converter.operations)
            for op in converter.operations:
                if isinstance(op, PermuteForRope):
                    assert op.n_heads_attr == "vision_config.num_attention_heads"

    def test_hf_keys_pass_through(self):
        renamings = mistral3_native_text_renamings()
        hf_key = "language_model.model.layers.0.self_attn.q_proj.weight"
        result = _apply_renamings(hf_key, renamings)
        assert isinstance(result, str)


@require_torch
class TestMistral4Renamings:
    def test_renamings(self):
        renamings = mistral4_native_renamings()
        test_cases = [
            ("output.weight", "lm_head.weight"),
            ("tok_embeddings.weight", "model.embed_tokens.weight"),
            ("norm.weight", "model.norm.weight"),
            ("layers.0.attention_norm.weight", "model.layers.0.input_layernorm.weight"),
            ("layers.0.ffn_norm.weight", "model.layers.0.post_attention_layernorm.weight"),
            ("layers.0.attention.wkv_a_with_mqa.weight", "model.layers.0.self_attn.kv_a_proj_with_mqa.weight"),
            ("layers.0.attention.wq_a.weight", "model.layers.0.self_attn.q_a_proj.weight"),
            ("layers.0.attention.wq_b.weight", "model.layers.0.self_attn.q_b_proj.weight"),
            ("layers.0.attention.wkv_b.weight", "model.layers.0.self_attn.kv_b_proj.weight"),
            ("layers.0.attention.wo.weight", "model.layers.0.self_attn.o_proj.weight"),
            ("layers.0.attention.q_a_norm.weight", "model.layers.0.self_attn.q_a_layernorm.weight"),
            ("layers.0.attention.kv_a_norm.weight", "model.layers.0.self_attn.kv_a_layernorm.weight"),
            ("layers.0.gate.weight", "model.layers.0.mlp.gate.weight"),
            ("layers.0.shared_experts.w1.weight", "model.layers.0.mlp.shared_experts.gate_proj.weight"),
            ("layers.0.shared_experts.w2.weight", "model.layers.0.mlp.shared_experts.down_proj.weight"),
            ("layers.0.shared_experts.w3.weight", "model.layers.0.mlp.shared_experts.up_proj.weight"),
        ]
        for native_key, expected in test_cases:
            result = _apply_renamings(native_key, renamings)
            assert result == expected, f"Renaming failed for {native_key!r}"


@require_torch
class TestFP8AwareMergeAndConcatenate:
    def test_merge_bf16(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim) for _ in range(n_experts)]
        result = op.convert(
            input_dict={"w1.weight": w1, "w3.weight": w3},
            source_patterns=["w1.weight", "w3.weight"],
            target_patterns=["gate_up_proj"],
        )
        assert "gate_up_proj" in result
        fused = result["gate_up_proj"]
        assert fused.shape == (n_experts, gate_dim + up_dim, in_dim)

    def test_merge_per_tensor_fp8(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w1_scales = [torch.tensor(0.5) for _ in range(n_experts)]
        w3_scales = [torch.tensor(0.3) for _ in range(n_experts)]
        result = op.convert(
            input_dict={
                "w1.weight": w1,
                "w3.weight": w3,
                "w1.qscale_weight": w1_scales,
                "w3.qscale_weight": w3_scales,
            },
            source_patterns=["w1.weight", "w3.weight", "w1.qscale_weight", "w3.qscale_weight"],
            target_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
        )
        assert "gate_up_proj" in result
        assert "gate_up_proj_scale_inv" in result
        assert result["gate_up_proj"].shape == (n_experts, gate_dim + up_dim, in_dim)
        assert result["gate_up_proj"].dtype == torch.float8_e4m3fn

    def test_merge_blockwise_fp8(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 2
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w1_scales = [torch.randn(gate_dim, 1) for _ in range(n_experts)]
        w3_scales = [torch.randn(up_dim, 1) for _ in range(n_experts)]
        result = op.convert(
            input_dict={
                "w1.weight": w1,
                "w3.weight": w3,
                "w1.qscale_weight": w1_scales,
                "w3.qscale_weight": w3_scales,
            },
            source_patterns=["w1.weight", "w3.weight", "w1.qscale_weight", "w3.qscale_weight"],
            target_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
        )
        assert "gate_up_proj" in result
        assert "gate_up_proj_scale_inv" in result
        # Block-wise: scales are concatenated independently
        assert result["gate_up_proj_scale_inv"].shape[0] == n_experts
        assert result["gate_up_proj_scale_inv"].shape[1] == gate_dim + up_dim

    def test_mismatched_expert_count_raises(self):
        op = FP8AwareMergeAndConcatenate()
        w1 = [torch.randn(16, 32) for _ in range(4)]
        w3 = [torch.randn(16, 32) for _ in range(3)]  # mismatched!
        with pytest.raises((ValueError, RuntimeError)):
            op.convert(
                input_dict={"w1.weight": w1, "w3.weight": w3},
                source_patterns=["w1.weight", "w3.weight"],
                target_patterns=["gate_up_proj"],
            )

    def test_single_expert(self):
        op = FP8AwareMergeAndConcatenate()
        w1 = [torch.randn(16, 32)]
        w3 = [torch.randn(16, 32)]
        result = op.convert(
            input_dict={"w1.weight": w1, "w3.weight": w3},
            source_patterns=["w1.weight", "w3.weight"],
            target_patterns=["gate_up_proj"],
        )
        assert "gate_up_proj" in result
        assert result["gate_up_proj"].shape == (1, 32, 32)

    def test_merge_bf16_glob_keys(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim) for _ in range(n_experts)]
        result = op.convert(
            input_dict={"experts.*.w1.weight": w1, "experts.*.w3.weight": w3},
            source_patterns=["experts.*.w1.weight", "experts.*.w3.weight"],
            target_patterns=["gate_up_proj"],
        )
        assert "gate_up_proj" in result
        assert result["gate_up_proj"].shape == (n_experts, gate_dim + up_dim, in_dim)

    def test_merge_per_tensor_fp8_renamed_keys(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 2
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w1_scales = [torch.tensor(0.5) for _ in range(n_experts)]
        w3_scales = [torch.tensor(0.3) for _ in range(n_experts)]
        result = op.convert(
            input_dict={
                "experts.*.w1.weight": w1,
                "experts.*.w3.weight": w3,
                "experts.*.w1.weight_scale_inv": w1_scales,
                "experts.*.w3.weight_scale_inv": w3_scales,
            },
            source_patterns=[
                "experts.*.w1.weight",
                "experts.*.w3.weight",
                "experts.*.w1.weight_scale_inv",
                "experts.*.w3.weight_scale_inv",
            ],
            target_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
        )
        assert "gate_up_proj" in result
        assert "gate_up_proj_scale_inv" in result
        assert result["gate_up_proj"].dtype == torch.float8_e4m3fn

    def test_merge_with_activation_scales(self):
        op = FP8AwareMergeAndConcatenate()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1 = [torch.randn(gate_dim, in_dim) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim) for _ in range(n_experts)]
        w1_act = [torch.tensor(float(i)) for i in range(n_experts)]
        w3_act = [torch.tensor(float(i + 1)) for i in range(n_experts)]
        result = op.convert(
            input_dict={
                "experts.*.w1.weight": w1,
                "experts.*.w3.weight": w3,
                "experts.*.w1.activation_scale": w1_act,
                "experts.*.w3.activation_scale": w3_act,
            },
            source_patterns=[
                "experts.*.w1.weight",
                "experts.*.w3.weight",
                "experts.*.w1.activation_scale",
                "experts.*.w3.activation_scale",
            ],
            target_patterns=["gate_up_proj", "gate_up_proj_activation_scale"],
        )
        assert "gate_up_proj" in result
        assert "gate_up_proj_activation_scale" in result
        act = result["gate_up_proj_activation_scale"]
        assert act.shape == (n_experts,)
        for e in range(n_experts):
            expected = max(float(e), float(e + 1))
            assert act[e].item() == pytest.approx(expected)


@require_torch
class TestFP8AwareSplitAndUnstack:
    def test_roundtrip_bf16(self):
        merge_op = FP8AwareMergeAndConcatenate()
        split_op = FP8AwareSplitAndUnstack()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1_orig = [torch.randn(gate_dim, in_dim) for _ in range(n_experts)]
        w3_orig = [torch.randn(up_dim, in_dim) for _ in range(n_experts)]
        fused = merge_op.convert(
            input_dict={"w1.weight": w1_orig, "w3.weight": w3_orig},
            source_patterns=["w1.weight", "w3.weight"],
            target_patterns=["gate_up_proj"],
        )
        # Reverse: pass the fused tensor as a single-element list (simulating loading)
        reversed_result = split_op.convert(
            input_dict={"gate_up_proj": [fused["gate_up_proj"]]},
            source_patterns=["gate_up_proj"],
            target_patterns=["w1.weight", "w3.weight"],
        )
        assert "w1.weight" in reversed_result
        assert "w3.weight" in reversed_result
        # Each should be a list of n_experts tensors
        w1_recovered = reversed_result["w1.weight"]
        w3_recovered = reversed_result["w3.weight"]
        assert len(w1_recovered) == n_experts
        assert len(w3_recovered) == n_experts
        for i in range(n_experts):
            torch.testing.assert_close(w1_recovered[i], w1_orig[i])
            torch.testing.assert_close(w3_recovered[i], w3_orig[i])

    def test_roundtrip_per_tensor_fp8(self):
        merge_op = FP8AwareMergeAndConcatenate()
        split_op = FP8AwareSplitAndUnstack()
        n_experts = 2
        gate_dim, up_dim, in_dim = 16, 16, 32
        scale = torch.tensor(0.5)
        w1 = [torch.randn(gate_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w3 = [torch.randn(up_dim, in_dim).to(torch.float8_e4m3fn) for _ in range(n_experts)]
        w1_scales = [scale.clone() for _ in range(n_experts)]
        w3_scales = [scale.clone() for _ in range(n_experts)]
        fused = merge_op.convert(
            input_dict={
                "w1.weight": w1,
                "w3.weight": w3,
                "w1.qscale_weight": w1_scales,
                "w3.qscale_weight": w3_scales,
            },
            source_patterns=["w1.weight", "w3.weight", "w1.qscale_weight", "w3.qscale_weight"],
            target_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
        )
        reversed_result = split_op.convert(
            input_dict={
                "gate_up_proj": [fused["gate_up_proj"]],
                "gate_up_proj_scale_inv": [fused["gate_up_proj_scale_inv"]],
            },
            source_patterns=["gate_up_proj", "gate_up_proj_scale_inv"],
            target_patterns=["w1.weight", "w3.weight", "w1.qscale_weight", "w3.qscale_weight"],
        )
        assert "w1.weight" in reversed_result
        assert "w3.weight" in reversed_result
        for i in range(n_experts):
            torch.testing.assert_close(reversed_result["w1.weight"][i], w1[i])
            torch.testing.assert_close(reversed_result["w3.weight"][i], w3[i])

    def test_roundtrip_activation_scales(self):
        merge_op = FP8AwareMergeAndConcatenate()
        split_op = FP8AwareSplitAndUnstack()
        n_experts = 4
        gate_dim, up_dim, in_dim = 16, 16, 32
        w1_orig = [torch.randn(gate_dim, in_dim) for _ in range(n_experts)]
        w3_orig = [torch.randn(up_dim, in_dim) for _ in range(n_experts)]
        act_scales = [torch.tensor(float(i + 1)) for i in range(n_experts)]
        fused = merge_op.convert(
            input_dict={
                "w1.weight": w1_orig,
                "w3.weight": w3_orig,
                "w1.qscale_act": act_scales,
                "w3.qscale_act": [s.clone() for s in act_scales],
            },
            source_patterns=["w1.weight", "w3.weight", "w1.qscale_act", "w3.qscale_act"],
            target_patterns=["gate_up_proj", "gate_up_proj_activation_scale"],
        )
        assert "gate_up_proj_activation_scale" in fused

        reversed_result = split_op.convert(
            input_dict={
                "gate_up_proj": [fused["gate_up_proj"]],
                "gate_up_proj_activation_scale": [fused["gate_up_proj_activation_scale"]],
            },
            source_patterns=["gate_up_proj", "gate_up_proj_activation_scale"],
            target_patterns=["w1.weight", "w3.weight", "w1.qscale_act", "w3.qscale_act"],
        )
        assert "w1.qscale_act" in reversed_result
        assert "w3.qscale_act" in reversed_result
        for i in range(n_experts):
            assert reversed_result["w1.qscale_act"][i].item() == pytest.approx(float(i + 1))
            assert reversed_result["w3.qscale_act"][i].item() == pytest.approx(float(i + 1))


@require_torch
class TestFP8ScaleFusionMerge:
    def test_per_tensor_merge(self):
        op = FP8ScaleFusionMerge()
        n_experts = 4
        w1_scales = [torch.tensor(0.5) for _ in range(n_experts)]
        w3_scales = [torch.tensor(0.3) for _ in range(n_experts)]
        result = op.convert(
            input_dict={"w1.weight_scale_inv": w1_scales, "w3.weight_scale_inv": w3_scales},
            source_patterns=["w1.weight_scale_inv", "w3.weight_scale_inv"],
            target_patterns=["gate_up_proj_scale_inv"],
        )
        assert "gate_up_proj_scale_inv" in result
        fused = result["gate_up_proj_scale_inv"]
        assert fused.shape == (n_experts, 1, 1)
        for e in range(n_experts):
            assert fused[e, 0, 0].item() == pytest.approx(0.5)

    def test_blockwise_merge(self):
        op = FP8ScaleFusionMerge()
        n_experts = 2
        w1_scales = [torch.randn(4, 2) for _ in range(n_experts)]
        w3_scales = [torch.randn(4, 2) for _ in range(n_experts)]
        result = op.convert(
            input_dict={"w1.weight_scale_inv": w1_scales, "w3.weight_scale_inv": w3_scales},
            source_patterns=["w1.weight_scale_inv", "w3.weight_scale_inv"],
            target_patterns=["gate_up_proj_scale_inv"],
        )
        fused = result["gate_up_proj_scale_inv"]
        assert fused.shape == (n_experts, 8, 2)

    def test_roundtrip_per_tensor(self):
        merge_op = FP8ScaleFusionMerge()
        split_op = FP8ScaleFusionSplit()
        n_experts = 3
        scales = [torch.tensor(float(i + 1)) for i in range(n_experts)]
        fused = merge_op.convert(
            input_dict={
                "w1.weight_scale_inv": scales,
                "w3.weight_scale_inv": [s.clone() for s in scales],
            },
            source_patterns=["w1.weight_scale_inv", "w3.weight_scale_inv"],
            target_patterns=["gate_up_proj_scale_inv"],
        )
        reversed_result = split_op.convert(
            input_dict={"gate_up_proj_scale_inv": [fused["gate_up_proj_scale_inv"]]},
            source_patterns=["gate_up_proj_scale_inv"],
            target_patterns=["w1.weight_scale_inv", "w3.weight_scale_inv"],
        )
        assert "w1.weight_scale_inv" in reversed_result
        assert "w3.weight_scale_inv" in reversed_result
        for i in range(n_experts):
            assert reversed_result["w1.weight_scale_inv"][i].item() == pytest.approx(float(i + 1))

    def test_roundtrip_blockwise(self):
        merge_op = FP8ScaleFusionMerge()
        split_op = FP8ScaleFusionSplit()
        n_experts = 2
        w1_scales = [torch.randn(4, 2) for _ in range(n_experts)]
        w3_scales = [torch.randn(4, 2) for _ in range(n_experts)]
        fused = merge_op.convert(
            input_dict={"w1.scale": w1_scales, "w3.scale": w3_scales},
            source_patterns=["w1.scale", "w3.scale"],
            target_patterns=["gate_up_proj_scale_inv"],
        )
        reversed_result = split_op.convert(
            input_dict={"gate_up_proj_scale_inv": [fused["gate_up_proj_scale_inv"]]},
            source_patterns=["gate_up_proj_scale_inv"],
            target_patterns=["w1.scale", "w3.scale"],
        )
        for i in range(n_experts):
            torch.testing.assert_close(reversed_result["w1.scale"][i], w1_scales[i])
            torch.testing.assert_close(reversed_result["w3.scale"][i], w3_scales[i])
