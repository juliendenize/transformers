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

import copy

from transformers import Ministral3Config, Mistral3Config, Mistral4Config, MistralConfig
from transformers.integrations.mistral.params_conversion import (
    Llama4Scaling,
    MistralNativeConfig,
    MOEModelArgs,
    VisionEncoderArgs,
    YarnArgs,
)
from transformers.models.pixtral.configuration_pixtral import PixtralVisionConfig


MISTRAL_BASE_FIELDS = {
    "dim": 4096,
    "n_layers": 32,
    "hidden_dim": 14336,
    "n_heads": 32,
    "norm_eps": 1e-5,
    "head_dim": 128,
    "vocab_size": 32000,
}

MISTRAL_PARAMS = {
    **MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 10000.0,
    "sliding_window": 4096,
    "max_position_embeddings": 32768,
}

MINISTRAL3_PARAMS = {
    **MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 1000000.0,
    "max_position_embeddings": 262144,
    "tied_embeddings": True,
    "yarn": {
        "factor": 16.0,
        "original_max_position_embeddings": 16384,
        "beta": 32.0,
        "alpha": 1.0,
        "apply_scale": False,
    },
    "quantization": {
        "qformat_weight": "fp8_e4m3",
        "qscheme_act": "TENSOR",
    },
}

MISTRAL4_PARAMS = {
    **MISTRAL_BASE_FIELDS,
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

MISTRAL3_PARAMS = {
    **MISTRAL_BASE_FIELDS,
    "n_kv_heads": 8,
    "rope_theta": 1000000000.0,
    "max_position_embeddings": 131072,
    "vision_encoder": {
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "num_attention_heads": 16,
        "patch_size": 14,
        "image_size": 1540,
        "intermediate_size": 4096,
        "num_channels": 3,
        "max_image_size": 1540,
        "rope_theta": 10000.0,
        "mm_projector_id": "patch_merge",
        "add_pre_mm_projector_layer_norm": True,
        "adapter_bias": False,
        "spatial_merge_size": 2,
        "image_token_id": 10,
        "image_break_token_id": 12,
        "image_end_token_id": 13,
    },
}


def mistral_params() -> dict:
    return copy.deepcopy(MISTRAL_PARAMS)


def ministral3_params() -> dict:
    return copy.deepcopy(MINISTRAL3_PARAMS)


def mistral4_params() -> dict:
    return copy.deepcopy(MISTRAL4_PARAMS)


def mistral3_params() -> dict:
    return copy.deepcopy(MISTRAL3_PARAMS)


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


def yarn_args() -> YarnArgs:
    return YarnArgs(factor=16.0, original_max_position_embeddings=16384, beta=32.0, alpha=1.0, apply_scale=False)


def llama4_scaling() -> Llama4Scaling:
    return Llama4Scaling(original_max_position_embeddings=16384, beta=0.1)


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


def ministral3_native_config() -> MistralNativeConfig:
    _yarn_args = yarn_args()
    _llama4_scaling = llama4_scaling()
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
        yarn=_yarn_args,
        llama_4_scaling=_llama4_scaling,
    )


def mistral3_native_config() -> MistralNativeConfig:
    _vision_encoder_args = vision_encoder_args()
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
        vision_encoder=_vision_encoder_args,
    )


def mistral4_native_config() -> MistralNativeConfig:
    _moe_args = moe_args()
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
        yarn=YarnArgs(factor=128.0, original_max_position_embeddings=8192, beta=32.0, alpha=1.0, apply_scale=False),
        llama_4_scaling=Llama4Scaling(original_max_position_embeddings=8192, beta=0.1),
        moe=_moe_args,
    )


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
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
        },
        quantization_config=None,
    )


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
            "mscale": 1.0,
            "mscale_all_dim": 1.0,
            "llama_4_scaling_beta": 0.1,
            "partial_rotary_factor": 0.5,
        },
    )


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
