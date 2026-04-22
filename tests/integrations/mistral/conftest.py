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

import pytest


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


@pytest.fixture(scope="session")
def mistral_base_fields():
    return copy.deepcopy(MISTRAL_BASE_FIELDS)


@pytest.fixture(scope="session")
def mistral_params():
    return copy.deepcopy(MISTRAL_PARAMS)


@pytest.fixture(scope="session")
def ministral3_params():
    return copy.deepcopy(MINISTRAL3_PARAMS)


@pytest.fixture(scope="session")
def mistral4_params():
    return copy.deepcopy(MISTRAL4_PARAMS)


@pytest.fixture(scope="session")
def mistral3_params():
    return copy.deepcopy(MISTRAL3_PARAMS)
