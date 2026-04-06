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
r"""Typed dataclass hierarchy for Mistral native `params.json` configs.

Provides conversion to/from HuggingFace config objects. Pure Python — no torch dependency.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Generic, Literal, Self, TypeVar, overload

from ...configuration_utils import PreTrainedConfig
from ...models.ministral3.configuration_ministral3 import Ministral3Config
from ...models.mistral.configuration_mistral import MistralConfig
from ...models.mistral3.configuration_mistral3 import Mistral3Config
from ...models.mistral4.configuration_mistral4 import Mistral4Config
from ...models.pixtral.configuration_pixtral import PixtralVisionConfig


HFConfigT = TypeVar("HFConfigT", bound=PreTrainedConfig)

MistralModelType = Literal["mistral", "ministral3", "mistral4", "mistral3"]


def _detect_text_model_type(params: dict) -> Literal["mistral", "ministral3", "mistral4"]:
    r"""Infer the text backbone model type from native `params.json` keys."""
    if "moe" in params:
        return "mistral4"
    if "yarn" in params or "quantization" in params:
        return "ministral3"
    return "mistral"


def _extract_rope_theta(config: PreTrainedConfig) -> float:
    r"""Extract `rope_theta` from an HF config, checking `rope_parameters` first."""
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params and isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    if hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    return 10000.0


def _extract_yarn(config: PreTrainedConfig) -> YarnNativeConfig | None:
    r"""Extract YaRN parameters from an HF config's `rope_parameters`."""
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or not isinstance(rope_params, dict):
        return None
    rope_type = rope_params.get("rope_type", rope_params.get("type"))
    if rope_type != "yarn":
        return None
    return YarnNativeConfig(
        factor=rope_params["factor"],
        original_max_position_embeddings=rope_params["original_max_position_embeddings"],
        beta=rope_params.get("beta_fast", 32.0),
        alpha=rope_params.get("beta_slow", 1.0),
    )


@dataclass
class YarnNativeConfig:
    r"""YaRN RoPE scaling parameters from native `params.json`.

    Args:
        factor: Scaling factor for YaRN interpolation.
        original_max_position_embeddings: Pre-training context length before scaling.
        beta: Maps to `beta_fast` in HF `rope_parameters`.
        alpha: Maps to `beta_slow` in HF `rope_parameters`.
    """

    factor: float
    original_max_position_embeddings: int
    beta: float
    alpha: float


@dataclass
class FP8NativeConfig:
    r"""FP8 quantization parameters from native `params.json`.

    Args:
        qformat_weight: Weight quantization format, must be `"fp8_e4m3"`.
        qscheme_act: Activation quantization scheme (`"TENSOR"` for static, `"DYNAMIC"` for dynamic).

    Raises:
        ValueError: If `qformat_weight` is not `"fp8_e4m3"`.
    """

    qformat_weight: str
    qscheme_act: str

    def __post_init__(self) -> None:
        if self.qformat_weight != "fp8_e4m3":
            raise ValueError(f"Unsupported quantization format {self.qformat_weight!r}; only 'fp8_e4m3' is supported.")


@dataclass
class MoeNativeConfig:
    r"""Mixture-of-Experts parameters from native `params.json`.

    Args:
        num_experts: Total number of routed experts.
        num_experts_per_tok: Experts activated per token.
        expert_hidden_dim: Hidden dimension of each expert FFN.
        first_k_dense_replace: Number of initial dense layers before MoE layers begin.
        num_shared_experts: Number of always-active shared experts.
        routed_scale: Multiplicative scale applied to routed expert outputs.
        num_expert_groups: Number of expert groups for grouped routing.
        num_expert_groups_per_tok: Expert groups selected per token.
    """

    num_experts: int
    num_experts_per_tok: int
    expert_hidden_dim: int
    first_k_dense_replace: int = 0
    num_shared_experts: int = 1
    routed_scale: float = 1.0
    num_expert_groups: int = 1
    num_expert_groups_per_tok: int = 1


@dataclass
class VisionEncoderNativeConfig:
    r"""Vision encoder parameters from native `params.json`.

    Args:
        hidden_size: Dimensionality of the encoder hidden states.
        num_hidden_layers: Number of transformer layers.
        num_attention_heads: Number of attention heads.
        patch_size: Size of each image patch in pixels.
        image_size: Input image resolution in pixels.
        head_dim: Dimensionality of each attention head.
        intermediate_size: Dimensionality of the FFN intermediate layer.
        adapter_bias: Whether the multimodal projector uses bias.
        spatial_merge_size: Factor for spatial token merging.
        image_token_id: Token ID used for image placeholders.
    """

    hidden_size: int
    num_hidden_layers: int
    num_attention_heads: int
    patch_size: int
    image_size: int
    head_dim: int
    intermediate_size: int
    adapter_bias: bool = False
    spatial_merge_size: int = 2
    image_token_id: int = 10


class NativeToHFConfigMixin(ABC, Generic[HFConfigT]):
    r"""Abstract mixin binding a native config dataclass to its HF config type.

    Subclasses must implement `to_hf_config`, `from_hf_config`, and `from_params_json`
    with their concrete `HFConfigT` bound.
    """

    @abstractmethod
    def to_hf_config(self) -> HFConfigT:
        r"""Convert this native config to the corresponding HF config object."""
        ...

    @classmethod
    @abstractmethod
    def from_hf_config(cls, config: HFConfigT) -> Self:
        r"""Construct a native config from an HF config object."""
        ...

    @classmethod
    @abstractmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Construct a native config from a `params.json` dict."""
        ...


@dataclass
class MistralNativeConfig(NativeToHFConfigMixin[MistralConfig]):
    r"""Base native config representing `params.json` for Mistral text models.

    Args:
        dim: Hidden size of the model (maps to HF `hidden_size`).
        n_layers: Number of transformer layers (maps to HF `num_hidden_layers`).
        hidden_dim: FFN intermediate size (maps to HF `intermediate_size`).
        n_heads: Number of attention heads (maps to HF `num_attention_heads`).
        norm_eps: Epsilon for RMS normalization (maps to HF `rms_norm_eps`).
        head_dim: Dimensionality of each attention head.
        vocab_size: Size of the vocabulary.
        n_kv_heads: Number of key-value heads for GQA (maps to HF `num_key_value_heads`).
        rope_theta: Base frequency for rotary position embeddings.
        sliding_window: Sliding window attention size, `None` for full attention.
        max_position_embeddings: Maximum sequence length the model supports.
        tied_embeddings: Whether input and output embeddings share weights
            (maps to HF `tie_word_embeddings`).
    """

    dim: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    norm_eps: float
    head_dim: int
    vocab_size: int
    n_kv_heads: int | None = None
    rope_theta: float = 10000.0
    sliding_window: int | None = None
    max_position_embeddings: int = 32768
    tied_embeddings: bool = False

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Parse a `params.json` dict into this dataclass."""
        return cls(
            dim=params["dim"],
            n_layers=params["n_layers"],
            hidden_dim=params["hidden_dim"],
            n_heads=params["n_heads"],
            norm_eps=params["norm_eps"],
            head_dim=params["head_dim"],
            vocab_size=params["vocab_size"],
            n_kv_heads=params.get("n_kv_heads"),
            rope_theta=params.get("rope_theta", 10000.0),
            sliding_window=params.get("sliding_window"),
            max_position_embeddings=params.get("max_position_embeddings", params.get("max_seq_len", 32768)),
            tied_embeddings=params.get("tied_embeddings", False),
        )

    def to_hf_config(self) -> MistralConfig:
        r"""Convert to an HF `MistralConfig` object."""
        return MistralConfig(
            hidden_size=self.dim,
            num_hidden_layers=self.n_layers,
            intermediate_size=self.hidden_dim,
            num_attention_heads=self.n_heads,
            rms_norm_eps=self.norm_eps,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            num_key_value_heads=self.n_kv_heads if self.n_kv_heads is not None else self.n_heads,
            rope_theta=self.rope_theta,
            sliding_window=self.sliding_window,
            max_position_embeddings=self.max_position_embeddings,
            tie_word_embeddings=self.tied_embeddings,
        )

    @classmethod
    def from_hf_config(cls, config: MistralConfig) -> Self:
        r"""Reverse: HF config → native config (for save)."""
        rope_theta = _extract_rope_theta(config)
        if config.head_dim is None:
            raise ValueError("`head_dim` should not be `None` for `MistralConfig`")
        return cls(
            dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            hidden_dim=config.intermediate_size,
            n_heads=config.num_attention_heads,
            norm_eps=config.rms_norm_eps,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            n_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            max_position_embeddings=config.max_position_embeddings,
            tied_embeddings=config.tie_word_embeddings,
        )


@dataclass
class Ministral3NativeConfig(MistralNativeConfig, NativeToHFConfigMixin[Ministral3Config]):
    r"""Extends base with YaRN RoPE and FP8 quantization.

    Args:
        yarn: YaRN RoPE scaling configuration, `None` when not using YaRN.
        quantization: FP8 quantization configuration, `None` when not quantized.
    """

    yarn: YarnNativeConfig | None = None
    quantization: FP8NativeConfig | None = None

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Parse a `params.json` dict into this dataclass."""
        base = MistralNativeConfig.from_params_json(params)
        yarn = None
        if "yarn" in params:
            y = params["yarn"]
            yarn = YarnNativeConfig(
                factor=y["factor"],
                original_max_position_embeddings=y["original_max_position_embeddings"],
                beta=y["beta"],
                alpha=y["alpha"],
            )
        quantization = None
        if "quantization" in params:
            q = params["quantization"]
            quantization = FP8NativeConfig(
                qformat_weight=q["qformat_weight"],
                qscheme_act=q["qscheme_act"],
            )
        return cls(
            dim=base.dim,
            n_layers=base.n_layers,
            hidden_dim=base.hidden_dim,
            n_heads=base.n_heads,
            norm_eps=base.norm_eps,
            head_dim=base.head_dim,
            vocab_size=base.vocab_size,
            n_kv_heads=base.n_kv_heads,
            rope_theta=base.rope_theta,
            sliding_window=base.sliding_window,
            max_position_embeddings=base.max_position_embeddings,
            tied_embeddings=base.tied_embeddings,
            yarn=yarn,
            quantization=quantization,
        )

    def to_hf_config(self) -> Ministral3Config:
        r"""Convert to an HF `Ministral3Config` object."""
        kwargs: dict = {
            "hidden_size": self.dim,
            "num_hidden_layers": self.n_layers,
            "intermediate_size": self.hidden_dim,
            "num_attention_heads": self.n_heads,
            "rms_norm_eps": self.norm_eps,
            "head_dim": self.head_dim,
            "vocab_size": self.vocab_size,
            "num_key_value_heads": self.n_kv_heads if self.n_kv_heads is not None else self.n_heads,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tied_embeddings,
        }
        if self.yarn is not None:
            kwargs["rope_parameters"] = {
                "type": "yarn",
                "rope_theta": self.rope_theta,
                "factor": self.yarn.factor,
                "original_max_position_embeddings": self.yarn.original_max_position_embeddings,
                "beta_fast": self.yarn.beta,
                "beta_slow": self.yarn.alpha,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
            }
        if self.quantization is not None:
            activation_scheme = "static" if self.quantization.qscheme_act == "TENSOR" else "dynamic"
            kwargs["quantization_config"] = {
                "quant_method": "fp8",
                "activation_scheme": activation_scheme,
            }
        return Ministral3Config(**kwargs)

    @classmethod
    def from_hf_config(cls, config: Ministral3Config) -> Self:
        r"""Reverse: HF config → native config (for save)."""
        rope_theta = _extract_rope_theta(config)
        yarn = _extract_yarn(config)
        return cls(
            dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            hidden_dim=config.intermediate_size,
            n_heads=config.num_attention_heads,
            norm_eps=config.rms_norm_eps,
            head_dim=config.head_dim,
            vocab_size=config.vocab_size,
            n_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window,
            max_position_embeddings=config.max_position_embeddings,
            tied_embeddings=config.tie_word_embeddings,
            yarn=yarn,
            quantization=None,  # FP8 doesn't roundtrip
        )


@dataclass
class Mistral4NativeConfig(MistralNativeConfig, NativeToHFConfigMixin[Mistral4Config]):
    r"""Extends base with MoE/MLA architecture fields.

    Args:
        q_lora_rank: Rank of the LoRA decomposition for query projections.
        qk_rope_head_dim: Dimensionality of the RoPE-applied portion of QK heads.
        qk_nope_head_dim: Dimensionality of the non-RoPE portion of QK heads.
        kv_lora_rank: Rank of the LoRA decomposition for key-value projections.
        v_head_dim: Dimensionality of each value head.
        moe: Mixture-of-Experts configuration, `None` for dense layers.
        yarn: YaRN RoPE scaling configuration, `None` when not using YaRN.
    """

    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    kv_lora_rank: int | None = None
    v_head_dim: int | None = None
    moe: MoeNativeConfig | None = None
    yarn: YarnNativeConfig | None = None

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Parse a `params.json` dict into this dataclass."""
        base = MistralNativeConfig.from_params_json(params)
        moe = None
        if "moe" in params:
            m = params["moe"]
            moe = MoeNativeConfig(
                num_experts=m["num_experts"],
                num_experts_per_tok=m["num_experts_per_tok"],
                expert_hidden_dim=m["expert_hidden_dim"],
                first_k_dense_replace=m.get("first_k_dense_replace", 0),
                num_shared_experts=m.get("num_shared_experts", 1),
                routed_scale=m.get("routed_scale", 1.0),
                num_expert_groups=m.get("num_expert_groups", 1),
                num_expert_groups_per_tok=m.get("num_expert_groups_per_tok", 1),
            )
        yarn = None
        if "yarn" in params:
            y = params["yarn"]
            yarn = YarnNativeConfig(
                factor=y["factor"],
                original_max_position_embeddings=y["original_max_position_embeddings"],
                beta=y["beta"],
                alpha=y["alpha"],
            )
        return cls(
            dim=base.dim,
            n_layers=base.n_layers,
            hidden_dim=base.hidden_dim,
            n_heads=base.n_heads,
            norm_eps=base.norm_eps,
            head_dim=base.head_dim,
            vocab_size=base.vocab_size,
            n_kv_heads=base.n_kv_heads,
            rope_theta=base.rope_theta,
            sliding_window=base.sliding_window,
            max_position_embeddings=base.max_position_embeddings,
            tied_embeddings=base.tied_embeddings,
            q_lora_rank=params.get("q_lora_rank"),
            qk_rope_head_dim=params.get("qk_rope_head_dim"),
            qk_nope_head_dim=params.get("qk_nope_head_dim"),
            kv_lora_rank=params.get("kv_lora_rank"),
            v_head_dim=params.get("v_head_dim"),
            moe=moe,
            yarn=yarn,
        )

    def to_hf_config(self) -> Mistral4Config:
        r"""Convert to an HF `Mistral4Config` object."""
        kwargs: dict = {
            "hidden_size": self.dim,
            "num_hidden_layers": self.n_layers,
            "intermediate_size": self.hidden_dim,
            "num_attention_heads": self.n_heads,
            "rms_norm_eps": self.norm_eps,
            "vocab_size": self.vocab_size,
            "num_key_value_heads": self.n_kv_heads if self.n_kv_heads is not None else self.n_heads,
            "sliding_window": self.sliding_window,
            "max_position_embeddings": self.max_position_embeddings,
            "tie_word_embeddings": self.tied_embeddings,
        }
        # MLA fields
        for field_name in ("q_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "kv_lora_rank", "v_head_dim"):
            value = getattr(self, field_name)
            if value is not None:
                kwargs[field_name] = value

        # MoE fields
        if self.moe is not None:
            kwargs.update(
                {
                    "n_routed_experts": self.moe.num_experts,
                    "num_experts_per_tok": self.moe.num_experts_per_tok,
                    "moe_intermediate_size": self.moe.expert_hidden_dim,
                    "first_k_dense_replace": self.moe.first_k_dense_replace,
                    "n_shared_experts": self.moe.num_shared_experts,
                    "routed_scaling_factor": self.moe.routed_scale,
                    "n_group": self.moe.num_expert_groups,
                    "topk_group": self.moe.num_expert_groups_per_tok,
                    "norm_topk_prob": True,
                }
            )

        # YaRN RoPE
        if self.yarn is not None:
            qk_rope = self.qk_rope_head_dim or 64
            qk_nope = self.qk_nope_head_dim or 64
            kwargs["rope_parameters"] = {
                "type": "yarn",
                "rope_theta": self.rope_theta,
                "factor": self.yarn.factor,
                "original_max_position_embeddings": self.yarn.original_max_position_embeddings,
                "beta_fast": self.yarn.beta,
                "beta_slow": self.yarn.alpha,
                "mscale_all_dim": 1.0,
                "mscale": 1.0,
                "llama_4_scaling_beta": 0.1,
                "partial_rotary_factor": qk_rope / (qk_nope + qk_rope),
            }

        return Mistral4Config(**kwargs)

    @classmethod
    def from_hf_config(cls, config: Mistral4Config) -> Self:
        r"""Reverse: HF config → native config (for save)."""
        rope_theta = _extract_rope_theta(config)
        yarn = _extract_yarn(config)
        moe = MoeNativeConfig(
            num_experts=config.n_routed_experts,
            num_experts_per_tok=config.num_experts_per_tok,
            expert_hidden_dim=config.moe_intermediate_size,
            first_k_dense_replace=config.first_k_dense_replace or 0,
            num_shared_experts=config.n_shared_experts,
            routed_scale=config.routed_scaling_factor,
            num_expert_groups=config.n_group or 1,
            num_expert_groups_per_tok=config.topk_group or 1,
        )
        return cls(
            dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            hidden_dim=config.intermediate_size,
            n_heads=config.num_attention_heads,
            norm_eps=config.rms_norm_eps,
            head_dim=config.qk_nope_head_dim + config.qk_rope_head_dim,
            vocab_size=config.vocab_size,
            n_kv_heads=config.num_key_value_heads,
            rope_theta=rope_theta,
            sliding_window=config.sliding_window if hasattr(config, "sliding_window") else None,
            max_position_embeddings=config.max_position_embeddings,
            tied_embeddings=config.tie_word_embeddings,
            q_lora_rank=config.q_lora_rank,
            qk_rope_head_dim=config.qk_rope_head_dim,
            qk_nope_head_dim=config.qk_nope_head_dim,
            kv_lora_rank=config.kv_lora_rank,
            v_head_dim=config.v_head_dim,
            moe=moe,
            yarn=yarn,
        )


@dataclass
class Mistral3NativeConfig(NativeToHFConfigMixin[Mistral3Config]):
    r"""Composite VLM config wrapping a text backbone config and a vision encoder config.

    The text backbone type is auto-detected in `from_params_json` based on the presence
    of `moe` (→ `Mistral4NativeConfig`), `yarn`/`quantization` (→ `Ministral3NativeConfig`),
    or neither (→ `MistralNativeConfig`).

    Args:
        text_config: Native config for the text backbone.
        vision_encoder: Native config for the vision encoder.
    """

    text_config: MistralNativeConfig | Ministral3NativeConfig | Mistral4NativeConfig
    vision_encoder: VisionEncoderNativeConfig

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Parse a `params.json` dict, auto-detecting the text backbone type."""
        ve_dict = params["vision_encoder"]
        vision_encoder = VisionEncoderNativeConfig(
            hidden_size=ve_dict["hidden_size"],
            num_hidden_layers=ve_dict["num_hidden_layers"],
            num_attention_heads=ve_dict["num_attention_heads"],
            patch_size=ve_dict["patch_size"],
            image_size=ve_dict["image_size"],
            head_dim=ve_dict["head_dim"],
            intermediate_size=ve_dict["intermediate_size"],
            adapter_bias=ve_dict.get("adapter_bias", False),
            spatial_merge_size=ve_dict.get("spatial_merge_size", 2),
            image_token_id=ve_dict.get("image_token_id", 10),
        )

        text_params = {k: v for k, v in params.items() if k != "vision_encoder"}
        text_model_type = _detect_text_model_type(text_params)
        text_config = native_config_for_model_type(text_model_type, text_params)

        return cls(text_config=text_config, vision_encoder=vision_encoder)

    def to_hf_config(self) -> Mistral3Config:
        r"""Convert to an HF `Mistral3Config` object."""
        text_hf = self.text_config.to_hf_config()
        vision_hf = PixtralVisionConfig(
            hidden_size=self.vision_encoder.hidden_size,
            num_hidden_layers=self.vision_encoder.num_hidden_layers,
            num_attention_heads=self.vision_encoder.num_attention_heads,
            patch_size=self.vision_encoder.patch_size,
            image_size=self.vision_encoder.image_size,
            intermediate_size=self.vision_encoder.intermediate_size,
            hidden_act="silu",
        )
        return Mistral3Config(
            text_config=text_hf,
            vision_config=vision_hf,
            multimodal_projector_bias=self.vision_encoder.adapter_bias,
            image_token_id=self.vision_encoder.image_token_id,
            spatial_merge_size=self.vision_encoder.spatial_merge_size,
            vision_feature_layer=-1,
            tie_word_embeddings=self.text_config.tied_embeddings,
        )

    @classmethod
    def from_hf_config(cls, config: Mistral3Config) -> Self:
        r"""Reverse: HF config → native config (for save)."""
        text_hf = config.text_config
        vision_hf = config.vision_config

        text_model_type = getattr(text_hf, "model_type", "mistral")
        text_config = native_config_from_hf_config(text_model_type, text_hf)

        vision_encoder = VisionEncoderNativeConfig(
            hidden_size=vision_hf.hidden_size,
            num_hidden_layers=vision_hf.num_hidden_layers,
            num_attention_heads=vision_hf.num_attention_heads,
            patch_size=vision_hf.patch_size if isinstance(vision_hf.patch_size, int) else vision_hf.patch_size[0],
            image_size=vision_hf.image_size if isinstance(vision_hf.image_size, int) else vision_hf.image_size[0],
            head_dim=vision_hf.head_dim,
            intermediate_size=vision_hf.intermediate_size,
            adapter_bias=getattr(config, "multimodal_projector_bias", False),
            spatial_merge_size=config.spatial_merge_size,
            image_token_id=config.image_token_index,
        )

        return cls(text_config=text_config, vision_encoder=vision_encoder)


@overload
def native_config_for_model_type(model_type: Literal["mistral"], params: dict) -> MistralNativeConfig: ...
@overload
def native_config_for_model_type(model_type: Literal["ministral3"], params: dict) -> Ministral3NativeConfig: ...
@overload
def native_config_for_model_type(model_type: Literal["mistral4"], params: dict) -> Mistral4NativeConfig: ...
@overload
def native_config_for_model_type(model_type: Literal["mistral3"], params: dict) -> Mistral3NativeConfig: ...


def native_config_for_model_type(
    model_type: MistralModelType, params: dict
) -> MistralNativeConfig | Ministral3NativeConfig | Mistral4NativeConfig | Mistral3NativeConfig:
    r"""Dispatch to the correct native config class by `model_type`."""
    match model_type:
        case "mistral":
            return MistralNativeConfig.from_params_json(params)
        case "ministral3":
            return Ministral3NativeConfig.from_params_json(params)
        case "mistral4":
            return Mistral4NativeConfig.from_params_json(params)
        case "mistral3":
            return Mistral3NativeConfig.from_params_json(params)
        case _:
            raise ValueError(f"Unknown model type: {model_type!r}")


@overload
def native_config_from_hf_config(model_type: Literal["mistral"], config: MistralConfig) -> MistralNativeConfig: ...
@overload
def native_config_from_hf_config(
    model_type: Literal["ministral3"], config: Ministral3Config
) -> Ministral3NativeConfig: ...
@overload
def native_config_from_hf_config(model_type: Literal["mistral4"], config: Mistral4Config) -> Mistral4NativeConfig: ...
@overload
def native_config_from_hf_config(model_type: Literal["mistral3"], config: Mistral3Config) -> Mistral3NativeConfig: ...


def native_config_from_hf_config(
    model_type: MistralModelType,
    config: MistralConfig | Ministral3Config | Mistral4Config | Mistral3Config,
) -> MistralNativeConfig | Ministral3NativeConfig | Mistral4NativeConfig | Mistral3NativeConfig:
    r"""Reverse dispatch: HF config → native config."""
    match model_type:
        case "mistral":
            assert isinstance(config, MistralConfig)
            return MistralNativeConfig.from_hf_config(config)
        case "ministral3":
            assert isinstance(config, Ministral3Config)
            return Ministral3NativeConfig.from_hf_config(config)
        case "mistral4":
            assert isinstance(config, Mistral4Config)
            return Mistral4NativeConfig.from_hf_config(config)
        case "mistral3":
            assert isinstance(config, Mistral3Config)
            return Mistral3NativeConfig.from_hf_config(config)
        case _:
            raise ValueError(f"Unknown model type: {model_type!r}")
