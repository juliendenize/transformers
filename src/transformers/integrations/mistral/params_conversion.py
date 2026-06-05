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
"""Bidirectional config conversion between Mistral native params.json and HF config formats."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from ...configuration_utils import PreTrainedConfig
from ...modeling_rope_utils import RopeParameters
from ...models.ministral3.configuration_ministral3 import Ministral3Config
from ...models.mistral.configuration_mistral import MistralConfig
from ...models.mistral3.configuration_mistral3 import Mistral3Config
from ...models.mistral4.configuration_mistral4 import Mistral4Config
from ...models.pixtral.configuration_pixtral import PixtralVisionConfig
from ...quantizers.auto import AutoQuantizationConfig
from ...utils.quantization_config import QuantizationConfigMixin


_QUANTIZATION_SCHEME_MAP = {"TENSOR": "static"}
_REVERSE_QUANTIZATION_SCHEME_MAP = {v: k for k, v in _QUANTIZATION_SCHEME_MAP.items()}


class MistralModelType(str, Enum):
    """Mistral model type identifiers."""

    MISTRAL = "mistral"
    MINISTRAL3 = "ministral3"
    MISTRAL4 = "mistral4"
    MISTRAL3 = "mistral3"


MistralHFConfigType = MistralConfig | Mistral3Config | Ministral3Config | Mistral4Config


def _extract_rope_theta(config: PreTrainedConfig) -> float:
    """Extract `rope_theta` from an HF config, checking `rope_parameters` first."""
    rope_params = getattr(config, "rope_parameters", None)
    if rope_params and isinstance(rope_params, dict) and "rope_theta" in rope_params:
        return float(rope_params["rope_theta"])
    elif hasattr(config, "rope_theta"):
        return float(config.rope_theta)
    raise ValueError("`rope_theta` not found.")


def _extract_yarn(config: PreTrainedConfig) -> YarnArgs | None:
    """Extract YaRN parameters from an HF config's `rope_parameters`."""
    rope_params = getattr(config, "rope_parameters", None)
    if not rope_params or not isinstance(rope_params, dict):
        return None
    rope_type = rope_params.get("rope_type", rope_params.get("type"))
    if rope_type != "yarn":
        return None
    rope_mscale_all_dim = rope_params.get("mscale_all_dim")
    apply_scale = rope_mscale_all_dim is None or rope_mscale_all_dim != 1.0
    return YarnArgs(
        factor=rope_params["factor"],
        original_max_position_embeddings=rope_params["original_max_position_embeddings"],
        beta=int(rope_params["beta_fast"]),
        alpha=int(rope_params["beta_slow"]),
        apply_scale=apply_scale,
    )


@dataclass
class Llama4Scaling:
    """Llama 4 scaling parameters for extended context RoPE."""

    original_max_position_embeddings: int
    beta: float


@dataclass
class YarnArgs:
    """YaRN (Yet another RoPE extensioN) parameters.

    Note on field naming: the Mistral native format uses ``beta`` / ``alpha``,
    which map to HF ``rope_parameters["beta_fast"]`` / ``rope_parameters["beta_slow"]``
    respectively.
    """

    factor: int
    original_max_position_embeddings: int
    beta: int  # corresponds to HF rope_parameters["beta_fast"]
    alpha: int  # corresponds to HF rope_parameters["beta_slow"]
    apply_scale: bool


class QFormat(str, Enum):
    """Supported quantization weight formats."""

    FP8_E4M3 = "fp8_e4m3"


@dataclass
class QuantizationArgs:
    """Native Mistral quantization configuration."""

    qformat_weight: QFormat
    qscheme_act: str

    _SUPPORTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"TENSOR"})

    def __post_init__(self) -> None:
        if self.qformat_weight not in list(QFormat):
            raise ValueError(
                f"Unsupported quantization format {self.qformat_weight!r}; only {[q.value for q in QFormat]} are supported."
            )
        if self.qscheme_act not in self._SUPPORTED_SCHEMES:
            raise ValueError(
                f"Unsupported quantization scheme {self.qscheme_act!r}; "
                f"supported schemes: {sorted(self._SUPPORTED_SCHEMES)}."
            )


@dataclass
class MOEModelArgs:
    """Mixture-of-Experts architecture parameters."""

    first_k_dense_replace: int
    num_experts: int
    num_experts_per_tok: int
    num_expert_groups: int
    num_expert_groups_per_tok: int
    routed_scale: float
    expert_hidden_dim: int
    num_shared_experts: int
    expert_parallel: int = 1
    expert_model_parallel: int = 1
    route_every_n: int = 1


@dataclass
class VisionEncoderArgs:
    """Vision encoder configuration for multimodal Mistral models."""

    image_token_id: int
    image_break_token_id: int
    image_end_token_id: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    mm_projector_id: str
    spatial_merge_size: int
    hidden_size: int
    num_channels: int
    image_size: int
    max_image_size: int
    patch_size: int
    rope_theta: float
    add_pre_mm_projector_layer_norm: bool
    adapter_bias: bool


@dataclass
class MistralNativeConfig:
    """Complete native Mistral configuration from params.json."""

    dim: int
    n_layers: int
    head_dim: int
    hidden_dim: int
    n_heads: int
    n_kv_heads: int
    rope_theta: float
    norm_eps: float
    vocab_size: int
    max_position_embeddings: int | None = None
    sliding_window: int | None = None
    tied_embeddings: bool = False
    llama_4_scaling: Llama4Scaling | None = None
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    kv_lora_rank: int | None = None
    v_head_dim: int | None = None
    quantization: QuantizationArgs | None = None
    quantization_config: QuantizationConfigMixin | None = None
    yarn: YarnArgs | None = None
    moe: MOEModelArgs | None = None
    vision_encoder: VisionEncoderArgs | None = None

    def __post_init__(self) -> None:
        if self.quantization is not None and self.quantization_config is not None:
            raise ValueError(
                "Cannot set both `quantization` (native tensor format) and `quantization_config` (HF format) "
                "at the same time. Use one or the other."
            )

    def to_hf_config(self) -> MistralHFConfigType:
        """Convert this native config to the corresponding HF config."""
        return _native_config_to_hf_config(self)


def _native_config_to_hf_config(native_config: MistralNativeConfig) -> MistralHFConfigType:
    """Map Mistral config to the correct model config in Transformers.

    The mapping is the following:
    - If it has vision, it is mapped to Mistral 3 that will resolve sub text config using the same function.
    - If it is a MOE, it should also be MLA and it is mapped to Mistral 4.
    - If it has yarn, it is mapped to Ministral 3 else to Mistral.
    """
    is_moe = native_config.moe is not None
    is_mla = native_config.q_lora_rank is not None
    has_vision = native_config.vision_encoder is not None
    has_yarn = native_config.yarn is not None

    if not (is_moe == is_mla):
        raise ValueError("MOE and MLA config are only supported together. Please ensure to have a valid model config.")

    match has_vision, is_moe, has_yarn:
        case True, _, _:
            return _native_config_to_mistral3(native_config=native_config)
        case False, True, _:
            return _native_config_to_mistral4(native_config=native_config)
        case False, False, True:
            return _native_config_to_ministral3(native_config=native_config)
        case False, False, False:
            return _native_config_to_mistral(native_config=native_config)
        case _:
            raise ValueError("Unknown config.")


def _get_maybe_quant_config(
    is_vision_model: bool, quantization_args: QuantizationArgs | None
) -> AutoQuantizationConfig | None:
    """Build an HF `AutoQuantizationConfig` from native quantization args.

    Args:
        is_vision_model: If `True`, adds vision/projector modules
            to `modules_to_not_convert`.
        quantization_args: Native quantization parameters. Returns `None`
            when this is `None`.

    Returns:
        The HF quantization config, or `None` if no quantization is configured.

    Raises:
        ValueError: If the quantization format is not `FP8_E4M3`.
    """
    if quantization_args is None:
        return None

    modules_to_not_convert = ["lm_head"]
    if is_vision_model:
        modules_to_not_convert += [
            "model.vision_tower",
            "model.multi_modal_projector",
        ]

    match quantization_args.qformat_weight:
        case QFormat.FP8_E4M3:
            activation_scheme = _QUANTIZATION_SCHEME_MAP.get(quantization_args.qscheme_act)
            if activation_scheme is None:
                raise ValueError(f"invalid quantization config {quantization_args.qscheme_act=}.")
            quantization_config = {
                "activation_scheme": activation_scheme,
                "modules_to_not_convert": modules_to_not_convert,
                "quant_method": "fp8",
                "weight_block_size": None,
            }
            return AutoQuantizationConfig.from_dict(quantization_config)
        case _:
            raise ValueError(f"invalid quantization config {quantization_args.qformat_weight=}.")


def _get_rope_parameters(
    rope_theta: float,
    yarn_args: YarnArgs | None,
    llama4_scaling: Llama4Scaling | None,
    qk_rope: float | None,
    qk_nope: float | None,
) -> RopeParameters:
    """Build a `RopeParameters` dict from native rope/yarn/MLA fields.

    Raises:
        ValueError: If `qk_rope` and `qk_nope` are not both set or both `None`.
    """
    if (qk_rope is None) != (qk_nope is None):
        raise ValueError(f"qk_rope and qk_nope must both be None or both set, got {qk_rope=}, {qk_nope=}")
    rope_kwargs = {}

    if qk_rope is not None and qk_nope is not None:
        rope_kwargs["partial_rotary_factor"] = qk_rope / (qk_nope + qk_rope)

    if yarn_args is None:
        return RopeParameters(rope_type="default", rope_theta=rope_theta, **rope_kwargs)
    elif llama4_scaling is not None:
        if yarn_args.original_max_position_embeddings != llama4_scaling.original_max_position_embeddings:
            raise ValueError(
                f"yarn.original_max_position_embeddings ({yarn_args.original_max_position_embeddings}) "
                f"must match llama_4_scaling.original_max_position_embeddings "
                f"({llama4_scaling.original_max_position_embeddings})"
            )
        rope_kwargs["llama_4_scaling_beta"] = llama4_scaling.beta

    mscale = 0.0 if yarn_args.apply_scale else 1.0
    mscale_all_dim = mscale

    return RopeParameters(
        rope_type="yarn",
        rope_theta=rope_theta,
        factor=float(yarn_args.factor),
        original_max_position_embeddings=yarn_args.original_max_position_embeddings,
        beta_fast=float(yarn_args.beta),
        beta_slow=float(yarn_args.alpha),
        mscale=mscale,
        mscale_all_dim=mscale_all_dim,
        **rope_kwargs,
    )


def _native_config_to_mistral(native_config: MistralNativeConfig) -> MistralConfig:
    """Convert a native config to a base `MistralConfig`.

    Raises:
        ValueError: If `llama_4_scaling` or `yarn` are set (use
            `_native_config_to_ministral3` instead).
    """
    if native_config.llama_4_scaling is not None or native_config.yarn is not None:
        raise ValueError("Base Mistral config must not have llama_4_scaling or yarn set")
    quant_config = native_config.quantization_config or _get_maybe_quant_config(
        is_vision_model=False, quantization_args=native_config.quantization
    )
    rope_parameters = _get_rope_parameters(
        rope_theta=native_config.rope_theta,
        yarn_args=native_config.yarn,
        llama4_scaling=native_config.llama_4_scaling,
        qk_rope=None,
        qk_nope=None,
    )

    optional_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        optional_kwargs["quantization_config"] = quant_config

    return MistralConfig(
        hidden_size=native_config.dim,
        num_hidden_layers=native_config.n_layers,
        intermediate_size=native_config.hidden_dim,
        num_attention_heads=native_config.n_heads,
        rms_norm_eps=native_config.norm_eps,
        head_dim=native_config.head_dim,
        vocab_size=native_config.vocab_size,
        num_key_value_heads=native_config.n_kv_heads,
        rope_parameters=rope_parameters,
        sliding_window=native_config.sliding_window,
        max_position_embeddings=native_config.max_position_embeddings,
        tie_word_embeddings=native_config.tied_embeddings,
        **optional_kwargs,
    )


def _native_config_to_ministral3(native_config: MistralNativeConfig) -> Ministral3Config:
    """Convert a native config to a `Ministral3Config`.

    Raises:
        ValueError: If `yarn` parameters are missing.
    """
    if native_config.yarn is None:
        raise ValueError("Ministral3 config requires yarn parameters")
    quant_config = native_config.quantization_config or _get_maybe_quant_config(
        is_vision_model=False, quantization_args=native_config.quantization
    )
    rope_parameters = _get_rope_parameters(
        rope_theta=native_config.rope_theta,
        yarn_args=native_config.yarn,
        llama4_scaling=native_config.llama_4_scaling,
        qk_rope=None,
        qk_nope=None,
    )

    optional_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        optional_kwargs["quantization_config"] = quant_config

    return Ministral3Config(
        hidden_size=native_config.dim,
        num_hidden_layers=native_config.n_layers,
        intermediate_size=native_config.hidden_dim,
        num_attention_heads=native_config.n_heads,
        rms_norm_eps=native_config.norm_eps,
        head_dim=native_config.head_dim,
        vocab_size=native_config.vocab_size,
        num_key_value_heads=native_config.n_kv_heads,
        sliding_window=native_config.sliding_window,
        max_position_embeddings=native_config.max_position_embeddings,
        tie_word_embeddings=native_config.tied_embeddings,
        rope_parameters=rope_parameters,
        **optional_kwargs,
    )


def _native_config_to_mistral4(native_config: MistralNativeConfig) -> Mistral4Config:
    """Convert a native config to a `Mistral4Config` (MLA + MoE architecture)."""
    quant_config = native_config.quantization_config or _get_maybe_quant_config(
        is_vision_model=False, quantization_args=native_config.quantization
    )
    rope_parameters = _get_rope_parameters(
        rope_theta=native_config.rope_theta,
        yarn_args=native_config.yarn,
        llama4_scaling=native_config.llama_4_scaling,
        qk_rope=native_config.qk_rope_head_dim,
        qk_nope=native_config.qk_nope_head_dim,
    )

    optional_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        optional_kwargs["quantization_config"] = quant_config

    return Mistral4Config(
        hidden_size=native_config.dim,
        num_hidden_layers=native_config.n_layers,
        intermediate_size=native_config.hidden_dim,
        num_attention_heads=native_config.n_heads,
        num_key_value_heads=native_config.n_kv_heads,
        rms_norm_eps=native_config.norm_eps,
        vocab_size=native_config.vocab_size,
        tie_word_embeddings=native_config.tied_embeddings,
        sliding_window=native_config.sliding_window,
        max_position_embeddings=native_config.max_position_embeddings,
        q_lora_rank=native_config.q_lora_rank,
        qk_rope_head_dim=native_config.qk_rope_head_dim,
        qk_nope_head_dim=native_config.qk_nope_head_dim,
        kv_lora_rank=native_config.kv_lora_rank,
        v_head_dim=native_config.v_head_dim,
        n_routed_experts=native_config.moe.num_experts,
        num_experts_per_tok=native_config.moe.num_experts_per_tok,
        first_k_dense_replace=native_config.moe.first_k_dense_replace,
        n_shared_experts=native_config.moe.num_shared_experts,
        moe_intermediate_size=native_config.moe.expert_hidden_dim,
        routed_scaling_factor=native_config.moe.routed_scale,
        n_group=native_config.moe.num_expert_groups,
        topk_group=native_config.moe.num_expert_groups_per_tok,
        norm_topk_prob=True,
        rope_parameters=rope_parameters,
        **optional_kwargs,
    )


def _native_config_to_mistral3(native_config: MistralNativeConfig) -> Mistral3Config:
    """Convert a native config to a `Mistral3Config` (vision-language model).

    Raises:
        ValueError: If `vision_encoder` parameters are missing.
    """
    if native_config.vision_encoder is None:
        raise ValueError("Mistral3 config requires vision_encoder parameters")
    vision_config = native_config.vision_encoder
    vision_hf = PixtralVisionConfig(
        hidden_size=vision_config.hidden_size,
        num_hidden_layers=vision_config.num_hidden_layers,
        num_attention_heads=vision_config.num_attention_heads,
        patch_size=vision_config.patch_size,
        image_size=vision_config.image_size,
        intermediate_size=vision_config.intermediate_size,
        num_channels=vision_config.num_channels,
        hidden_act="silu",
        rope_theta=vision_config.rope_theta,
    )
    quant_config = native_config.quantization_config or _get_maybe_quant_config(
        is_vision_model=True, quantization_args=native_config.quantization
    )

    native_text_config = copy.deepcopy(native_config)
    native_text_config.vision_encoder = None
    native_text_config.quantization = None
    native_text_config.quantization_config = None

    text_hf = _native_config_to_hf_config(native_config=native_text_config)

    optional_kwargs: dict[str, Any] = {}
    if quant_config is not None:
        optional_kwargs["quantization_config"] = quant_config

    return Mistral3Config(
        vision_config=vision_hf,
        text_config=text_hf,
        multimodal_projector_bias=vision_config.adapter_bias,
        image_token_id=vision_config.image_token_id,
        spatial_merge_size=vision_config.spatial_merge_size,
        vision_feature_layer=-1,
        tie_word_embeddings=native_config.tied_embeddings,
        **optional_kwargs,
    )


def _extract_hf_quantization_config(hf_config: PreTrainedConfig) -> QuantizationConfigMixin | None:
    """Extract the `quantization_config` attribute from an HF config, if present."""
    quant_cfg = getattr(hf_config, "quantization_config", None)
    if quant_cfg is None:
        return None
    if isinstance(quant_cfg, QuantizationConfigMixin):
        return quant_cfg
    if isinstance(quant_cfg, dict):
        return AutoQuantizationConfig.from_dict(quant_cfg)
    return None


def _hf_quant_config_to_native(hf_config: PreTrainedConfig) -> QuantizationArgs | None:
    """Convert an HF quantization config back to native `QuantizationArgs`."""
    quant_cfg = _extract_hf_quantization_config(hf_config)
    if quant_cfg is None:
        return None
    qc = quant_cfg.to_dict()
    if qc.get("quant_method") != "fp8":
        return None
    scheme = _REVERSE_QUANTIZATION_SCHEME_MAP.get(qc.get("activation_scheme", "static"), "TENSOR")
    return QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act=scheme)


def _resolve_hf_quant_for_native(
    hf_config: PreTrainedConfig,
) -> tuple[QuantizationArgs | None, QuantizationConfigMixin | None]:
    """Resolve an HF quantization config to native fields.

    Tries the direct reverse mapping first. If that is not feasible (e.g.
    non-fp8 quant method), the original HF config is preserved as-is in
    `quantization_config` so it is not silently dropped.

    Returns:
        A `(quantization, quantization_config)` pair where exactly one
        (or neither) is non-`None`.
    """
    native_quant = _hf_quant_config_to_native(hf_config)
    if native_quant is not None:
        return native_quant, None
    hf_quant = _extract_hf_quantization_config(hf_config)
    return None, hf_quant


def _extract_llama4_scaling_from_rope_params(rope_params: dict | RopeParameters | None) -> Llama4Scaling | None:
    """Extract `Llama4Scaling` from an HF `rope_parameters` dict, if present."""
    if (
        not rope_params
        or not isinstance(rope_params, dict)
        or (beta := rope_params.get("llama_4_scaling_beta")) is None
    ):
        return None

    if (original_max_position_embeddings := rope_params.get("original_max_position_embeddings")) is None:
        raise ValueError("original_max_position_embeddings should not be None if llama4 scaling is set.")

    return Llama4Scaling(
        original_max_position_embeddings=int(original_max_position_embeddings),
        beta=float(beta),
    )


def _hf_config_to_native_config(hf_config: MistralHFConfigType) -> MistralNativeConfig:
    """Dispatch an HF config to the appropriate reverse converter based on its type.

    Raises:
        ValueError: If the HF config type is not supported.
    """
    # Order matters
    match hf_config:
        case Mistral3Config():
            return _hf_mistral3_to_native(hf_config)
        case Mistral4Config():
            return _hf_mistral4_to_native(hf_config)
        case Ministral3Config():
            return _hf_ministral3_to_native(hf_config)
        case MistralConfig():
            return _hf_mistral_to_native(hf_config)
        case _:
            raise ValueError(f"Unsupported HF config type: {type(hf_config).__name__}")


def _hf_mistral_to_native(hf_config: MistralConfig) -> MistralNativeConfig:
    """Convert a `MistralConfig` back to a `MistralNativeConfig`.

    Raises:
        ValueError: If `head_dim` is not set on the HF config.
    """
    if hf_config.head_dim is None:
        raise ValueError("head_dim must be set on the HF config")
    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)

    return MistralNativeConfig(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        head_dim=hf_config.head_dim,
        hidden_dim=hf_config.intermediate_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        rope_theta=_extract_rope_theta(hf_config),
        norm_eps=hf_config.rms_norm_eps,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        sliding_window=hf_config.sliding_window,
        tied_embeddings=hf_config.tie_word_embeddings,
        yarn=_extract_yarn(hf_config),
        quantization=quantization,
        quantization_config=quantization_config,
    )


def _hf_ministral3_to_native(hf_config: Ministral3Config) -> MistralNativeConfig:
    """Convert a `Ministral3Config` back to a `MistralNativeConfig`."""
    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)

    return MistralNativeConfig(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        head_dim=hf_config.head_dim,
        hidden_dim=hf_config.intermediate_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        rope_theta=_extract_rope_theta(hf_config),
        norm_eps=hf_config.rms_norm_eps,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        sliding_window=hf_config.sliding_window,
        tied_embeddings=hf_config.tie_word_embeddings,
        yarn=_extract_yarn(hf_config),
        llama_4_scaling=_extract_llama4_scaling_from_rope_params(
            getattr(hf_config, "rope_parameters", None),
        ),
        quantization=quantization,
        quantization_config=quantization_config,
    )


def _hf_mistral4_to_native(hf_config: Mistral4Config) -> MistralNativeConfig:
    """Convert a `Mistral4Config` (MLA + MoE) back to a `MistralNativeConfig`."""
    rope_params = getattr(hf_config, "rope_parameters", None)
    required_fields = {
        "num_key_value_heads": hf_config.num_key_value_heads,
        "num_experts_per_tok": hf_config.num_experts_per_tok,
        "first_k_dense_replace": hf_config.first_k_dense_replace,
        "n_group": hf_config.n_group,
        "topk_group": hf_config.topk_group,
    }
    missing = [name for name, value in required_fields.items() if value is None]
    if missing:
        raise ValueError(f"Mistral4 config requires non-None fields: {missing}")
    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)

    return MistralNativeConfig(
        dim=hf_config.hidden_size,
        n_layers=hf_config.num_hidden_layers,
        head_dim=hf_config.qk_nope_head_dim + hf_config.qk_rope_head_dim,
        hidden_dim=hf_config.intermediate_size,
        n_heads=hf_config.num_attention_heads,
        n_kv_heads=hf_config.num_key_value_heads,
        rope_theta=_extract_rope_theta(hf_config),
        norm_eps=hf_config.rms_norm_eps,
        vocab_size=hf_config.vocab_size,
        max_position_embeddings=hf_config.max_position_embeddings,
        sliding_window=getattr(hf_config, "sliding_window", None),
        tied_embeddings=hf_config.tie_word_embeddings,
        q_lora_rank=hf_config.q_lora_rank,
        qk_rope_head_dim=hf_config.qk_rope_head_dim,
        qk_nope_head_dim=hf_config.qk_nope_head_dim,
        kv_lora_rank=hf_config.kv_lora_rank,
        v_head_dim=hf_config.v_head_dim,
        yarn=_extract_yarn(hf_config),
        llama_4_scaling=_extract_llama4_scaling_from_rope_params(rope_params),
        moe=MOEModelArgs(
            num_experts=hf_config.n_routed_experts,
            num_experts_per_tok=hf_config.num_experts_per_tok,
            first_k_dense_replace=hf_config.first_k_dense_replace,
            num_shared_experts=hf_config.n_shared_experts,
            expert_hidden_dim=hf_config.moe_intermediate_size,
            routed_scale=hf_config.routed_scaling_factor,
            num_expert_groups=hf_config.n_group,
            num_expert_groups_per_tok=hf_config.topk_group,
        ),
        quantization=quantization,
        quantization_config=quantization_config,
    )


def _hf_mistral3_to_native(hf_config: Mistral3Config) -> MistralNativeConfig:
    """Convert a `Mistral3Config` (VLM) back to a `MistralNativeConfig`.

    The text sub-config is recursively converted via `_hf_config_to_native_config`.
    """
    text_native = _hf_config_to_native_config(hf_config.text_config)
    vision_hf: PixtralVisionConfig = hf_config.vision_config

    vision_encoder = VisionEncoderArgs(
        hidden_size=vision_hf.hidden_size,
        num_hidden_layers=vision_hf.num_hidden_layers,
        num_attention_heads=vision_hf.num_attention_heads,
        patch_size=vision_hf.patch_size,
        image_size=vision_hf.image_size,
        intermediate_size=vision_hf.intermediate_size,
        num_channels=vision_hf.num_channels,
        rope_theta=_extract_rope_theta(vision_hf),
        adapter_bias=hf_config.multimodal_projector_bias,
        spatial_merge_size=hf_config.spatial_merge_size,
        image_token_id=hf_config.image_token_id,
        # These token IDs match the hardcoded positions in Mistral's tekken.json
        # vocabulary: [IMG_BREAK] = 12, [IMG_END] = 13. They have no HF config
        # counterpart and should ideally be derived from the tokenizer at runtime.
        image_break_token_id=12,
        image_end_token_id=13,
        mm_projector_id="patch_merge",
        max_image_size=vision_hf.image_size,
        add_pre_mm_projector_layer_norm=True,
    )

    quantization, quantization_config = _resolve_hf_quant_for_native(hf_config)
    text_native.vision_encoder = vision_encoder
    text_native.quantization = quantization
    text_native.quantization_config = quantization_config
    return text_native


def _parse_native_config_from_dict(params: dict) -> MistralNativeConfig:
    """Parse a raw `params.json` dict into a `MistralNativeConfig` dataclass.

    Handles optional sub-structures (yarn, llama_4_scaling, quantization,
    moe, vision_encoder) and raises `KeyError` for missing required fields.
    """
    yarn_dict = params.get("yarn")
    yarn = (
        YarnArgs(
            factor=yarn_dict["factor"],
            original_max_position_embeddings=yarn_dict["original_max_position_embeddings"],
            beta=yarn_dict["beta"],
            alpha=yarn_dict["alpha"],
            apply_scale=yarn_dict.get("apply_scale", False),
        )
        if yarn_dict is not None
        else None
    )

    llama4_dict = params.get("llama_4_scaling")
    llama_4_scaling = (
        Llama4Scaling(
            original_max_position_embeddings=llama4_dict["original_max_position_embeddings"],
            beta=llama4_dict["beta"],
        )
        if llama4_dict is not None
        else None
    )

    quant_dict = params.get("quantization")
    quantization = (
        QuantizationArgs(qformat_weight=quant_dict["qformat_weight"], qscheme_act=quant_dict["qscheme_act"])
        if quant_dict is not None
        else None
    )

    moe_dict = params.get("moe")
    moe = (
        MOEModelArgs(
            num_experts=moe_dict["num_experts"],
            num_experts_per_tok=moe_dict["num_experts_per_tok"],
            expert_hidden_dim=moe_dict["expert_hidden_dim"],
            first_k_dense_replace=moe_dict["first_k_dense_replace"],
            num_shared_experts=moe_dict["num_shared_experts"],
            routed_scale=moe_dict["routed_scale"],
            num_expert_groups=moe_dict["num_expert_groups"],
            num_expert_groups_per_tok=moe_dict["num_expert_groups_per_tok"],
            expert_parallel=moe_dict.get("expert_parallel", 1),
            expert_model_parallel=moe_dict.get("expert_model_parallel", 1),
            route_every_n=moe_dict.get("route_every_n", 1),
        )
        if moe_dict is not None
        else None
    )

    vision_dict = params.get("vision_encoder")
    vision_encoder = (
        VisionEncoderArgs(
            hidden_size=vision_dict["hidden_size"],
            num_hidden_layers=vision_dict["num_hidden_layers"],
            num_attention_heads=vision_dict["num_attention_heads"],
            patch_size=vision_dict["patch_size"],
            image_size=vision_dict["image_size"],
            intermediate_size=vision_dict["intermediate_size"],
            num_channels=vision_dict["num_channels"],
            max_image_size=vision_dict["max_image_size"],
            rope_theta=vision_dict["rope_theta"],
            mm_projector_id=vision_dict["mm_projector_id"],
            add_pre_mm_projector_layer_norm=vision_dict["add_pre_mm_projector_layer_norm"],
            adapter_bias=vision_dict["adapter_bias"],
            spatial_merge_size=vision_dict["spatial_merge_size"],
            image_token_id=vision_dict["image_token_id"],
            image_break_token_id=vision_dict["image_break_token_id"],
            image_end_token_id=vision_dict["image_end_token_id"],
        )
        if vision_dict is not None
        else None
    )

    return MistralNativeConfig(
        dim=params["dim"],
        n_layers=params["n_layers"],
        head_dim=params["head_dim"],
        hidden_dim=params["hidden_dim"],
        n_heads=params["n_heads"],
        n_kv_heads=params["n_kv_heads"],
        rope_theta=params["rope_theta"],
        norm_eps=params["norm_eps"],
        vocab_size=params["vocab_size"],
        max_position_embeddings=params.get("max_position_embeddings"),
        sliding_window=params.get("sliding_window"),
        tied_embeddings=params.get("tied_embeddings", False),
        q_lora_rank=params.get("q_lora_rank"),
        qk_rope_head_dim=params.get("qk_rope_head_dim"),
        qk_nope_head_dim=params.get("qk_nope_head_dim"),
        kv_lora_rank=params.get("kv_lora_rank"),
        v_head_dim=params.get("v_head_dim"),
        yarn=yarn,
        llama_4_scaling=llama_4_scaling,
        quantization=quantization,
        moe=moe,
        vision_encoder=vision_encoder,
    )


def native_config_for_model_type(model_type: str, params: dict) -> MistralNativeConfig:
    """Build a `MistralNativeConfig` from a raw `params.json` dict.

    The `model_type` is validated against known Mistral model types but the
    construction itself is model-type agnostic since `MistralNativeConfig` is
    a single generic dataclass.

    Args:
        model_type (`str`): One of `"mistral"`, `"ministral3"`, `"mistral4"`,
            `"mistral3"`.
        params (`dict`): Raw key/value pairs from a Mistral `params.json` file.

    Raises:
        ValueError: If `model_type` is unknown.

    Returns:
        The parsed native config.
    """
    try:
        MistralModelType(model_type)
    except ValueError:
        raise ValueError(
            f"Unknown Mistral model type {model_type!r}. Supported types: {[m.value for m in MistralModelType]}"
        ) from None
    return _parse_native_config_from_dict(params)


def native_config_from_hf_config(model_type: str, hf_config: MistralHFConfigType) -> MistralNativeConfig:
    """Convert an HF config to a `MistralNativeConfig`, dispatched by model type.

    This is a thin wrapper around :func:`hf_config_to_native_config` that
    accepts a `model_type` string for compatibility with the config-format
    integration layer.

    Args:
        model_type (`str`): One of `"mistral"`, `"ministral3"`, `"mistral4"`,
            `"mistral3"`.
        hf_config (`MistralHFConfigType`): The HuggingFace config to convert.

    Raises:
        ValueError: If `model_type` is unknown or the config type is
            unsupported.

    Returns:
        The converted native config.
    """
    try:
        MistralModelType(model_type)
    except ValueError:
        raise ValueError(
            f"Unknown Mistral model type {model_type!r}. Supported types: {[m.value for m in MistralModelType]}"
        ) from None
    return _hf_config_to_native_config(hf_config)
