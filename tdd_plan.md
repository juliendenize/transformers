# TDD Plan: Automatic HF / Mistral Format Conversion

## Goal

Make `from_pretrained` and `save_pretrained` automatically detect and convert between
HuggingFace and Mistral native formats. Support `mistral`, `ministral3`, `mistral3`
(composite VLM), and `mistral4` (MoE/MLA) model types. No separate conversion scripts needed.

## Current State

The branch is at the merge-base with main (`4f28719324`). All prior implementation
lives in git stashes (`stash@{0}` has source files, `stash@{1}` has tests + fixes).
These serve as reference for the implementation design but should NOT be restored
directly — tests and code will be written from scratch following this plan.

## Format Detection Rules

| Signal | HF Format | Mistral Native Format |
|--------|-----------|----------------------|
| Config | `config.json` | `params.json` |
| Weights | `model.safetensors` / `model-NNNNN-of-MMMMM.safetensors` | `consolidated.safetensors` / `consolidated-NNNNN-of-MMMMM.safetensors` |
| Weight index | `model.safetensors.index.json` | `consolidated.safetensors.index.json` |
| Tokenizer | `tokenizer.json` + `tokenizer_config.json` | `tekken.json` |
| FP8 scales | `.weight_scale_inv` / `.activation_scale` | `.qscale_weight` / `.qscale_act` |

**Detection priority**: When both formats exist, prefer HF. Only fall back to native
when HF files are absent.

**API**: Fully automatic detection based on file presence. User can override with
`mistral_format=True` (force native) or `mistral_format=False` (force HF, error if absent).

**Save format**: By default, save in the same format as loaded. User can override with
`save_format="hf"` or `save_format="mistral"`.

---

## TDD Approach

For each phase:
1. **Write all tests** for that phase
2. **Run tests**, verify they fail (ImportError or AssertionError)
3. **Implement** until all tests pass
4. **Verify** no regressions on prior phases

Phases are sequential — each builds on the previous. Within each phase, ALL tests
are written before ANY implementation begins.

### Phase Overview

| Phase | What | Test File | Implementation Files |
|-------|------|-----------|---------------------|
| 1 | Native config dataclasses (pure Python) | `test_params_conversion.py` | `params_conversion.py`, `__init__.py` |
| 2 | Weight key conversion (torch) | `test_weight_conversion.py` | `weight_conversion.py` |
| 3 | PermuteForRope fixes | `test_weight_conversion.py` (appended) | `core_model_loading.py` |
| 4 | Config format detection | `test_config_format.py` | `config_format.py`, config classes |
| 5 | Full from_pretrained pipeline | `test_integration.py` | `conversion_mapping.py` |
| 6 | save_pretrained native format | `test_integration.py` (appended) | `modeling_utils.py`, `core_model_loading.py` |
| 7 | Slow integration tests | `test_slow_integration.py` | None (validation only) |

---

## Phase 1: Native Config Dataclasses (Pure Python, No Torch)

### What We're Building

A typed dataclass hierarchy representing Mistral native `params.json` configs, with
methods to convert to/from HF config objects. Pure Python, no torch dependency.

Every field that exists in `params.json` is an explicit dataclass field — there is no
passthrough mechanism. Fields that share the same name in native and HF format are
still explicit (e.g., `vocab_size` is a field on both the native dataclass and the
HF config). Conversion uses direct field reading in `from_params_json()` and explicit
mapping in `to_hf_config()` / `from_hf_config()`.

**File**: `src/transformers/integrations/mistral/params_conversion.py`

### Data Model

#### Sub-Config Dataclasses

```python
@dataclass
class YarnNativeConfig:
    factor: float
    original_max_position_embeddings: int
    beta: float     # maps to beta_fast in HF rope_parameters
    alpha: float    # maps to beta_slow in HF rope_parameters

@dataclass
class FP8NativeConfig:
    qformat_weight: str   # must be "fp8_e4m3", validated in __post_init__
    qscheme_act: str      # "TENSOR" → static, "DYNAMIC" → dynamic

@dataclass
class MoeNativeConfig:
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
```

#### Top-Level Config Dataclasses (Inheritance)

```python
@dataclass
class MistralNativeConfig:
    r"""Base: params.json for mistral models."""
    # Required
    dim: int
    n_layers: int
    hidden_dim: int
    n_heads: int
    norm_eps: float
    head_dim: int
    vocab_size: int
    # Optional
    n_kv_heads: int | None = None
    rope_theta: float = 10000.0
    sliding_window: int | None = None
    max_position_embeddings: int = 32768
    tied_embeddings: bool = False

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Parse a params.json dict into this dataclass. Reads fields directly."""
        return cls(
            dim=params["dim"],
            n_layers=params["n_layers"],
            ...
            n_kv_heads=params.get("n_kv_heads"),
            rope_theta=params.get("rope_theta", 10000.0),
        )

    def to_hf_config(self) -> MistralConfig:
        r"""Convert to HF MistralConfig object."""
        return MistralConfig(
            hidden_size=self.dim,
            num_hidden_layers=self.n_layers,
            intermediate_size=self.hidden_dim,
            num_attention_heads=self.n_heads,
            rms_norm_eps=self.norm_eps,
            head_dim=self.head_dim,
            vocab_size=self.vocab_size,
            num_key_value_heads=self.n_kv_heads,
            rope_theta=self.rope_theta,
            sliding_window=self.sliding_window,
            max_position_embeddings=self.max_position_embeddings,
            tie_word_embeddings=self.tied_embeddings,
        )

    @classmethod
    def from_hf_config(cls, config: MistralConfig) -> Self:
        r"""Reverse: HF config → native config (for save)."""
        return cls(
            dim=config.hidden_size,
            n_layers=config.num_hidden_layers,
            ...
        )

@dataclass
class Ministral3NativeConfig(MistralNativeConfig):
    r"""Extends base with YaRN RoPE and FP8 quantization."""
    yarn: YarnNativeConfig | None = None
    quantization: FP8NativeConfig | None = None

    def to_hf_config(self) -> Ministral3Config: ...
    @classmethod
    def from_hf_config(cls, config: Ministral3Config) -> Self: ...

@dataclass
class Mistral4NativeConfig(MistralNativeConfig):
    r"""Extends base with MoE/MLA architecture fields."""
    q_lora_rank: int | None = None
    qk_rope_head_dim: int | None = None
    qk_nope_head_dim: int | None = None
    kv_lora_rank: int | None = None
    v_head_dim: int | None = None
    moe: MoeNativeConfig | None = None
    yarn: YarnNativeConfig | None = None

    def to_hf_config(self) -> Mistral4Config: ...
    @classmethod
    def from_hf_config(cls, config: Mistral4Config) -> Self: ...

@dataclass
class Mistral3NativeConfig:
    r"""Composite VLM: wraps a text backbone config + vision encoder config."""
    text_config: MistralNativeConfig | Ministral3NativeConfig | Mistral4NativeConfig
    vision_encoder: VisionEncoderNativeConfig

    @classmethod
    def from_params_json(cls, params: dict) -> Self:
        r"""Auto-detects text backbone type based on moe/yarn/quantization presence."""
        ...

    def to_hf_config(self) -> Mistral3Config: ...
    @classmethod
    def from_hf_config(cls, config: Mistral3Config) -> Self: ...
```

#### Dispatcher Functions

```python
def native_config_for_model_type(model_type: str, params: dict) -> MistralNativeConfig | Mistral3NativeConfig:
    r"""Dispatch to the correct native config class by model_type."""
    match model_type:
        case "mistral":  return MistralNativeConfig.from_params_json(params)
        case "ministral3": return Ministral3NativeConfig.from_params_json(params)
        case "mistral4": return Mistral4NativeConfig.from_params_json(params)
        case "mistral3": return Mistral3NativeConfig.from_params_json(params)
        case _: raise ValueError(f"Unknown model type: {model_type!r}")

def native_config_from_hf_config(model_type: str, config) -> MistralNativeConfig | Mistral3NativeConfig:
    r"""Reverse dispatch: HF config → native config."""
    match model_type:
        case "mistral":  return MistralNativeConfig.from_hf_config(config)
        case "ministral3": return Ministral3NativeConfig.from_hf_config(config)
        case "mistral4": return Mistral4NativeConfig.from_hf_config(config)
        case "mistral3": return Mistral3NativeConfig.from_hf_config(config)
        case _: raise ValueError(f"Unknown model type: {model_type!r}")
```

### Tests to Write First

**File**: `tests/integrations/mistral/test_params_conversion.py`

All tests are pure Python (no `@require_torch`, no Hub access). Tests use `unittest.TestCase`.

**Imports needed in test file**:
```python
from transformers.integrations.mistral.params_conversion import (
    FP8NativeConfig,
    MistralNativeConfig,
    Ministral3NativeConfig,
    Mistral3NativeConfig,
    Mistral4NativeConfig,
    MoeNativeConfig,
    VisionEncoderNativeConfig,
    YarnNativeConfig,
    native_config_for_model_type,
    native_config_from_hf_config,
)
```

**Fixture requirements**: Define module-level dicts for each model type's `params.json` format:

- `_MISTRAL_BASE_FIELDS`: Shared required fields (`dim`, `n_layers`, `hidden_dim`, `n_heads`, `norm_eps`, `head_dim`, `vocab_size`)
- `_MISTRAL_PARAMS`: Base fields + `n_kv_heads`, `rope_theta`, `sliding_window`, `max_position_embeddings`
- `_MINISTRAL3_PARAMS`: Base fields + `n_kv_heads`, `rope_theta=1e6`, `max_position_embeddings=262144`, `tied_embeddings=True`, `yarn` sub-dict (factor, original_max_position_embeddings, beta, alpha), `quantization` sub-dict (qformat_weight, qscheme_act)
- `_MISTRAL4_PARAMS`: Base fields + MLA fields (`q_lora_rank`, `qk_rope_head_dim`, `qk_nope_head_dim`, `kv_lora_rank`, `v_head_dim`), `moe` sub-dict (num_experts, num_experts_per_tok, first_k_dense_replace, num_shared_experts, expert_hidden_dim, routed_scale, num_expert_groups, num_expert_groups_per_tok), `yarn` sub-dict
- `_MISTRAL3_PARAMS`: Base fields + `n_kv_heads`, `rope_theta`, `vision_encoder` sub-dict (hidden_size, num_hidden_layers, num_attention_heads, patch_size, image_size, head_dim, intermediate_size, adapter_bias, spatial_merge_size, image_token_id)

#### 1.1 Dataclass Construction from params.json (7 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 1 | `test_mistral_native_config_from_params_json` | `MistralNativeConfig.from_params_json(_MISTRAL_PARAMS)` | `config.dim == 4096`, `config.n_layers == 32`, `config.hidden_dim == 14336`, `config.n_heads == 32`, `config.n_kv_heads == 8`, `config.norm_eps == 1e-5`, `config.head_dim == 128`, `config.vocab_size == 32000`, `config.rope_theta == 10000.0`, `config.sliding_window == 4096` |
| 2 | `test_mistral_native_config_minimal_required` | `MistralNativeConfig.from_params_json(_MISTRAL_BASE_FIELDS)` (only required fields) | No error; optional fields at defaults: `config.n_kv_heads is None`, `config.rope_theta == 10000.0`, `config.sliding_window is None` |
| 3 | `test_mistral_native_config_missing_required_raises` | `MistralNativeConfig.from_params_json(params)` with `dim` absent | `assertRaises(KeyError)` |
| 4 | `test_ministral3_native_config_from_params_json` | `Ministral3NativeConfig.from_params_json(_MINISTRAL3_PARAMS)` | `config.tied_embeddings is True`; `config.yarn` is `YarnNativeConfig` with `factor == 16.0`; `config.quantization` is `FP8NativeConfig` with `qformat_weight == "fp8_e4m3"` |
| 5 | `test_mistral4_native_config_from_params_json` | `Mistral4NativeConfig.from_params_json(_MISTRAL4_PARAMS)` | `config.q_lora_rank == 1024`; `config.moe` is `MoeNativeConfig` with `num_experts == 128`, `num_experts_per_tok == 4`, `expert_hidden_dim == 2048` |
| 6 | `test_mistral3_native_config_from_params_json` | `Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS)` | `config.vision_encoder` is `VisionEncoderNativeConfig` with `hidden_size == 1024`; `config.text_config` is an instance of `MistralNativeConfig` |
| 7 | `test_params_json_ignores_unknown_keys` | `MistralNativeConfig.from_params_json({**_MISTRAL_BASE_FIELDS, "unknown_field": 999})` | No error; constructed config has no `unknown_field` attribute (or `hasattr` returns False) |

#### 1.2 to_hf_config() Conversion (5 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 8 | `test_mistral_to_hf_config` | `MistralNativeConfig.from_params_json(_MISTRAL_PARAMS).to_hf_config()` | Returns `MistralConfig`; `hf.hidden_size == 4096`, `hf.num_hidden_layers == 32`, `hf.intermediate_size == 14336`, `hf.num_attention_heads == 32`, `hf.num_key_value_heads == 8`, `hf.rms_norm_eps == 1e-5`, `hf.head_dim == 128`, `hf.vocab_size == 32000`, `hf.sliding_window == 4096` |
| 9 | `test_ministral3_to_hf_config_with_yarn` | `Ministral3NativeConfig.from_params_json(_MINISTRAL3_PARAMS).to_hf_config()` | Returns `Ministral3Config`; `hf.rope_parameters` is not None; `hf.rope_parameters["type"] == "yarn"` or rope type is yarn; `hf.tie_word_embeddings is True` |
| 10 | `test_ministral3_to_hf_config_with_fp8` | `Ministral3NativeConfig` with `quantization=FP8NativeConfig("fp8_e4m3", "TENSOR")` → `.to_hf_config()` | `hf.quantization_config` present with `quant_method == "fp8"`, `activation_scheme == "static"` |
| 11 | `test_mistral4_to_hf_config_with_moe` | `Mistral4NativeConfig.from_params_json(_MISTRAL4_PARAMS).to_hf_config()` | Returns `Mistral4Config`; `hf.n_routed_experts == 128`, `hf.num_experts_per_tok == 4`, `hf.moe_intermediate_size == 2048`, `hf.n_shared_experts == 1`, `hf.q_lora_rank == 1024` |
| 12 | `test_mistral3_to_hf_config_composite` | `Mistral3NativeConfig.from_params_json(_MISTRAL3_PARAMS).to_hf_config()` | Returns `Mistral3Config`; `hf.text_config` is a config object; `hf.vision_config` is a config object with `hidden_size == 1024`, `num_attention_heads == 16`; `hf.spatial_merge_size == 2`; `hf.image_token_index` set |

#### 1.3 Sub-Config Validation & Construction (5 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 13 | `test_fp8_native_config_valid` | `FP8NativeConfig("fp8_e4m3", "TENSOR")` | No error; `config.qformat_weight == "fp8_e4m3"`, `config.qscheme_act == "TENSOR"` |
| 14 | `test_fp8_native_config_unsupported_format_raises` | `FP8NativeConfig("int8", "TENSOR")` | `assertRaises(ValueError)` with `"fp8_e4m3"` in message |
| 15 | `test_fp8_native_config_dynamic_scheme` | `FP8NativeConfig("fp8_e4m3", "DYNAMIC")` | No error; `config.qscheme_act == "DYNAMIC"` |
| 16 | `test_moe_native_config_defaults` | `MoeNativeConfig(num_experts=128, num_experts_per_tok=4, expert_hidden_dim=2048)` | `config.first_k_dense_replace == 0`, `config.num_shared_experts == 1`, `config.routed_scale == 1.0` |
| 17 | `test_mistral3_backbone_auto_detection` | Three sub-assertions in one test: (a) params with `vision_encoder` but no `moe`/`yarn` → `text_config` is `MistralNativeConfig`; (b) params with `vision_encoder` + `yarn` → `text_config` is `Ministral3NativeConfig`; (c) params with `vision_encoder` + `moe` + MLA fields → `text_config` is `Mistral4NativeConfig` |

#### 1.4 Reverse: from_hf_config() Roundtrip (4 tests)

Each test converts params → native config → HF config → native config → compare to original.

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 18 | `test_reverse_mistral_roundtrip` | `native = MistralNativeConfig.from_params_json(_MISTRAL_PARAMS)` → `hf = native.to_hf_config()` → `restored = MistralNativeConfig.from_hf_config(hf)` | `restored.dim == native.dim`, `restored.n_layers == native.n_layers`, all base fields match |
| 19 | `test_reverse_ministral3_roundtrip` | Same flow for `Ministral3NativeConfig` (without `quantization` — FP8 doesn't roundtrip via HF config) | `restored.yarn.factor == native.yarn.factor`, base fields match |
| 20 | `test_reverse_mistral4_roundtrip` | Same flow for `Mistral4NativeConfig` | MoE sub-fields match: `restored.moe.num_experts == native.moe.num_experts`, MLA fields match |
| 21 | `test_reverse_mistral3_roundtrip` | Same flow for `Mistral3NativeConfig` | `restored.vision_encoder.hidden_size == native.vision_encoder.hidden_size`, text config type preserved |

#### 1.5 Dispatcher & Error Handling (4 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 22 | `test_native_config_for_model_type_unknown_raises` | `native_config_for_model_type("unknown", {})` | `assertRaises(ValueError)` |
| 23 | `test_native_config_from_hf_config_unknown_raises` | `native_config_from_hf_config("unknown", object())` | `assertRaises(ValueError)` |
| 24 | `test_native_config_for_model_type_dispatches_correctly` | Call with each valid model_type string ("mistral", "ministral3", "mistral4", "mistral3") and matching params | Returns instance of the expected native config class |
| 25 | `test_ministral3_no_quantization_creates_none` | `Ministral3NativeConfig.from_params_json(params)` where params has no `quantization` key | `config.quantization is None` |

**Phase 1 total: 25 tests**

### Implementation

After all 25 tests are written and failing:

1. Create `src/transformers/integrations/mistral/__init__.py` (empty or with future exports).
2. Create `tests/integrations/__init__.py` and `tests/integrations/mistral/__init__.py` (empty).
3. Create `src/transformers/integrations/mistral/params_conversion.py`:
   - Sub-config dataclasses: `YarnNativeConfig`, `FP8NativeConfig` (with `__post_init__` validation for `qformat_weight == "fp8_e4m3"`), `MoeNativeConfig`, `VisionEncoderNativeConfig`
   - `MistralNativeConfig` (base): dataclass with all base fields, `from_params_json(cls, params)` reads fields directly via `params["key"]` (required) / `params.get("key", default)` (optional), `to_hf_config()` returns `MistralConfig(hidden_size=self.dim, ...)`, `from_hf_config(cls, config)` reverses
   - `Ministral3NativeConfig(MistralNativeConfig)`: adds `yarn: YarnNativeConfig | None` and `quantization: FP8NativeConfig | None`. `from_params_json` constructs sub-config dataclasses from nested dicts. `to_hf_config` builds `rope_parameters` from YaRN and `quantization_config` from FP8
   - `Mistral4NativeConfig(MistralNativeConfig)`: adds MLA fields + `moe: MoeNativeConfig | None` + `yarn`. `to_hf_config` maps MoE fields and computes `partial_rotary_factor` from `qk_rope_head_dim / (qk_rope_head_dim + qk_nope_head_dim)`
   - `Mistral3NativeConfig`: composite with `text_config` (union type) + `vision_encoder`. `from_params_json` auto-detects backbone: `moe` → `Mistral4NativeConfig`, `yarn` or `quantization` → `Ministral3NativeConfig`, else → `MistralNativeConfig`. `to_hf_config` returns `Mistral3Config` with sub-configs
   - Dispatchers: `native_config_for_model_type(model_type, params)` and `native_config_from_hf_config(model_type, config)` using `match`/`case`

### Key Design Decisions

- **No mapping dicts or passthrough**: Every field is an explicit dataclass field. `from_params_json` reads directly from the dict. `to_hf_config` maps explicitly.
- **`rope_theta`**: absorbed into `rope_parameters` by HF config `__post_init__`, so `from_hf_config` must extract it from `config.rope_parameters` when not available as a direct attribute.
- **FP8 validation**: `FP8NativeConfig.__post_init__` raises `ValueError` if `qformat_weight != "fp8_e4m3"`.
- **FP8 roundtrip**: `quantization` doesn't roundtrip via HF config attrs (HF uses `quantization_config` which isn't stored as individual config fields). Tests exclude FP8 from roundtrip assertions.
- **Mistral3 backbone auto-detection**: In `Mistral3NativeConfig.from_params_json`, presence of `moe` → `Mistral4NativeConfig`, presence of `yarn` or `quantization` → `Ministral3NativeConfig`, otherwise → `MistralNativeConfig`.

---

## Phase 2: Weight Key Conversion (Requires Torch)

### What We're Building

Factory functions that return lists of `WeightRenaming` and `WeightConverter` entries
for each model type, plus `FP8AwareMergeAndConcatenate` for expert fusion.

**File**: `src/transformers/integrations/mistral/weight_conversion.py`

### Tests to Write First

**File**: `tests/integrations/mistral/test_weight_conversion.py`

All tests use `@require_torch`, no Hub downloads.

**Imports needed in test file** (guarded by `if is_torch_available()`):
```python
import torch
from transformers.core_model_loading import PermuteForRope
from transformers.integrations.mistral.weight_conversion import (
    FP8AwareMergeAndConcatenate,
    FP8AwareSplitAndUnstack,
    fp8_scale_renamings,
    mistral3_native_text_renamings,
    mistral3_native_vision_converters,
    mistral3_native_vision_renamings,
    mistral4_native_converters,
    mistral4_native_renamings,
    mistral_base_native_converters,
    mistral_base_native_renamings,
)
```

**Helper**: Define a `_source_target_pairs(entries)` function that extracts `(source, target)` tuples from a list of `WeightRenaming`/`WeightConverter` by iterating `entry.source_patterns` × `entry.target_patterns`.

#### 2.1 Factory Function Outputs (8 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 1 | `test_mistral_base_renamings_count_and_patterns` | `renamings = mistral_base_native_renamings()` | `len(renamings) >= 8`; source set contains `output`, `tok_embeddings`; target set contains `lm_head`, `embed_tokens` |
| 2 | `test_mistral_base_converters_rope` | `converters = mistral_base_native_converters()` | `len(converters) == 2`; `converters[0].operations[0]` is `PermuteForRope` with `n_heads_attr=="num_attention_heads"`; `converters[1].operations[0]` is `PermuteForRope` with `n_heads_attr=="num_key_value_heads"` |
| 3 | `test_fp8_scale_renamings` | `renamings = fp8_scale_renamings()` | Sources include `.qscale_weight` → target `.weight_scale_inv`; `.qscale_act` → `.activation_scale` |
| 4 | `test_mistral3_text_renamings_prefixed` | `renamings = mistral3_native_text_renamings()` | Every target pattern starts with `language_model.` |
| 5 | `test_mistral3_vision_renamings` | `renamings = mistral3_native_vision_renamings()` | Source set contains `vision_encoder`; target set contains `vision_tower`; source set contains `vision_language_adapter`; target set contains `multi_modal_projector` |
| 6 | `test_mistral3_vision_converters_dotted_heads` | `converters = mistral3_native_vision_converters()` | Each converter's first operation is `PermuteForRope` with `n_heads_attr=="vision_config.num_attention_heads"` |
| 7 | `test_mistral4_renamings_mla_keys` | `renamings = mistral4_native_renamings()` | Source/target pairs contain `wkv_a_with_mqa`→`kv_a_proj_with_mqa`, `wq_a`→`q_a_proj`, `wq_b`→`q_b_proj` |
| 8 | `test_mistral4_converters_expert_fusion` | `converters = mistral4_native_converters()` | First converter's first operation is `FP8AwareMergeAndConcatenate` |

#### 2.2 FP8AwareMergeAndConcatenate (5 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 9 | `test_fp8_merge_bf16` | 4 experts, BF16 `gate` and `up` tensors of shape `(intermediate=8, hidden=16)` each, no scales | `result["fused"].shape == (4, 16, 16)` (2*intermediate) |
| 10 | `test_fp8_merge_per_tensor_fp8` | 4 experts, `float8_e4m3fn` gate+up, scalar scales | `result["fused"].shape == (4, 16, 16)`, `result["fused_scale"]` exists with `shape[0]==4` |
| 11 | `test_fp8_merge_blockwise_fp8` | 4 experts, `float8_e4m3fn` gate+up, multi-dimensional scales `(block_rows, block_cols)` | `result["fused_scale"].shape == (4, 2*block_rows, block_cols)` |
| 12 | `test_fp8_merge_mismatched_expert_count_raises` | gate has 4 experts, up has 3 | `assertRaises(ValueError)` |
| 13 | `test_fp8_merge_single_expert` | 1 expert, BF16 | `result["fused"].shape == (1, 2*intermediate, hidden)` — edge case |

#### 2.3 FP8AwareSplitAndUnstack (Reverse) (2 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 14 | `test_fp8_split_roundtrip_bf16` | Merge 4 BF16 experts → split → compare to original | `torch.testing.assert_close` for each expert's gate and up |
| 15 | `test_fp8_split_roundtrip_per_tensor_fp8` | Merge 4 FP8 experts with uniform scales → split | Shapes match, values approximately match (FP8 rescaling loses precision) |

#### 2.4 End-to-End Key Conversion (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 16 | `test_apply_renamings_to_native_mistral_state_dict` | Build a state dict with native keys for 1-layer model (output.weight, norm.weight, tok_embeddings.weight, layers.0.attention.wq/wk/wv/wo.weight, layers.0.feed_forward.w1/w2/w3.weight, layers.0.attention_norm.weight, layers.0.ffn_norm.weight), apply `rename_source_key` per-key | All keys renamed to HF format (lm_head.weight, model.norm.weight, model.embed_tokens.weight, model.layers.0.self_attn.q_proj.weight, etc.) |
| 17 | `test_hf_keys_pass_through_native_renamings` | Apply native renamings to HF-format keys (lm_head.weight, model.layers.0.self_attn.q_proj.weight, etc.) | All keys unchanged (native source regex patterns don't match HF keys) |
| 18 | `test_mistral4_renamings_cover_all_mla_keys` | Build native state dict with MLA keys (wkv_a_with_mqa, wq_a, wq_b, wkv_b, gate, shared_experts, experts.*.w1/w2/w3) | After renaming, all expected HF keys present (kv_a_proj_with_mqa, q_a_proj, q_b_proj, kv_b_proj, router, shared_experts, etc.) |

**Phase 2 total: 18 tests**

### Implementation

After tests are written and failing:

1. Create `src/transformers/integrations/mistral/weight_conversion.py`:
   - Import `ConversionOps`, `MergeModulelist`, `PermuteForRope`, `WeightConverter`, `WeightRenaming` from `core_model_loading`
   - `_rescale_fp8(tensor, old_scale, new_scale)` helper
   - `FP8AwareMergeAndConcatenate(ConversionOps)`:
     - `convert()` method handling 2 source patterns (gate+up, BF16) or 4 (gate+up+gate_scale+up_scale, FP8)
     - BF16: stack gate+up, concat along dim=1
     - Per-tensor FP8 (scalar scales): rescale all to common max scale, then concat
     - Block-wise FP8 (multi-dim scales): concat scales independently without rescaling
     - `reverse_op` returns `FP8AwareSplitAndUnstack()`
   - `FP8AwareSplitAndUnstack(ConversionOps)`:
     - Reverse: split fused tensor along dim=1, unstack along dim=0
     - `reverse_op` returns `FP8AwareMergeAndConcatenate()`
   - Factory functions: `mistral_base_native_renamings()`, `mistral_base_native_converters()`,
     `fp8_scale_renamings()`, `mistral3_native_text_renamings()`, `mistral3_native_text_converters()`,
     `mistral3_native_vision_renamings()`, `mistral3_native_vision_converters()`,
     `mistral4_native_renamings()`, `mistral4_native_converters()`

2. Register `FP8AwareMergeAndConcatenate` and `FP8AwareSplitAndUnstack` in
   `_INTERNAL_MANY_TO_MANY_CONVERSIONS` in `core_model_loading.py` via lazy import
   (convert the tuple to a function `_get_internal_many_to_many_conversions()` to
   avoid circular imports).

### Key Weight Mappings Reference

**Base Mistral** (shared by mistral/ministral3):
- `output.weight` → `lm_head.weight`
- `tok_embeddings.weight` → `model.embed_tokens.weight`
- `norm.weight` → `model.norm.weight`
- `layers.{i}.attention_norm.weight` → `model.layers.{i}.input_layernorm.weight`
- `layers.{i}.ffn_norm.weight` → `model.layers.{i}.post_attention_layernorm.weight`
- `layers.{i}.attention.wv.weight` → `model.layers.{i}.self_attn.v_proj.weight`
- `layers.{i}.attention.wo.weight` → `model.layers.{i}.self_attn.o_proj.weight`
- `layers.{i}.feed_forward.w1.weight` → `model.layers.{i}.mlp.gate_proj.weight`
- `layers.{i}.feed_forward.w2.weight` → `model.layers.{i}.mlp.down_proj.weight`
- `layers.{i}.feed_forward.w3.weight` → `model.layers.{i}.mlp.up_proj.weight`
- Q/K via `PermuteForRope`: `layers.{i}.attention.wq.weight` → `model.layers.{i}.self_attn.q_proj.weight`, `layers.{i}.attention.wk.weight` → `model.layers.{i}.self_attn.k_proj.weight`

**FP8 scales**:
- `.qscale_weight` → `.weight_scale_inv`
- `.qscale_act` → `.activation_scale`

**Mistral3 text** (same as base but prefixed with `language_model.`):
- All base keys with `model.` prefix replaced by `language_model.model.` in target and equivalent changes in source

**Mistral3 vision**:
- `vision_encoder.transformer.layers.{i}.attention.wq/wk/wv/wo` → `vision_tower.transformer.layers.{i}.attention.wq/wk/wv/wo`
- `vision_language_adapter` → `multi_modal_projector`
- `patch_merger` mappings

**Mistral4 MLA-specific**:
- `wkv_a_with_mqa` → `kv_a_proj_with_mqa`
- `wq_a` → `q_a_proj`
- `wq_b` → `q_b_proj`
- `wkv_b` → `kv_b_proj`
- `gate` → `router`
- `shared_experts` → `shared_experts`
- `experts.{i}.w1/w3` → fused via `FP8AwareMergeAndConcatenate` → `experts.{i}.gate_up_proj`
- `experts.{i}.w2` → stacked via `MergeModulelist` → `experts.{i}.down_proj`

---

## Phase 3: PermuteForRope Fixes

### What We're Building

Fix `PermuteForRope` to support configurable `n_heads_attr` (dotted attribute paths)
and proper error handling. Current version (at merge base) has a bug: uses
`self.config.getattr(...)` instead of `getattr(self.config, ...)`, and doesn't
support dotted paths or `reverse_op`.

**File**: `src/transformers/core_model_loading.py`

### Tests to Write First

**File**: `tests/integrations/mistral/test_weight_conversion.py` (appended to Phase 2 tests)

These tests need fake config objects (simple classes with attributes).

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 19 | `test_permute_for_rope_self_inverse` | `head_dim=4`, `n_heads=8`, tensor shape `(n_heads*head_dim, 16)`. Create `PermuteForRope("num_attention_heads")`, set `config` with `num_attention_heads=8`. Apply `_apply` twice. | `torch.testing.assert_close(restored, original)` — self-inverse only works with head_dim=4 |
| 20 | `test_permute_for_rope_missing_attr_raises` | Config with `num_attention_heads=8`. Create `PermuteForRope("num_atention_heads")` (typo). Set config, call `_apply`. | `assertRaises(AttributeError)` |
| 21 | `test_permute_for_rope_dotted_attr` | `InnerConfig` with `num_attention_heads=4`. `OuterConfig` with `vision_config=InnerConfig()`. Create `PermuteForRope("vision_config.num_attention_heads")`. Set config, apply to tensor `(4*64, 256)`. | `result.shape == original.shape`, no error |
| 22 | `test_permute_for_rope_reverse_op` | `perm = PermuteForRope("num_key_value_heads")` | `rev = perm.reverse_op` is `PermuteForRope` with `n_heads_attr=="num_key_value_heads"` |
| 23 | `test_permute_for_rope_repr` | `perm = PermuteForRope("num_attention_heads")` | `repr(perm)` contains `"num_attention_heads"` |

**Phase 3 total: 5 tests (appended to test_weight_conversion.py)**

### Implementation

After tests are written and failing:

1. `PermuteForRope.__init__(self, n_heads_attr: str = "num_attention_heads")` — stores `self.n_heads_attr`
2. Add `_resolve_attr(self, config)` — splits `n_heads_attr` on `.`, walks attributes; raises `AttributeError` on missing segment
3. Fix `_apply` to use `self._resolve_attr(self.config)` instead of `self.config.getattr(...)`
4. Add `reverse_op` property returning `PermuteForRope(n_heads_attr=self.n_heads_attr)`
5. Add `__repr__` showing `n_heads_attr`
6. Make `convert()` accept `config` as optional kwarg (for compatibility with `WeightConverter.convert` calling convention)

---

## Phase 4: Config Format Detection & Loading

### What We're Building

`MistralFormatConfig` — a `PreTrainedConfig` subclass that overrides `get_config_dict()`
to auto-detect and load from `params.json` when `config.json` is absent.

**Files**:
- `src/transformers/integrations/mistral/config_format.py` (new)
- `src/transformers/models/mistral/configuration_mistral.py` (modify inheritance)
- `src/transformers/models/ministral3/configuration_ministral3.py` (modify inheritance)
- `src/transformers/models/mistral3/configuration_mistral3.py` (modify inheritance)
- `src/transformers/models/mistral4/configuration_mistral4.py` (modify inheritance)

### Tests to Write First

**File**: `tests/integrations/mistral/test_config_format.py`

**Imports needed**:
```python
from unittest.mock import patch
from transformers import MistralConfig
from transformers.integrations.mistral.config_format import (
    _CONSOLIDATED_INDEX, _CONSOLIDATED_SINGLE, _HF_INDEX, _HF_SINGLE,
    MistralFormatConfig,
)
```

Some tests require `@require_torch` for config classes that need torch (Ministral3Config, Mistral4Config).

**Mock helper**: Define `_make_cached_file_side_effect(existing_files: set[str])` that returns
a function simulating `cached_file` — returns a fake path for files in the set, `None` otherwise.

#### 4.1 Weight File Detection (Mocked) (5 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 1 | `test_detect_weight_file_hf_single_exists` | Mock `cached_file` to find `model.safetensors` only | `_detect_weight_file("repo") is None` (prefer HF) |
| 2 | `test_detect_weight_file_hf_index_exists` | Mock to find `model.safetensors.index.json` | Returns `None` |
| 3 | `test_detect_weight_file_consolidated_single` | Mock: no HF files, finds `consolidated.safetensors` | Returns `"consolidated.safetensors"` |
| 4 | `test_detect_weight_file_consolidated_index` | Mock: no HF files, finds `consolidated.safetensors.index.json` | Returns the index filename |
| 5 | `test_detect_weight_file_nothing_found` | Mock: nothing found | Returns `None` |

#### 4.2 Weight File Detection Edge Cases (2 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 6 | `test_detect_weight_file_both_hf_and_consolidated` | Mock: both `model.safetensors` AND `consolidated.safetensors` exist | Returns `None` (HF preferred) |
| 7 | `test_detect_weight_file_hf_index_and_consolidated_single` | Mock: `model.safetensors.index.json` + `consolidated.safetensors` | Returns `None` (HF preferred) |

#### 4.3 Config from params.json (Mocked cached_file) (6 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 8 | `test_get_config_dict_prefers_config_json` | Both `config.json` and `params.json` exist in mock | Uses `config.json` (standard PreTrainedConfig path) |
| 9 | `test_get_config_dict_falls_back_to_params_json` | Only `params.json` exists | Auto-detects and uses `params.json`; result dict contains correct HF fields |
| 10 | `test_get_config_dict_mistral_format_true_forces_params` | Both exist, pass `mistral_format=True` | Forces `params.json` loading |
| 11 | `test_get_config_dict_mistral_format_false_errors` | Only `params.json` exists, pass `mistral_format=False` | Raises `OSError` (config.json not found) |
| 12 | `test_get_config_dict_sets_loaded_from_mistral_format` | Load from `params.json` | `result_dict["_loaded_from_mistral_format"] is True` |
| 13 | `test_get_config_dict_sets_transformers_weights` | Load from `params.json`, consolidated exists in mock | `result_dict["transformers_weights"]` is set to the consolidated filename |

#### 4.4 Config-to-params Reverse (for save) (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 14 | `test_config_to_params_json_mistral` | Create `MistralConfig(hidden_size=4096, ...)`, call `_config_to_params_json()` | Returns dict with `dim==4096`, `n_layers`, `hidden_dim`, `n_heads`, `head_dim`, `vocab_size`, `n_kv_heads`, `sliding_window`, `max_position_embeddings` — all native field names |
| 15 | `test_config_to_params_json_ministral3` | Create `Ministral3Config(...)`, call `_config_to_params_json()` | Returns dict with YaRN fields if rope_parameters present |
| 16 | `test_config_to_params_json_mistral4` | Create `Mistral4Config(...)`, call `_config_to_params_json()` | Returns dict with `moe` sub-dict containing `num_experts`, `num_experts_per_tok`, `expert_hidden_dim` |

#### 4.5 Dummy-Model Config Integration (Tiny Models, Local Files) (4 tests)

These create actual `params.json` files in temp directories and load configs from them.

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 17 | `test_dummy_mistral_config_from_params_json` | Write tiny mistral `params.json` (dim=32, n_layers=2, etc.) to tmpdir → `MistralConfig.from_pretrained(tmpdir, mistral_format=True)` | `config.hidden_size==32`, `config.num_hidden_layers==2`, `config.num_attention_heads==2` |
| 18 | `test_dummy_ministral3_config_from_params_json` | Same for Ministral3Config | Config loaded with correct fields |
| 19 | `test_dummy_mistral4_config_from_params_json` | Same for Mistral4Config with `moe` sub-dict | `config.n_routed_experts` matches, `config.moe_intermediate_size` matches |
| 20 | `test_dummy_mistral3_config_from_params_json` | Same for Mistral3Config with `vision_encoder` | `config.text_config` and `config.vision_config` present |

**Phase 4 total: 20 tests**

### Implementation

After tests are written and failing:

1. Create `src/transformers/integrations/mistral/config_format.py`:
   - Constants: `_PARAMS_JSON = "params.json"`, `_CONSOLIDATED_SINGLE = "consolidated.safetensors"`, `_CONSOLIDATED_INDEX = "consolidated.safetensors.index.json"`, `_HF_SINGLE = "model.safetensors"`, `_HF_INDEX = "model.safetensors.index.json"`
   - `MistralFormatConfig(PreTrainedConfig)` class:
     - `get_config_dict(cls, pretrained_model_name_or_path, **kwargs)` classmethod:
       - Pop `mistral_format` from kwargs
       - If `mistral_format is not True`: try standard `config.json` via `super().get_config_dict()`
       - If that fails (OSError) or `mistral_format is True`: try `_get_config_dict_from_params_json()`
       - If `mistral_format is False` and `config.json` fails: re-raise
     - `_get_config_dict_from_params_json(cls, path, **kwargs)` classmethod:
       - `cached_file(path, "params.json")` → load JSON
       - `native_config_for_model_type(cls.model_type, params_dict).to_hf_config().to_dict()` → config_dict
       - `_detect_weight_file(path)` → set `transformers_weights`
       - Set `_loaded_from_mistral_format = True`
       - Return `(config_dict, kwargs)`
     - `_detect_weight_file(cls, path, **kwargs)` classmethod:
       - Probe in order: `_HF_SINGLE`, `_HF_INDEX` (→ return None if found), then `_CONSOLIDATED_INDEX`, `_CONSOLIDATED_SINGLE` (→ return filename if found)
     - `_config_to_params_json(self)` method:
       - `native = native_config_from_hf_config(self.model_type, self)`
       - `dataclasses.asdict(native)` → params dict (with nested dataclasses serialized to dicts)

2. Modify each config class to inherit from `MistralFormatConfig`:
   - Add `from transformers.integrations.mistral.config_format import MistralFormatConfig`
   - Change `class MistralConfig(PreTrainedConfig)` → `class MistralConfig(MistralFormatConfig)`
   - Same for Ministral3Config, Mistral3Config, Mistral4Config

3. Update `src/transformers/integrations/mistral/__init__.py` exports to include `MistralFormatConfig`.

---

## Phase 5: Full from_pretrained Pipeline (Tiny Model Integration Tests)

### What We're Building

Registration of all weight conversions in `CONVERSION_MAPPING` and end-to-end
`from_pretrained` with tiny locally-created models in native format.

**Files**:
- `src/transformers/conversion_mapping.py` (register all model types)

### Tests to Write First

**File**: `tests/integrations/mistral/test_integration.py`

All tests use `@require_torch`, no Hub downloads. They create tiny models with
`hidden_size=32`, `num_hidden_layers=2`, `num_attention_heads=2`, `num_key_value_heads=2`, etc.

**Imports needed** (guarded by `is_torch_available()`):
```python
import torch
from safetensors.torch import save_file
from transformers import (
    AutoModelForCausalLM, MistralConfig, MistralForCausalLM,
    Ministral3Config, Mistral4Config, Mistral4ForCausalLM,
)
from transformers.core_model_loading import PermuteForRope, WeightRenaming, WeightConverter
from transformers.conversion_mapping import get_checkpoint_conversion_mapping
```

**Fixture strategy**: For each model type:
1. Create a tiny HF config
2. Instantiate the model
3. Get the HF state dict
4. Manually reverse HF keys → native keys (using hardcoded key mappings, not `revert_weight_conversion` which may not work yet)
5. For Q/K weights, apply inverse RoPE permutation
6. Write `params.json` + `consolidated.safetensors` to tmpdir

Define helper functions:
- `_tiny_mistral_config() → MistralConfig` with hidden_size=32, num_hidden_layers=2, etc.
- `_tiny_mistral4_config() → Mistral4Config` with tiny MoE params (n_routed_experts=4, num_experts_per_tok=2, etc.)
- `_build_native_mistral_checkpoint(tmpdir)` → writes params.json + consolidated.safetensors
- `_build_native_mistral4_checkpoint(tmpdir)` → same for mistral4

#### 5.1 Conversion Mapping Registration (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 1 | `test_mistral_conversion_mapping_registered` | `get_checkpoint_conversion_mapping("mistral")` | Returns non-None list; contains `WeightRenaming` and `WeightConverter` entries |
| 2 | `test_ministral3_conversion_mapping_registered` | `get_checkpoint_conversion_mapping("ministral3")` | Returns non-None list; includes FP8 scale renamings |
| 3 | `test_mistral4_conversion_mapping_registered` | `get_checkpoint_conversion_mapping("mistral4")` | Returns non-None list; includes `FP8AwareMergeAndConcatenate` |

#### 5.2 from_pretrained with Native Format (4 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 4 | `test_from_pretrained_mistral_native_format` | Write `params.json` + `consolidated.safetensors` with native keys → `AutoModelForCausalLM.from_pretrained(tmpdir)` | Model loads; state dict has HF keys (lm_head.weight, model.layers.0.self_attn.q_proj.weight, etc.); config `_loaded_from_mistral_format` is True |
| 5 | `test_from_pretrained_ministral3_native_format` | Same fixture approach for ministral3 | Model loads with correct config, HF keys in state dict |
| 6 | `test_from_pretrained_mistral4_native_format` | Same for mistral4 (with tiny expert tensors for MoE) | Expert gate+up tensors fused into `gate_up_proj`; model loads |
| 7 | `test_from_pretrained_mistral_native_weights_match` | Load from native, compare to model created from config directly | Non-Q/K weights should match exactly; Q/K weights should differ only by RoPE permutation |

#### 5.3 HF Save and Reload Roundtrip (2 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 8 | `test_dummy_mistral_hf_save_reload_roundtrip` | Create tiny model from config → save HF → reload → compare state dicts | All keys and values match exactly |
| 9 | `test_dummy_mistral4_hf_save_reload_roundtrip` | Same for mistral4 | All keys and values match |

#### 5.4 from_pretrained Prefers HF When Both Exist (1 test)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 10 | `test_from_pretrained_prefers_hf_when_both_exist` | Write both `config.json` + `model.safetensors` AND `params.json` + `consolidated.safetensors` | Loads from HF format (no native conversion applied) |

#### 5.5 from_pretrained with mistral_format Override (2 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 11 | `test_from_pretrained_mistral_format_true` | Both formats in tmpdir, pass `mistral_format=True` | Forces native format loading |
| 12 | `test_from_pretrained_mistral_format_false_no_hf` | Only native exists, pass `mistral_format=False` | `assertRaises(OSError)` |

#### 5.6 Native Weight Key Conversion Correctness (1 test)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 13 | `test_native_weight_conversion_keys_correct` | Build native state dict manually, apply registered conversion mapping key-by-key via `rename_source_key` | Set of renamed keys equals expected HF keys exactly |

**Phase 5 total: 13 tests**

### Implementation

After tests are written and failing:

1. Register all model types in `conversion_mapping.py`'s `_build_checkpoint_conversion_mapping()`:
   ```python
   mapping["mistral"] = mistral_base_native_renamings() + mistral_base_native_converters()
   mapping["ministral3"] = mistral_base_native_renamings() + mistral_base_native_converters() + fp8_scale_renamings()
   mapping["mistral3"] = mapping["llava"].copy() + mistral3_native_text_renamings() + mistral3_native_text_converters()
                          + mistral3_native_vision_renamings() + mistral3_native_vision_converters()
                          + fp8_scale_renamings()
   mapping["mistral4"] = mistral4_native_renamings() + fp8_scale_renamings() + mistral4_native_converters()
   ```

2. Remove `"mistral3": "llava"` alias from `_MODEL_TO_CONVERSION_PATTERN` (now explicit).

3. Wire `transformers_explicit_filename` in `from_pretrained` — already exists in the
   codebase for `getattr(config, "transformers_weights", None)` — just needs the config
   to set it via `MistralFormatConfig._get_config_dict_from_params_json`.

### Important Notes

- **HF keys pass through native patterns safely**: Native source patterns (e.g. `^output\.weight`)
  don't match HF keys (e.g. `lm_head.weight`), so registering native conversions is harmless
  for HF-format loading.
- **`mistral3` needs explicit entry**: Since we remove the `"llava"` alias, we must include
  the llava-style renamings (`language_model.model`→`language_model`, etc.) in the `mistral3` entry.
- **Vision RoPE**: Vision Q/K use `PermuteForRope("vision_config.num_attention_heads")` to
  read from the composite config's vision sub-config.

---

## Phase 6: save_pretrained Native Format + Roundtrip

### What We're Building

Support for `save_pretrained(save_format="mistral")` that saves in native Mistral format:
`params.json` + `consolidated.safetensors` with native key names. Also fix the
`revert_weight_conversion` regex-key bug that blocks this.

**Files**:
- `src/transformers/modeling_utils.py` (add `save_format` parameter)
- `src/transformers/core_model_loading.py` (fix `revert_weight_conversion` / `WeightConverter.convert`)
- `src/transformers/configuration_utils.py` (strip `transformers_weights` on save)

### Tests to Write First

**File**: `tests/integrations/mistral/test_integration.py` (appended to Phase 5 tests)

#### 6.1 Save Format (5 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 14 | `test_save_pretrained_default_hf_format` | Create model from HF config → `save_pretrained(tmpdir)` | `config.json` exists, `model.safetensors` (or index) exists, NO `params.json` |
| 15 | `test_save_pretrained_default_preserves_native_format` | Load from native → `save_pretrained(tmpdir)` (default) | `params.json` exists, `consolidated.safetensors` exists, NO `config.json` alone |
| 16 | `test_save_pretrained_force_hf` | Load from native → `save_pretrained(tmpdir, save_format="hf")` | `config.json` + `model.safetensors` exists |
| 17 | `test_save_pretrained_force_mistral` | Load from HF config → `save_pretrained(tmpdir, save_format="mistral")` | `params.json` + `consolidated.safetensors` exists |
| 18 | `test_save_pretrained_invalid_save_format_raises` | `save_pretrained(tmpdir, save_format="invalid")` | `assertRaises(ValueError)` |

#### 6.2 Roundtrip Tests (6 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 19 | `test_roundtrip_native_to_hf_to_native_mistral` | Load native → save HF → load HF → save native → load native | Final state dict matches original (exact for non-Q/K, approximate for Q/K due to RoPE) |
| 20 | `test_roundtrip_native_to_hf_to_native_mistral4` | Same for mistral4 | Final state dict matches |
| 21 | `test_roundtrip_hf_to_native_to_hf_mistral` | Create model → save native → load native → save HF → load HF | Final state dict matches original |
| 22 | `test_roundtrip_hf_to_native_to_hf_mistral4` | Same for mistral4 | Final state dict matches |
| 23 | `test_roundtrip_config_native_to_hf_to_native` | Load config from params.json → save as config.json → load config.json → save as params.json → compare | Architecture-defining fields match |
| 24 | `test_roundtrip_preserves_model_output` | Create model, run forward pass, roundtrip save/load, run forward pass again | Output tensors match (functional equivalence) |

#### 6.3 revert_weight_conversion Fix Verification (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 25 | `test_revert_weight_conversion_no_regex_keys` | Load model, call `revert_weight_conversion(model, state_dict)` | No key in output contains regex metacharacters (`^`, `\d+`, `\(`, `$`) |
| 26 | `test_revert_weight_conversion_roundtrip` | Build native state dict, apply forward conversion, then revert | Original native keys restored |
| 27 | `test_revert_weight_conversion_returns_all_keys` | Revert conversion on a complete model state dict | Output has exactly the expected number of keys (same count as input) |

**Phase 6 total: 14 tests (appended to test_integration.py)**

### Implementation

After tests are written and failing:

1. **Fix `revert_weight_conversion` / `WeightConverter.convert` regex-key bug**:

   The root cause is in `WeightConverter.convert()` (lines 752-758 at merge base):
   ```python
   try:
       prefix, _, suffix = next(full_name.partition(k) for k in collected_tensors.keys() if k in full_name)
       collected_tensors = {prefix + k + suffix: v for k, v in collected_tensors.items()}
   except StopIteration:
       pass
   ```
   When invoked via `revert_weight_conversion`, `collected_tensors` keys are regex patterns
   (the reversed source patterns), which don't appear as substrings of `full_name`.

   **Fix approach**: After ops produce output with target pattern keys, map them to concrete
   names using the regex substitution. Instead of simple `partition`, use `re.sub` on the
   source pattern with the layer name to derive the concrete output key.

2. **Add `save_format` parameter to `save_pretrained`**:
   - `save_format: str | None = None` — `"hf"`, `"mistral"`, or `None` (auto)
   - Validate: raise `ValueError` for unknown values
   - When `None`: check `config._loaded_from_mistral_format`; if True → save native, else → save HF
   - When saving native:
      - Write `params.json` via `config._config_to_params_json()` (which calls `native_config_from_hf_config` + `dataclasses.asdict`)
     - Save weights, then rename `model*.safetensors` → `consolidated*.safetensors`
     - Rewrite index JSON with updated weight_map filenames

3. **Strip internal metadata from saved config**:
   - Remove `_loaded_from_mistral_format` and `transformers_weights` from `config.json` output
   - These are runtime flags, not config parameters

---

## Phase 7: Slow Integration Tests (Real Hub Models)

### What We're Building

Tests against real Mistral models on HuggingFace Hub. These validate the full pipeline
with production-scale configs and weights.

**File**: `tests/integrations/mistral/test_slow_integration.py`

All tests marked `@slow` and `@require_torch`. Skipped unless `RUN_SLOW=1`.

### Hub Models

| Model Type | Hub Repo | Size | Notes |
|-----------|----------|------|-------|
| `mistral` | `mistralai/Mistral-7B-v0.3` or similar | 7B | Base text model |
| `ministral3` | `mistralai/Ministral-3-3B-Instruct-2512` | 3B | FP8 + YaRN |
| `mistral3` | `mistralai/Mistral-Small-3.2-24B-Instruct-2506` | 24B | Composite VLM |
| `mistral4` | `mistralai/Mistral-Small-4-119B-2603` | 119B | MoE/MLA |

### Tests

#### 7.1 Config Comparison (Config-Only, Fast-ish) (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 1 | `test_ministral3_native_vs_hf_config_match` | `AutoConfig.from_pretrained(repo, mistral_format=True)` vs `AutoConfig.from_pretrained(repo, mistral_format=False)` | Architecture fields match: hidden_size, num_hidden_layers, intermediate_size, num_attention_heads, num_key_value_heads, head_dim, vocab_size |
| 2 | `test_mistral3_native_vs_hf_config_match` | Same for mistral3 (compare text_config sub-fields) | Text config architecture fields match |
| 3 | `test_mistral4_native_vs_hf_config_match` | Same for mistral4 | Architecture fields match |

#### 7.2 Weight Loading (Downloads Full Models) (3 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 4 | `test_ministral3_native_weight_loading` | `AutoModelForCausalLM.from_pretrained(repo, mistral_format=True, torch_dtype=torch.bfloat16)` | State dict contains `model.layers` keys and `lm_head.weight` |
| 5 | `test_mistral3_native_weight_loading` | Same for mistral3 | State dict contains `language_model.model.layers` keys |
| 6 | `test_mistral4_native_weight_loading` | Same for mistral4 | State dict contains `model.layers` and `lm_head.weight` |

#### 7.3 Save and Reload (Full Roundtrip) (2 tests)

| # | Test | Setup | Assertions |
|---|------|-------|------------|
| 7 | `test_save_reload_native_format_ministral3` | Load native → save native to tmpdir → verify params.json exists → reload from tmpdir | State dicts match |
| 8 | `test_save_reload_hf_format_ministral3` | Load native → save HF to tmpdir → reload from tmpdir | State dicts match |

**Phase 7 total: 8 tests**

### Implementation

These tests don't require new implementation — they validate the work from Phases 1-6.
Any failures found here feed back as bug fixes in earlier phases.

---

## File Structure Summary

### New Files

```
src/transformers/integrations/mistral/
    __init__.py                    # Package init with exports
    config_format.py               # MistralFormatConfig (Phase 4)
    params_conversion.py           # Native config dataclass hierarchy (Phase 1)
    weight_conversion.py           # Weight conversion factories + FP8 ops (Phase 2)

tests/integrations/
    __init__.py
    mistral/
        __init__.py
        test_params_conversion.py  # Phase 1 tests (25 tests, pure Python)
        test_weight_conversion.py  # Phase 2+3 tests (23 tests, @require_torch)
        test_config_format.py      # Phase 4 tests (20 tests, mocked + local)
        test_integration.py        # Phase 5+6 tests (27 tests, @require_torch, tiny models)
        test_slow_integration.py   # Phase 7 tests (8 tests, @slow, real Hub models)
```

### Modified Files

```
src/transformers/core_model_loading.py              # Phase 3: PermuteForRope fixes + Phase 6: revert bug fix
src/transformers/conversion_mapping.py              # Phase 5: Register all model types
src/transformers/modeling_utils.py                   # Phase 6: save_format parameter
src/transformers/configuration_utils.py             # Phase 6: Strip internal metadata on save
src/transformers/models/mistral/configuration_mistral.py       # Phase 4: MistralFormatConfig inheritance
src/transformers/models/ministral3/configuration_ministral3.py # Phase 4: MistralFormatConfig inheritance
src/transformers/models/mistral3/configuration_mistral3.py     # Phase 4: MistralFormatConfig inheritance
src/transformers/models/mistral4/configuration_mistral4.py     # Phase 4: MistralFormatConfig inheritance
```

### Pre-Existing (Moved)

```
src/transformers/integrations/mistral.py → src/transformers/integrations/mistral/tokenizer.py
```

The existing `MistralConverter` and `convert_tekken_tokenizer()` are moved into the
new package structure. Imports elsewhere need updating.

---

## Test Count Summary

| File | Unit | Integration | Slow | Total |
|------|------|-------------|------|-------|
| `test_params_conversion.py` | 25 | — | — | 25 |
| `test_weight_conversion.py` | 23 | — | — | 23 |
| `test_config_format.py` | — | 20 | — | 20 |
| `test_integration.py` | — | 27 | — | 27 |
| `test_slow_integration.py` | — | — | 8 | 8 |
| **Total** | **48** | **47** | **8** | **103** |

---

## Running Tests

```bash
# Phase 1: Pure Python unit tests (no torch, no Hub)
pytest tests/integrations/mistral/test_params_conversion.py -v

# Phase 2+3: Weight conversion unit tests (torch, no Hub)
pytest tests/integrations/mistral/test_weight_conversion.py -v

# Phase 4: Config format tests (mocked + local files)
pytest tests/integrations/mistral/test_config_format.py -v

# Phase 5+6: Integration tests with tiny models (torch, no Hub)
pytest tests/integrations/mistral/test_integration.py -v

# All fast tests
pytest tests/integrations/mistral/ -v -k "not slow"

# Phase 7: Slow tests (requires Hub access + credentials)
RUN_SLOW=1 pytest tests/integrations/mistral/test_slow_integration.py -v

# Everything
RUN_SLOW=1 pytest tests/integrations/mistral/ -v
```

---

## Known Challenges & Mitigations

### 1. `revert_weight_conversion` Regex-Key Bug

**Problem**: `WeightConverter.convert()` emits regex patterns as literal key names when
invoked via `revert_weight_conversion`.

**Impact**: Blocks `save_pretrained(save_format="mistral")` and all roundtrip tests.

**Fix**: Modify `WeightConverter.convert()` to properly map output keys from target
patterns to concrete names. The `StopIteration` fallback currently lets regex patterns
leak through — instead, use regex substitution to derive concrete keys from the
layer name and target patterns.

**TDD**: Write roundtrip tests (Phase 6) that will fail until this is fixed.

### 2. `rope_theta` Not Round-Tripped

**Problem**: `MistralConfig.__post_init__` absorbs `rope_theta` into `rope_parameters`,
so `getattr(config, "rope_theta")` returns `None`.

**Fix**: In `from_hf_config()`, extract `rope_theta` from `config.rope_parameters`
when not available as a direct attribute.

**TDD**: Include `rope_theta` assertion in roundtrip tests (Phase 1 test #18).

### 3. PermuteForRope Not Self-Inverse for Large head_dim

**Problem**: The permutation `view(n, h/2, 2, d).transpose(1,2).reshape(...)` is only
a mathematical involution when `h/2 <= 2` (head_dim <= 4).

**Impact**: Not a bug — `reverse_op` applies the permutation exactly once. But means
we can't test self-inverse property with production head dimensions.

**TDD**: Self-inverse test uses `head_dim=4`. Roundtrip tests verify correctness
through the full forward+reverse pipeline.

### 4. Vision RoPE Head Count

**Problem**: Vision Q/K converters registered under `"mistral3"` (composite model type)
need the vision config's `num_attention_heads`, but receive the composite config.

**Fix**: Use dotted attribute path: `PermuteForRope("vision_config.num_attention_heads")`.

**TDD**: Test 6 in Phase 2 verifies this.

### 5. FP8 Precision Loss in Roundtrip

**Problem**: Per-tensor FP8 expert fusion rescales gate+up to a common max scale.
The reverse split loses precision because we can't recover the original individual scales.

**Impact**: Roundtrip tests for FP8 must use approximate comparison (`torch.allclose`
with tolerance).

**TDD**: Test 15 in Phase 2 uses approximate comparison.

### 6. `_INTERNAL_MANY_TO_MANY_CONVERSIONS` Circular Import

**Problem**: `FP8AwareMergeAndConcatenate` is defined in `weight_conversion.py` which
imports from `core_model_loading.py`. Adding it to `_INTERNAL_MANY_TO_MANY_CONVERSIONS`
(defined in `core_model_loading.py`) would create a circular import.

**Fix**: Convert `_INTERNAL_MANY_TO_MANY_CONVERSIONS` from a static tuple to a lazy
function `_get_internal_many_to_many_conversions()` that imports at call time. Update
the validation in `WeightConverter.__post_init__` to call the function.

**TDD**: Phase 2 tests for `WeightConverter` with `FP8AwareMergeAndConcatenate` will
fail with `ValueError` until this is fixed.

---

## Implementation Order

```
Phase 1 → Phase 2 → Phase 3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
  │           │         │         │         │         │         │
  ▼           ▼         ▼         ▼         ▼         ▼         ▼
Tests       Tests     Tests     Tests     Tests     Tests     Tests
  │           │         │         │         │         │         │
  ▼           ▼         ▼         ▼         ▼         ▼         ▼
Implement  Implement  Implement Implement Implement Implement  Validate
  │           │         │         │         │         │
  ▼           ▼         ▼         ▼         ▼         ▼
Verify     Verify    Verify    Verify    Verify    Verify
```

Each phase is self-contained: write tests, see them fail, implement, see them pass.
Later phases build on earlier ones but don't modify them.

After all phases pass, run `make style` and `make fix-repo` before opening a PR.
