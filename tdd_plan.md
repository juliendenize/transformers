# TDD Plan: Automatic HF / Mistral Format Conversion

## Goal

Make `from_pretrained` and `save_pretrained` automatically detect and convert between
HuggingFace and Mistral native formats. Support `mistral`, `ministral3`, `mistral3`
(composite VLM), and `mistral4` (MoE/MLA) model types. No separate conversion scripts needed.

## Conventions

- **Copyright header**: All new files under `src/transformers/integrations/mistral/` and
  `tests/integrations/mistral/` must use:
  `# Copyright <year> Mistral AI and The HuggingFace Inc. team. All rights reserved.`
  followed by the Apache 2.0 license block. This matches the convention in existing
  Mistral model files (e.g. `configuration_mistral.py`, `convert_ministral3_weights_to_hf.py`).

- **Test organization**: Test classes are grouped by the **class or module under test**,
  not by feature. Each test class is named `Test<ClassUnderTest>` and contains all tests
  for that class (construction, conversion, roundtrip, edge cases, errors). Use object
  equality (`assertEqual(actual, expected)`) for dataclasses and HF configs rather than
  field-by-field assertions.

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

## Phase 1: Native Config Dataclasses (Pure Python, No Torch) — IMPLEMENTED ✅

### What We Built

A typed dataclass hierarchy representing Mistral native `params.json` configs, with
methods to convert to/from HF config objects. Pure Python, no torch dependency.

Every field that exists in `params.json` is an explicit dataclass field — there is no
passthrough mechanism. Fields that share the same name in native and HF format are
still explicit (e.g., `vocab_size` is a field on both the native dataclass and the
HF config). Conversion uses direct field reading in `from_params_json()` and explicit
mapping in `to_hf_config()` / `from_hf_config()`.

**File**: `src/transformers/integrations/mistral/params_conversion.py`
**Tests**: `tests/integrations/mistral/test_params_conversion.py`
**30/30 tests passing.**

### Architecture

#### ABC Mixin (Generic over HF config type)

```python
HFConfigT = TypeVar("HFConfigT", bound=PreTrainedConfig)

class MistralModelType(StrEnum):
    MISTRAL = "mistral"
    MINISTRAL3 = "ministral3"
    MISTRAL4 = "mistral4"
    MISTRAL3 = "mistral3"

class NativeToHFConfigMixin(ABC, Generic[HFConfigT]):
    r"""Abstract mixin binding a native config dataclass to its HF config type."""

    @abstractmethod
    def to_hf_config(self) -> HFConfigT: ...

    @classmethod
    @abstractmethod
    def from_hf_config(cls, config: HFConfigT) -> Self: ...

    @classmethod
    @abstractmethod
    def from_params_json(cls, params: dict) -> Self: ...
```

#### Sub-Config Dataclasses (no mixin — pure data)

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

#### Top-Level Config Dataclasses (Generic Inheritance)

```
NativeToHFConfigMixin[HFConfigT]  (ABC, Generic — defines the protocol)
├── MistralNativeConfig(NativeToHFConfigMixin[MistralConfig])     — concrete, binds to MistralConfig
│   ├── Ministral3NativeConfig(MistralNativeConfig, NativeToHFConfigMixin[Ministral3Config])
│   └── Mistral4NativeConfig(MistralNativeConfig, NativeToHFConfigMixin[Mistral4Config])
└── Mistral3NativeConfig(NativeToHFConfigMixin[Mistral3Config])   — separate root, composite VLM
```

Subclasses inherit data fields from `MistralNativeConfig` and rebind `HFConfigT`
to their own HF config type via the mixin. This means a type checker correctly infers
e.g. `Ministral3NativeConfig().to_hf_config() → Ministral3Config`.

`Mistral3NativeConfig` is a separate root (not inheriting from `MistralNativeConfig`)
because it is a composite VLM config, not a text model config.

#### No Code Duplication in Mistral3NativeConfig

`Mistral3NativeConfig.from_params_json` delegates text backbone detection to:
1. `_detect_text_model_type(params)` — returns `MistralModelType`
   based on key presence (`moe` → `MISTRAL4`, `yarn`/`quantization` → `MINISTRAL3`, else → `MISTRAL`)
2. `native_config_for_model_type(model_type, text_params)` — the same dispatcher used externally

`Mistral3NativeConfig.from_hf_config` similarly delegates to `native_config_from_hf_config`
with `text_hf.model_type`. This avoids duplicating detection logic.

#### Dispatcher Functions (with @overload and StrEnum)

Both dispatchers use `@overload` with `MistralModelType` enum members so that type
checkers can narrow the return type. A `str` fallback overload accepts plain strings
(e.g., `cls.model_type` from `PreTrainedConfig`) and returns the union type:

```python
@overload
def native_config_for_model_type(model_type: MistralModelType.MISTRAL, params: dict) -> MistralNativeConfig: ...
@overload
def native_config_for_model_type(model_type: MistralModelType.MINISTRAL3, params: dict) -> Ministral3NativeConfig: ...
@overload
def native_config_for_model_type(model_type: MistralModelType.MISTRAL4, params: dict) -> Mistral4NativeConfig: ...
@overload
def native_config_for_model_type(model_type: MistralModelType.MISTRAL3, params: dict) -> Mistral3NativeConfig: ...
@overload
def native_config_for_model_type(model_type: str, params: dict) -> MistralNativeConfig | ...: ...

def native_config_for_model_type(
    model_type: str, params: dict
) -> MistralNativeConfig | Ministral3NativeConfig | Mistral4NativeConfig | Mistral3NativeConfig:
    ...
```

Same pattern for `native_config_from_hf_config`. The `match`/`case` branches use
enum members (`MistralModelType.MISTRAL`, etc.), which compare equal to plain strings
because `StrEnum` inherits from `str`.

### Tests (30 total, all passing)

**File**: `tests/integrations/mistral/test_params_conversion.py`

All tests are pure Python (no `@require_torch`, no Hub access). Tests use pytest-style
classes (no `unittest.TestCase`), with `@pytest.fixture(scope="session")` from
`conftest.py`. Assertions use dataclass `__eq__` and HF config `__eq__` (full object
equality) rather than field-by-field checks. `TestDispatchers` uses
`@pytest.mark.parametrize` for the dispatch test.

| Test class | Count | Methods |
|---|---|---|
| `TestFP8NativeConfig` | 3 | `test_valid_construction`, `test_unsupported_format_raises`, `test_dynamic_scheme` |
| `TestMoeNativeConfig` | 1 | `test_defaults` |
| `TestMistralNativeConfig` | 6 | `test_from_params_json`, `test_from_params_json_minimal_required`, `test_from_params_json_missing_required_raises`, `test_from_params_json_ignores_unknown_keys`, `test_to_hf_config`, `test_roundtrip` |
| `TestMinistral3NativeConfig` | 5 | `test_from_params_json`, `test_from_params_json_no_quantization`, `test_to_hf_config_with_yarn`, `test_to_hf_config_with_fp8`, `test_roundtrip` |
| `TestMistral4NativeConfig` | 3 | `test_from_params_json`, `test_to_hf_config`, `test_roundtrip` |
| `TestMistral3NativeConfig` | 6 | `test_from_params_json`, `test_to_hf_config`, `test_roundtrip`, `test_backbone_auto_detection_base`, `test_backbone_auto_detection_yarn`, `test_backbone_auto_detection_moe` |
| `TestDispatchers` | 6 | `test_native_config_for_model_type_dispatches_correctly` (×4 parametrized), `test_native_config_for_model_type_unknown_raises`, `test_native_config_from_hf_config_unknown_raises` |

### Key Design Decisions

- **ABC mixin for type safety**: `NativeToHFConfigMixin[HFConfigT]` is an abstract
  generic mixin. Each concrete class binds `HFConfigT` to its specific HF config type.
  Subclasses rebind by listing the mixin again (e.g.
  `Ministral3NativeConfig(MistralNativeConfig, NativeToHFConfigMixin[Ministral3Config])`).
- **`@overload` dispatchers with `StrEnum`**: Both dispatcher functions use `@overload`
  with `MistralModelType` enum members for type narrowing, plus a `str` fallback
  overload for callers passing `cls.model_type` (which is `ClassVar[str]` on
  `PreTrainedConfig`). Since `StrEnum` inherits from `str`, enum members compare
  equal to plain strings in `match`/`case` and equality checks.
- **No code duplication**: `Mistral3NativeConfig.from_params_json` and `.from_hf_config`
  delegate to the dispatcher functions and `_detect_text_model_type` rather than
  reimplementing detection logic.
- **No mapping dicts or passthrough**: Every field is an explicit dataclass field.
  `from_params_json` reads directly from the dict. `to_hf_config` maps explicitly.
- **`rope_theta` extraction**: `_extract_rope_theta(config)` checks
  `config.rope_parameters["rope_theta"]` first, then falls back to `config.rope_theta`,
  then to `10000.0`. This handles HF configs where `rope_theta` is absorbed into
  `rope_parameters` during `__post_init__`.
- **FP8 validation**: `FP8NativeConfig.__post_init__` raises `ValueError` if
  `qformat_weight != "fp8_e4m3"`.
- **FP8 roundtrip**: `quantization` doesn't roundtrip via HF config attrs (HF uses
  `quantization_config` which isn't stored as individual config fields). Tests exclude
  FP8 from roundtrip assertions.

---

## Phase 2: Weight Key Conversion (Requires Torch) — IMPLEMENTED ✅

### What We Built

Factory functions that return lists of `WeightRenaming` and `WeightConverter` entries
for each model type, plus `FP8AwareMergeAndConcatenate` for expert fusion.

**File**: `src/transformers/integrations/mistral/weight_conversion.py`

### Tests to Write First

**File**: `tests/integrations/mistral/test_weight_conversion.py`

All tests use `@require_torch`, no Hub downloads.

**Imports needed in test file** (guarded by `if is_torch_available()`):
```python
import torch
from transformers.core_model_loading import PermuteForRope, WeightConverter, WeightRenaming
from transformers.integrations.mistral.weight_conversion import (
    FP8AwareMergeAndConcatenate,
    FP8AwareSplitAndUnstack,
    fp8_scale_renamings,
    mistral3_native_text_renamings,
    mistral3_native_vision_converters,
    mistral3_native_vision_renamings,
    mistral4_native_renamings,
    mistral_base_native_converters,
    mistral_base_native_renamings,
)
```

**Helpers**: `_source_target_pairs(entries)` extracts `(source, target)` tuples from
renaming/converter entries. `_apply_renamings(key, renamings)` chains `rename_source_key`
calls.

Test classes are grouped by class/module under test. Each renaming factory gets one
`test_renamings` method that checks concrete key→key mappings end-to-end (no
structural/isinstance checks — those are redundant busywork). The RoPE test verifies
the permutation is actually applied and is self-inverse for small `head_dim`.

| Test class | Count | Methods |
|---|---|---|
| `TestMistralBaseRenamings` | 2 | `test_renamings`, `test_rope_permutation_applied` |
| `TestFP8ScaleRenamings` | 1 | `test_renamings` |
| `TestMistral3Renamings` | 4 | `test_text_renamings_prefixed`, `test_vision_renamings`, `test_vision_converters_dotted_heads`, `test_hf_keys_pass_through` |
| `TestMistral4Renamings` | 1 | `test_renamings` |
| `TestFP8AwareMergeAndConcatenate` | 5 | `test_merge_bf16`, `test_merge_per_tensor_fp8`, `test_merge_blockwise_fp8`, `test_mismatched_expert_count_raises`, `test_single_expert` |
| `TestFP8AwareSplitAndUnstack` | 2 | `test_roundtrip_bf16`, `test_roundtrip_per_tensor_fp8` |

**Phase 2 total: 15 tests**

**Removed tests vs original plan (redundant structural checks):**
- `TestMistralBaseRenamings.test_count_and_patterns` — isinstance/any-in-sources; covered by `test_renamings`.
- `TestMistralBaseRenamings.test_converters_rope` — isinstance/len/has-op; replaced by `test_rope_permutation_applied` which tests actual behavior.
- `TestMistral4Renamings.test_cover_all_mla_keys` — target fragment presence; covered by concrete mappings in `test_renamings`.
- `TestMistral4Converters.test_expert_fusion` — converter list structure; covered by `TestFP8AwareMergeAndConcatenate`.

**Added vs original plan:**
- `TestMistralBaseRenamings.test_rope_permutation_applied` — calls `PermuteForRope.convert` through the actual converters with specific named keys (`attention.wq`, `attention.wk`), verifies the tensor is permuted and the permutation is self-inverse for `head_dim=4`.

**Merged vs original plan:**
- `TestMistral4Renamings.test_mla_keys` + non-MLA keys → single `test_renamings` covering structural, MLA, router, and shared-expert keys.

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

## Phase 3: PermuteForRope Fixes — IMPLEMENTED ✅ (folded into Phase 2)

### What We Built

Fixed `PermuteForRope` to support configurable `n_heads_attr` (dotted attribute paths)
and proper error handling. The merge-base version had a bug: `self.config.getattr(...)`
instead of `getattr(self.config, ...)`, and didn't support dotted paths or `reverse_op`.

**File**: `src/transformers/core_model_loading.py`

### Implementation (done during Phase 2)

Phase 2's converters require `PermuteForRope(n_heads_attr=...)`, so all Phase 3
changes were applied during Phase 2 implementation:

1. `PermuteForRope.__init__(self, n_heads_attr: str = "num_attention_heads")` — stores `self.n_heads_attr`
2. `_resolve_attr(self, config)` — splits `n_heads_attr` on `.`, walks attributes; raises `AttributeError` on missing segment
3. Fixed `_apply` to use `self._resolve_attr(self.config)` instead of `self.config.getattr(...)`
4. Added `reverse_op` property returning `PermuteForRope(n_heads_attr=self.n_heads_attr)`
5. Added `__repr__` showing `n_heads_attr`
6. Made `convert()` accept `config` as optional kwarg

### Tests (covered by Phase 2 tests)

The original Phase 3 planned a separate `TestPermuteForRope` class with 5 tests. These
are now covered by existing Phase 2 tests:

- `test_self_inverse` → `TestMistralBaseRenamings.test_rope_permutation_applied` (verifies double-apply == identity)
- `test_dotted_attr` → `TestMistral3Renamings.test_vision_converters_dotted_heads` (verifies `"vision_config.num_attention_heads"`)
- `test_reverse_op` → implicitly covered: `reverse_op` is used by `FP8AwareSplitAndUnstack` roundtrip tests
- `test_repr` → trivial, not worth a standalone test
- `test_missing_attr_raises` → can be added later if needed; the `_resolve_attr` path is exercised by all converter tests

**No additional tests needed. Phase 3 is complete.**

---

## Phase 4: Config Format Detection & Loading — IMPLEMENTED ✅

### What We Built

`MistralFormatConfig` — a `PreTrainedConfig` subclass that overrides `get_config_dict()`
to auto-detect and load from `params.json` when `config.json` is absent.

**File**: `src/transformers/integrations/mistral/config_format.py`
**Tests**: `tests/integrations/mistral/test_config_format.py`
**Shared fixtures**: `tests/integrations/mistral/conftest.py`
**19/19 tests passing.**

### Architecture

#### `MistralFormatConfig(PreTrainedConfig)` class

- `get_config_dict(cls, pretrained_model_name_or_path, **kwargs)` classmethod:
  - Pops `mistral_format` from kwargs
  - When `mistral_format` is `None` or `False`: tries `config.json` via `super().get_config_dict()`
  - When result is empty and `mistral_format is False`: raises `OSError`
  - When result is empty and `mistral_format is None`: falls through to `_get_config_dict_from_params_json()`
  - When `mistral_format is True`: goes directly to `_get_config_dict_from_params_json()`
- `_get_config_dict_from_params_json(cls, path, **kwargs)` classmethod:
  - `cached_file(path, "params.json")` → load JSON
  - `native_config_for_model_type(cls.model_type, params_dict).to_hf_config().to_dict()` → config_dict
  - `_detect_weight_file(path)` → set `transformers_weights` if consolidated found
  - Set `_loaded_from_mistral_format = True`
  - Return `(config_dict, kwargs)`
- `_detect_weight_file(cls, path, **kwargs)` classmethod:
  - Probes HF files first (`model.safetensors`, `model.safetensors.index.json`) → return `None` if found
  - Then probes consolidated files → return filename if found
  - Returns `None` when nothing found (not an error — weights may not be present yet)
- `_config_to_params_json(self)` method:
  - `native_config_from_hf_config(self.model_type, self)` → `dataclasses.asdict(native)`

Constants: `_PARAMS_JSON`, `_CONSOLIDATED_SINGLE`, `_CONSOLIDATED_INDEX`, `_HF_SINGLE`, `_HF_INDEX`

#### Circular import handling

`config_format.py` → `params_conversion.py` → `configuration_mistral.py` → `config_format.py`

Broken via lazy imports: `config_format.py` imports `native_config_for_model_type` and
`native_config_from_hf_config` inside the two methods that use them, not at module level.
All other imports are top-level.

#### Config class inheritance changes

Each config class now inherits from `MistralFormatConfig` instead of `PreTrainedConfig`:
- `MistralConfig(MistralFormatConfig)` — `configuration_mistral.py`
- `Ministral3Config(MistralFormatConfig)` — `configuration_ministral3.py`
- `Mistral3Config(MistralFormatConfig)` — `configuration_mistral3.py`
- `Mistral4Config(MistralFormatConfig)` — `configuration_mistral4.py`

### Tests (19 total, all passing)

**File**: `tests/integrations/mistral/test_config_format.py`

All tests are pure Python (no `@require_torch`, no Hub access). Tests use a single
pytest-style class `TestMistralFormat` (no `unittest.TestCase`), with
`@pytest.fixture(scope="session")` from `conftest.py` and `@pytest.mark.parametrize`
for multi-variant tests. Fixture names are resolved via `request.getfixturevalue()`.

| Group | Count | Methods |
|---|---|---|
| `_detect_weight_file` | 7 | `test_detect_weight_file` (×7 parametrized) |
| `get_config_dict` | 5 | `test_get_config_dict_prefers_config_json`, `test_get_config_dict_falls_back_to_params_json`, `test_get_config_dict_mistral_format_true_forces_params`, `test_get_config_dict_mistral_format_false_errors`, `test_get_config_dict_sets_transformers_weights` |
| `_config_to_params_json` | 3 | `test_config_to_params_json` (×3 parametrized) |
| `from_pretrained` | 4 | `test_from_pretrained` (×4 parametrized) |

### Test design

- **`test_detect_weight_file`**: Parametrized with 7 cases. Mocks `cached_file` to
  control which files "exist". Asserts `None` when HF weights found or nothing found,
  filename when consolidated found.
- **`get_config_dict` tests**: Use `mistral_params` fixture and `tmp_path`. Assert
  `isinstance(dict)` and `_loaded_from_mistral_format` presence/absence.
- **`test_config_to_params_json`**: Parametrized over `(config_cls, params_fixture,
  skip_keys)`. Loads from fixture via `from_pretrained`, roundtrips back, asserts
  every fixture key is preserved. `Ministral3Config` skips `quantization` (FP8
  doesn't roundtrip).
- **`test_from_pretrained`**: Parametrized over `(config_cls, params_fixture)`. Builds
  expected config via `config_cls.from_dict(...)` to match `from_pretrained` metadata.
  Asserts full config equality.

### Shared test fixtures

**File**: `tests/integrations/mistral/conftest.py`

Module-level constants (`MISTRAL_PARAMS`, etc.) and `@pytest.fixture(scope="session")`
wrappers (`mistral_params`, etc.) returning `deepcopy` to prevent cross-test mutation.
Pytest-style tests receive fixtures via argument injection; any remaining
`unittest.TestCase` tests can import the constants directly.

### Key Design Decisions

- **No torch dependency**: All config classes are pure Python. All Phase 4 tests run
  without torch.
- **`_detect_weight_file` returns `None`, never raises**: Two semantically different
  `None` cases (HF weights found = no override needed; nothing found = weights may not
  be present yet). Config loading should not fail when weights are absent.
- **Lazy imports for circular dependency**: Only the two methods that call
  `params_conversion` functions use lazy imports. Everything else is top-level.
- **Pytest-style classes with fixtures**: All test classes are plain classes (not
  `unittest.TestCase`). Fixtures injected via arguments. `@pytest.mark.parametrize`
  for multi-variant tests with `request.getfixturevalue()` for fixture resolution.

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

Test classes are grouped by class/module under test:

| Test class | Count | Methods |
|---|---|---|
| `TestConversionMappingRegistration` | 3 | `test_mistral`, `test_ministral3`, `test_mistral4` |
| `TestMistralFromPretrained` | 4 | `test_native_format`, `test_native_weights_match`, `test_hf_save_reload_roundtrip`, `test_native_weight_conversion_keys_correct` |
| `TestMinistral3FromPretrained` | 1 | `test_native_format` |
| `TestMistral4FromPretrained` | 2 | `test_native_format`, `test_hf_save_reload_roundtrip` |
| `TestFromPretrainedFormatSelection` | 3 | `test_prefers_hf_when_both_exist`, `test_mistral_format_true`, `test_mistral_format_false_no_hf` |

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

## Phase 6: save_pretrained Native Format + Roundtrip — IMPLEMENTED ✅

### What We Built

Support for `save_pretrained(save_format="mistral")` that saves in native Mistral format:
`params.json` + `consolidated.safetensors` with native key names. Also fixed the
`revert_weight_conversion` regex-key bug that blocks this.

**Files**:
- `src/transformers/modeling_utils.py` (add `save_format` parameter, minimal diff)
- `src/transformers/integrations/mistral/weight_conversion.py` (file renaming + `params.json` logic)
- `src/transformers/core_model_loading.py` (fix `revert_weight_conversion` / `WeightConverter.convert`)
- `src/transformers/configuration_utils.py` (strip `_loaded_from_mistral_format` on save)

### Tests to Write First

**File**: `tests/integrations/mistral/test_integration.py` (appended to Phase 5 tests)

Test classes are grouped by class/module under test (appended to test_integration.py):

| Test class | Count | Methods |
|---|---|---|
| `TestSavePretrained` | 5 | `test_default_hf_format`, `test_default_preserves_native_format`, `test_force_hf`, `test_force_mistral`, `test_invalid_save_format_raises` |
| `TestMistralSaveLoadRoundtrip` | 4 | `test_native_to_hf_to_native`, `test_hf_to_native_to_hf`, `test_config_native_to_hf_to_native`, `test_preserves_model_output` |
| `TestMistral4SaveLoadRoundtrip` | 2 | `test_native_to_hf_to_native`, `test_hf_to_native_to_hf` |
| `TestRevertWeightConversion` | 3 | `test_no_regex_keys`, `test_roundtrip`, `test_returns_all_keys` |

**Phase 6 total: 14 tests (appended to test_integration.py)**

### Implementation

1. **Fixed `revert_weight_conversion` / `WeightConverter.convert` regex-key bug**
   (`core_model_loading.py`):

   The root cause was in `WeightConverter.convert()`: the `StopIteration` fallback let
   regex pattern keys leak through as literal key names. Fixed by adding regex
   substitution to derive concrete keys from the full parameter name.

2. **Added `save_format` parameter to `save_pretrained`** (`modeling_utils.py`):
   - `save_format: str | None = None` — `"hf"`, `"mistral"`, or `None` (auto)
   - Validate: raise `ValueError` for unknown values
   - `save_format="hf"`: skip `revert_weight_conversion` (keep HF key names)
   - `save_format="mistral"`: revert keys to native, then call
     `save_native_mistral_format` for file renaming + `params.json`

3. **Moved `save_native_mistral_format` to `weight_conversion.py`**:
   The function that renames `model.safetensors` → `consolidated.safetensors` and writes
   `params.json` lives in `integrations/mistral/weight_conversion.py`, not in
   `modeling_utils.py`. This keeps the `modeling_utils.py` diff minimal (~16 added lines):
   - 1 top-level import
   - 1 new parameter + 4-line docstring + 2-line validation
   - 4-line `save_format="hf"` branch (skip revert)
   - 3-line `save_format="mistral"` post-save call
   The function inlines a local `_add_variant` helper to avoid circular imports
   (the canonical `_add_variant` is defined in `modeling_utils.py`).

4. **Stripped internal metadata from saved config** (`configuration_utils.py`):
   - Remove `_loaded_from_mistral_format` and `transformers_weights` from `config.json`
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

Test classes are grouped by model type under test:

| Test class | Count | Methods |
|---|---|---|
| `TestMinistral3SlowIntegration` | 4 | `test_native_vs_hf_config_match`, `test_native_weight_loading`, `test_save_reload_native_format`, `test_save_reload_hf_format` |
| `TestMistral3SlowIntegration` | 2 | `test_native_vs_hf_config_match`, `test_native_weight_loading` |
| `TestMistral4SlowIntegration` | 2 | `test_native_vs_hf_config_match`, `test_native_weight_loading` |

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
    weight_conversion.py           # Weight conversion factories + FP8 ops (Phase 2) + native save (Phase 6)

tests/integrations/
    __init__.py
    mistral/
        __init__.py
        conftest.py                # Shared fixture dicts (params.json-style)
        test_params_conversion.py  # Phase 1 tests (27 tests, pure Python)
        test_weight_conversion.py  # Phase 2+3 tests (15 tests, @require_torch)
        test_config_format.py      # Phase 4 tests (19 tests, pure Python)
        test_integration.py        # Phase 5+6 tests (27 tests, @require_torch, tiny models)
        test_slow_integration.py   # Phase 7 tests (8 tests, @slow, real Hub models)
```

### Modified Files

```
src/transformers/core_model_loading.py              # Phase 3: PermuteForRope fixes + Phase 6: revert bug fix
src/transformers/conversion_mapping.py              # Phase 5: Register all model types
src/transformers/modeling_utils.py                   # Phase 6: save_format parameter (minimal diff: ~16 lines)
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
| `test_params_conversion.py` | 30 | — | — | 30 |
| `test_weight_conversion.py` | 15 | — | — | 15 |
| `test_config_format.py` | — | 19 | — | 19 |
| `test_integration.py` | — | 27 | — | 27 |
| `test_slow_integration.py` | — | — | 8 | 8 |
| **Total** | **45** | **46** | **8** | **99** |

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
When w1 and w3 have different scales the reverse split loses precision because we
can't recover the original individual scales.

**Mitigation**: The roundtrip test uses identical scales for w1 and w3, and FP8-typed
inputs.  With equal scales the fused scale equals the originals (`max(s, s) == s`),
the rescaling ratio is 1.0, and FP8 bit-patterns survive the roundtrip exactly.
This still exercises the full per-tensor FP8 merge/split code path while allowing
exact equality assertions — no approximate comparison needed.

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
Phase 1 → Phase 2+3 → Phase 4 → Phase 5 → Phase 6 → Phase 7
  │           │          │         │         │         │
  ▼           ▼          ▼         ▼         ▼         ▼
Tests       Tests      Tests     Tests     Tests     Tests
  │           │          │         │         │         │
  ▼           ▼          ▼         ▼         ▼         ▼
Implement  Implement  Implement Implement Implement  Validate
  │           │          │         │         │
  ▼           ▼          ▼         ▼         ▼
Verify     Verify     Verify    Verify    Verify
```

Phase 3 (PermuteForRope fixes) was folded into Phase 2 because Phase 2's converters
depend on `n_heads_attr` and dotted attribute resolution.

Each phase is self-contained: write tests, see them fail, implement, see them pass.
Later phases build on earlier ones but don't modify them.

After all phases pass, run `make style` and `make fix-repo` before opening a PR.
