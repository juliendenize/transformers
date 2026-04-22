# Fix Plan: Mistral Native Format Roundtrip Failures

## Failing Tests

1. **`TestMinistral3RealModelIntegration.test_save_recovers_native_format`** — `KeyError: 'apply_scale'` when comparing `params.json` sub-keys for the `yarn` dict.
2. **`TestMistral4RealModelIntegration.test_save_recovers_native_format`** — thousands of missing `qscale_act` / `qscale_weight` weight keys after HF roundtrip.

---

## Failure 1: `yarn.apply_scale` lost during params roundtrip

### Root cause

The native `params.json` contains `yarn.apply_scale` (a boolean, always `False` for current models). The `YarnArgs` dataclass in `params_conversion.py` only has `factor`, `original_max_position_embeddings`, `beta`, `alpha` — no `apply_scale` field. The field is silently dropped during parsing (`_parse_native_config_from_dict` line 586-591) and never recovered.

### How `apply_scale` maps to HF

In `_get_rope_parameters` (line 259-268), the forward conversion hardcodes `mscale_all_dim=1.0`. This is correct when `apply_scale=False` (no yarn attention scaling). The mapping is:

- `apply_scale=False` ↔ `mscale_all_dim == 1.0`
- `apply_scale=True` ↔ `mscale_all_dim != 1.0` (or absent)

All current Mistral models with `apply_scale` set it to `False`.

### Fix

**File: `src/transformers/integrations/mistral/params_conversion.py`**

1. Add `apply_scale: bool = False` to the `YarnArgs` dataclass.
2. In `_parse_native_config_from_dict`, extract `apply_scale` from `yarn_dict`:
   ```python
   YarnArgs(
       factor=yarn_dict["factor"],
       original_max_position_embeddings=yarn_dict["original_max_position_embeddings"],
       beta=yarn_dict["beta"],
       alpha=yarn_dict["alpha"],
       apply_scale=yarn_dict.get("apply_scale", False),
   )
   ```
3. In `_extract_yarn` (reverse path), infer `apply_scale` from the HF rope params:
   ```python
   rope_mscale_all_dim = rope_params.get("mscale_all_dim")
   apply_scale = rope_mscale_all_dim is None or rope_mscale_all_dim != 1.0
   ```
4. In `_get_rope_parameters` (forward path), set `mscale_all_dim` based on `apply_scale`:
   - When `apply_scale=False`: keep `mscale_all_dim=1.0` (current behavior).
   - When `apply_scale=True`: omit `mscale_all_dim` or set a non-1.0 value (to be determined — currently all models use `False`).

**File: `tests/integrations/mistral/conftest.py`**

5. Add `"apply_scale": False` to `MINISTRAL3_PARAMS["yarn"]` fixture.

**File: `tests/integrations/mistral/test_params_conversion.py`**

6. Update `YarnArgs(...)` constructions in test fixtures to include `apply_scale=False`.

---

## Failure 2: Mistral4 FP8 scale keys lost during weight roundtrip

### Root cause

Two interacting issues in `convert_state_dict_to_native`:

**Issue A — FP8 scale operations don't expand `*` to expert indices.**
`FP8ScaleFusionSplit` and `DuplicateAndSplit` return output keys with unexpanded `*` (e.g. `"experts.*.w1.weight_scale_inv"`) and values as `list[Tensor]`. In contrast, `SplitModulelist` expands `*` to concrete indices (`"experts.0.w1.weight"`, `"experts.1.w1.weight"`, ...) and returns individual tensors.

**Issue B — Suffix mismatch between converter output and renamed key.**
The reverse flow in `convert_state_dict_to_native`:
1. Converter renames HF key using its target pattern → e.g. `experts.*.w1.weight_scale_inv`
2. FP8 renamings fire next, converting suffix → `experts.*.w1.qscale_weight`
3. Converter operations run, producing output keys using the converter's target patterns → still `weight_scale_inv` suffix
4. `WeightConverter.convert()` tries to match output keys against `full_name` (from step 2, with `qscale_weight` suffix) — substring and regex both fail

Result: per-expert FP8 scale keys end up mangled (wrong names, unexpanded `*`, only first expert's tensor kept via `param[0]`).

Regular weight keys (`w1.weight`, `w2.weight`, etc.) work because `SplitModulelist` expands `*` and the `.weight` suffix is unaffected by FP8 renamings.

### Fix (at converter level)

**File: `src/transformers/integrations/mistral/weight_conversion.py`**

1. **Fix `FP8ScaleFusionSplit.convert()`** — expand `*` to concrete expert indices in output keys, returning `dict[str, Tensor]` (individual tensors) instead of `dict[str, list[Tensor]]`.

2. **Fix `DuplicateAndSplit.convert()`** — same: expand `*` and return individual tensors per expert.

3. **Create `MistralFP8WeightConverter` subclass** that overrides `_reverse_multi_source` and `_reverse_single_source` to apply the FP8 suffix renaming (`weight_scale_inv` → `qscale_weight`, `activation_scale` → `qscale_act`) to the reversed target patterns at construction time. This ensures operations produce output keys that match the already-renamed `full_name`.

   The suffix map:
   ```python
   _HF_TO_NATIVE_FP8_SUFFIX = {
       "weight_scale_inv": "qscale_weight",
       "activation_scale": "qscale_act",
   }
   ```

4. **Update `mistral4_native_converters()`** — use `MistralFP8WeightConverter` for all six FP8 scale entries (lines 799-823: `gate_up_proj_scale_inv`, `gate_up_proj_activation_scale`, `down_proj_scale_inv`, `down_proj_activation_scale`).

### Affected converters

| Forward source patterns | Forward target | Reverse op | Issue |
|---|---|---|---|
| `experts.*.w1.weight_scale_inv` + `experts.*.w3.weight_scale_inv` | `mlp.experts.gate_up_proj_scale_inv` | `FP8ScaleFusionSplit` | A + B |
| `experts.*.w1.activation_scale` + `experts.*.w3.activation_scale` | `mlp.experts.gate_up_proj_activation_scale` | `DuplicateAndSplit` | A + B |
| `experts.*.w2.weight_scale_inv` | `mlp.experts.down_proj_scale_inv` | `SplitModulelist` | B only |
| `experts.*.w2.activation_scale` | `mlp.experts.down_proj_activation_scale` | `SplitModulelist` | B only |

---

## Additional cleanup

### `_config_to_params_json` output

**File: `src/transformers/integrations/mistral/config_format.py`**

The `dataclasses.asdict(native)` output currently includes:
- `quantization_config`: HF-format object (should not appear in native `params.json`)
- `_SUPPORTED_SCHEMES`: internal `frozenset` from `QuantizationArgs` (not JSON-serializable, should not appear)

Fix: filter out HF-only and internal fields from the output dict before returning. Alternatively, exclude `_SUPPORTED_SCHEMES` from `QuantizationArgs` dataclass fields (make it a class variable instead).

### `QuantizationArgs._SUPPORTED_SCHEMES`

**File: `src/transformers/integrations/mistral/params_conversion.py`**

`_SUPPORTED_SCHEMES` is declared as a dataclass field with a default. It should be a `ClassVar` to avoid being included in `dataclasses.asdict()` output:
```python
_SUPPORTED_SCHEMES: ClassVar[frozenset[str]] = frozenset({"TENSOR"})
```

### Quantization roundtrip

The reverse path (`_hf_*_to_native` functions) currently stores `quantization_config` (HF format) instead of converting back to native `quantization` (with `QuantizationArgs`). For a proper Mistral-format save, the HF `quantization_config` should be converted back to native `QuantizationArgs`.

Add a reverse helper:
```python
_REVERSE_QUANTIZATION_SCHEME_MAP = {v: k for k, v in _QUANTIZATION_SCHEME_MAP.items()}

def _hf_quant_config_to_native(hf_config: PreTrainedConfig) -> QuantizationArgs | None:
    quant_cfg = _extract_hf_quantization_config(hf_config)
    if quant_cfg is None:
        return None
    qc = quant_cfg.to_dict()
    if qc.get("quant_method") != "fp8":
        return None
    scheme = _REVERSE_QUANTIZATION_SCHEME_MAP.get(qc.get("activation_scheme", "static"), "TENSOR")
    return QuantizationArgs(qformat_weight=QFormat.FP8_E4M3, qscheme_act=scheme)
```

Update all four `_hf_*_to_native` functions to set `quantization=_hf_quant_config_to_native(hf_config)` instead of (or in addition to) `quantization_config=...`.

Update `_config_to_params_json` to strip `quantization_config` from the output dict.

Remove `"quantization"` from `_NON_ROUNDTRIPPABLE_PARAMS_KEYS` in the test once quantization roundtrips correctly.

---

## Test updates

**File: `tests/integrations/mistral/test_integration.py`**

- Remove `"quantization"` from `_NON_ROUNDTRIPPABLE_PARAMS_KEYS` (once quantization roundtrips).
- The `params.json` comparison loop should tolerate extra sub-keys in the original that the reverse path intentionally drops (currently none after the fixes above, but could use a `_NON_ROUNDTRIPPABLE_PARAMS_SUB_KEYS` set if needed in the future).

**File: `tests/integrations/mistral/test_params_conversion.py`**

- Update roundtrip tests to assert `restored.quantization is not None` and `restored.quantization_config is None` (opposite of current assertions).
- Update `YarnArgs` fixtures to include `apply_scale=False`.

**File: `tests/integrations/mistral/conftest.py`**

- Add `"apply_scale": False` to `MINISTRAL3_PARAMS["yarn"]`.
