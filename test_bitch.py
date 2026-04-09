"""Compare Mistral3 special-token tokenization between MistralCommonBackend and PreTrainedTokenizerFast."""

from io import BytesIO

import httpx
from PIL import Image

from transformers.integrations.mistral import convert_tekken_tokenizer
from transformers.tokenization_mistral_common import MistralCommonBackend
from transformers.utils import cached_file


model_id = "mistralai/Mistral-Medium-3.5-128B"

# --- Path 1: MistralCommonBackend (native mistral-common) ---
tokenizer_mc = MistralCommonBackend.from_pretrained(model_id)

# --- Path 2: PreTrainedTokenizerFast built from tekken.json ---
tekken_file = cached_file(model_id, "tekken.json")
tokenizer_fast = convert_tekken_tokenizer(tokenizer_file=tekken_file)

print(f"MistralCommonBackend type      : {type(tokenizer_mc).__name__}")
print(f"PreTrainedTokenizerFast type   : {type(tokenizer_fast).__name__}")

# --- Fetch test image ---
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
with httpx.stream("GET", url) as response:
    image = Image.open(BytesIO(response.read()))

# MistralCommonBackend expects url/path/base64 for images, not raw PIL
messages_mc = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "how are you ?"},
            {"type": "image", "url": url},
        ],
    }
]

# Original format with PIL image object (what the user script had)
messages_orig = [
    {
        "role": "user",
        "content": [
            {"type": "text", "text": "how are you ?"},
            {"type": "image", "image": image},
        ],
    }
]

# =====================================================================
# Path 1 – MistralCommonBackend.apply_chat_template
# Handles images natively via mistral-common
# =====================================================================
mc_output = tokenizer_mc.apply_chat_template(messages_mc, return_tensors="pt", tokenize=True, return_dict=True)
mc_ids = mc_output["input_ids"][0].tolist()
mc_decoded = tokenizer_mc.decode(mc_ids, skip_special_tokens=False)

print("\n" + "=" * 70)
print("MistralCommonBackend")
print("=" * 70)
print(f"  length   : {len(mc_ids)}")
print(f"  input_ids: {mc_ids[:80]}{'…' if len(mc_ids) > 80 else ''}")
print(f"  decoded  : {mc_decoded[:300]}{'…' if len(mc_decoded) > 300 else ''}")

# =====================================================================
# Path 2 – PreTrainedTokenizerFast: tokenize the decoded MC output
# This simulates what the Processor path does: render special tokens
# as strings, then tokenize with the fast tokenizer
# =====================================================================
mc_text = tokenizer_mc.apply_chat_template(messages_mc, tokenize=False)
print("\n" + "=" * 70)
print("MistralCommonBackend text output (tokenize=False)")
print("=" * 70)
print(f"  text: {mc_text[:300]}{'…' if len(mc_text) > 300 else ''}")

# Now tokenize that text with the fast tokenizer (like the Processor would)
fast_encoded = tokenizer_fast(mc_text, add_special_tokens=False, return_tensors="pt")
fast_ids = fast_encoded["input_ids"][0].tolist()
fast_decoded = tokenizer_fast.decode(fast_ids, skip_special_tokens=False)

print("\n" + "=" * 70)
print("PreTrainedTokenizerFast (re-tokenizing MC text output)")
print("=" * 70)
print(f"  length   : {len(fast_ids)}")
print(f"  input_ids: {fast_ids[:80]}{'…' if len(fast_ids) > 80 else ''}")
print(f"  decoded  : {fast_decoded[:300]}{'…' if len(fast_decoded) > 300 else ''}")

# =====================================================================
# Comparison
# =====================================================================
print("\n" + "=" * 70)
print("Comparison: MC tokenized vs Fast re-tokenized")
print("=" * 70)
print(f"  Lengths match: {len(mc_ids) == len(fast_ids)}  (MC={len(mc_ids)}, Fast={len(fast_ids)})")
print(f"  IDs match    : {mc_ids == fast_ids}")

if mc_ids != fast_ids:
    max_diffs = 30
    diff_count = 0
    max_len = max(len(mc_ids), len(fast_ids))
    for i in range(max_len):
        a = mc_ids[i] if i < len(mc_ids) else "<missing>"
        b = fast_ids[i] if i < len(fast_ids) else "<missing>"
        if a != b:
            tok_a = tokenizer_mc.convert_ids_to_tokens(a) if isinstance(a, int) else a
            tok_b = tokenizer_fast.convert_ids_to_tokens(b) if isinstance(b, int) else b
            print(f"    pos {i:>5}: MC id={a!s:>6} ({tok_a!r:>20})  |  Fast id={b!s:>6} ({tok_b!r:>20})")
            diff_count += 1
            if diff_count >= max_diffs:
                remaining = sum(
                    1
                    for j in range(i + 1, max_len)
                    if (mc_ids[j] if j < len(mc_ids) else None) != (fast_ids[j] if j < len(fast_ids) else None)
                )
                if remaining:
                    print(f"    ... and {remaining} more differences")
                break

# =====================================================================
# Spot-check: are known special tokens single IDs or split?
# =====================================================================
special_tokens_to_check = ["[INST]", "[/INST]", "[IMG]", "[IMG_BREAK]", "[IMG_END]", "<s>", "</s>"]
print("\n" + "=" * 70)
print("Special token ID check")
print("=" * 70)
for tok in special_tokens_to_check:
    mc_id = tokenizer_mc.convert_tokens_to_ids(tok)
    fast_id = tokenizer_fast.convert_tokens_to_ids(tok)
    mc_encoded = tokenizer_mc.encode(tok, add_special_tokens=False)
    fast_encoded_ids = tokenizer_fast.encode(tok, add_special_tokens=False)
    match_flag = "✓" if mc_id == fast_id else "✗"
    split_mc = "SPLIT" if len(mc_encoded) > 1 else "ok"
    split_fast = "SPLIT" if len(fast_encoded_ids) > 1 else "ok"
    print(
        f"  {tok:>15}  MC_id={mc_id:>6}  Fast_id={fast_id:>6}  {match_flag}"
        f"  | encode MC={mc_encoded} ({split_mc})  Fast={fast_encoded_ids} ({split_fast})"
    )
