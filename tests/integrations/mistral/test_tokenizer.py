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

"""Tests for MistralConverter: tekken.json parsing and HuggingFace tokenizer conversion.

MistralConverter.converted() returns a raw tokenizers.Tokenizer — it does NOT
add BOS/EOS. Special-token injection is the responsibility of the higher-level
wrapper (PreTrainedTokenizerFast or MistralCommonBackend). Cross-backend tests
therefore compare raw encoding (no special tokens) and decoding only.
"""

import base64
import json
import tempfile
import unittest
from pathlib import Path

from huggingface_hub import hf_hub_download

from transformers.integrations.mistral import MistralConverter
from transformers.testing_utils import require_mistral_common, slow
from transformers.utils.import_utils import is_mistral_common_available


if is_mistral_common_available():
    from transformers.tokenization_mistral_common import MistralCommonBackend


_NUM_SPECIAL_TOKENS = 20

_FAKE_TEKKEN_SPECIAL_TOKENS = [
    {"rank": 0, "token_str": "<unk>", "is_control": True},
    {"rank": 1, "token_str": "<s>", "is_control": True},
    {"rank": 2, "token_str": "</s>", "is_control": True},
    {"rank": 3, "token_str": "[INST]", "is_control": True},
    {"rank": 4, "token_str": "[/INST]", "is_control": True},
    {"rank": 5, "token_str": "[AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 6, "token_str": "[/AVAILABLE_TOOLS]", "is_control": True},
    {"rank": 7, "token_str": "[TOOL_RESULTS]", "is_control": True},
    {"rank": 8, "token_str": "[/TOOL_RESULTS]", "is_control": True},
    {"rank": 9, "token_str": "[TOOL_CALLS]", "is_control": True},
    {"rank": 10, "token_str": "[IMG]", "is_control": True},
    {"rank": 11, "token_str": "<pad>", "is_control": True},
    {"rank": 12, "token_str": "[IMG_BREAK]", "is_control": True},
    {"rank": 13, "token_str": "[IMG_END]", "is_control": True},
    {"rank": 14, "token_str": "[PREFIX]", "is_control": True},
    {"rank": 15, "token_str": "[MIDDLE]", "is_control": True},
    {"rank": 16, "token_str": "[SUFFIX]", "is_control": True},
    {"rank": 17, "token_str": "[SYSTEM_PROMPT]", "is_control": True},
    {"rank": 18, "token_str": "[/SYSTEM_PROMPT]", "is_control": True},
    {"rank": 19, "token_str": "[TOOL_CONTENT]", "is_control": True},
]

_FAKE_TEKKEN_PATTERN = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+"""

# 256 byte-level BPE tokens + 20 special tokens = full single-byte coverage.
_FULL_BYTE_VOCAB = 256 + _NUM_SPECIAL_TOKENS

# Diverse test strings used across all test classes.
_TEST_STRINGS = [
    "Hello, world!",
    "Bonjour le monde!",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "The quick brown fox jumps over the lazy dog.",
    "🎉 Unicode: café, naïve, résumé",
    "   Multiple   spaces   and\ttabs\nand\nnewlines",
    "12345 + 67890 = 80235",
    "!@#$%^&*()",
    "  leading and trailing  ",
    "MiXeD CaSe TeXt",
    "a",
]

_MINISTRAL_REPO = "mistralai/Ministral-3-3B-Instruct-2512"


def _build_fake_tekken_json(
    directory: Path,
    vocab_size: int = _FULL_BYTE_VOCAB,
    image_config: dict | None = None,
) -> Path:
    """Build a minimal tekken.json for testing."""
    num_bpe = vocab_size - _NUM_SPECIAL_TOKENS

    vocab_list: list[dict] = []
    for rank in range(num_bpe):
        raw_byte = bytes([rank % 256])
        vocab_list.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(raw_byte).decode("ascii"),
                "token_str": None,
            }
        )

    tekken_data: dict = {
        "vocab": vocab_list,
        "special_tokens": _FAKE_TEKKEN_SPECIAL_TOKENS,
        "config": {
            "pattern": _FAKE_TEKKEN_PATTERN,
            "num_vocab_tokens": num_bpe,
            "default_vocab_size": vocab_size,
            "default_num_special_tokens": _NUM_SPECIAL_TOKENS,
            "version": "v3",
        },
        "version": 1,
        "type": "tekken",
    }

    if image_config is not None:
        tekken_data["image"] = image_config

    output_path = directory / "tekken.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tekken_data, f, ensure_ascii=False)

    return output_path


class TestMistralConverter(unittest.TestCase):
    """Unit tests for MistralConverter using a synthetic tekken.json."""

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        cls._tekken_path = _build_fake_tekken_json(Path(cls._tmp_dir.name))
        cls._converter = MistralConverter.from_tekken_file(str(cls._tekken_path))
        cls._tokenizer = cls._converter.converted()

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def test_from_tekken_file_sets_precomputed_fields(self):
        self.assertIsNotNone(self._converter._precomputed_vocab)
        self.assertIsNotNone(self._converter._precomputed_merges)
        self.assertIsNotNone(self._converter._tekken_metadata)

    def test_converted_produces_working_tokenizer(self):
        ids = self._tokenizer.encode("a b c").ids
        self.assertIsInstance(ids, list)
        self.assertGreater(len(ids), 0)

    def test_roundtrip_encode_decode(self):
        for text in ["hello world", "abc 123", "test"]:
            encoded = self._tokenizer.encode(text)
            decoded = self._tokenizer.decode(encoded.ids)
            self.assertEqual(decoded, text, f"Roundtrip failed for {text!r}")

    def test_special_tokens_in_vocab(self):
        vocab = self._tokenizer.get_vocab()
        for entry in _FAKE_TEKKEN_SPECIAL_TOKENS:
            self.assertIn(entry["token_str"], vocab, f"Special token {entry['token_str']!r} missing")

    def test_vocab_size(self):
        self.assertEqual(self._tokenizer.get_vocab_size(), _FULL_BYTE_VOCAB)

    def test_tekken_metadata_content(self):
        metadata = self._converter.tekken_metadata
        self.assertEqual(metadata["version"], 1)
        self.assertEqual(metadata["type"], "tekken")
        self.assertIn("config", metadata)
        self.assertEqual(metadata["config"]["version"], "v3")
        self.assertNotIn("vocab", metadata, "Metadata should not contain the vocab itself")

    def test_tekken_metadata_raises_without_from_file(self):
        converter = MistralConverter()
        with self.assertRaises(AttributeError):
            _ = converter.tekken_metadata


@require_mistral_common
class TestMistralConverterVsCommonBackend(unittest.TestCase):
    """Compare MistralConverter raw encoding/decoding with MistralCommonBackend on a synthetic tekken.json.

    MistralConverter.converted() does NOT add BOS/EOS — that is the wrapper's job.
    All comparisons use add_special_tokens=False on MistralCommonBackend.
    """

    @classmethod
    def setUpClass(cls) -> None:
        cls._tmp_dir = tempfile.TemporaryDirectory()
        tekken_path = _build_fake_tekken_json(Path(cls._tmp_dir.name))

        converter = MistralConverter.from_tekken_file(str(tekken_path))
        cls.hf_tokenizer = converter.converted()
        cls.mc_tokenizer = MistralCommonBackend(tokenizer_path=str(tekken_path))

    @classmethod
    def tearDownClass(cls) -> None:
        cls._tmp_dir.cleanup()

    def test_encode_matches(self) -> None:
        for text in _TEST_STRINGS:
            hf_ids = self.hf_tokenizer.encode(text).ids
            mc_ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Encoding mismatch for {text!r}")

    def test_decode_matches(self) -> None:
        for text in _TEST_STRINGS:
            ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            hf_decoded = self.hf_tokenizer.decode(ids)
            mc_decoded = self.mc_tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode mismatch for {text!r}")

    def test_vocab_size(self) -> None:
        self.assertEqual(self.hf_tokenizer.get_vocab_size(), self.mc_tokenizer.vocab_size)


@require_mistral_common
@slow
class TestMistralConverterIntegration(unittest.TestCase):
    """Integration tests with real tekken.json from mistralai/Ministral-3-3B-Instruct-2512.

    MistralConverter.converted() returns a raw tokenizers.Tokenizer without
    BOS/EOS injection. All encoding comparisons use add_special_tokens=False
    on MistralCommonBackend to compare at the same abstraction level.
    """

    @classmethod
    def setUpClass(cls) -> None:
        tekken_path = hf_hub_download(_MINISTRAL_REPO, "tekken.json")

        converter = MistralConverter.from_tekken_file(tekken_path)
        cls.hf_tokenizer = converter.converted()
        cls.mc_tokenizer = MistralCommonBackend(tokenizer_path=tekken_path)

    # ── Vocabulary ──────────────────────────────────────────────────────

    def test_vocab_size(self) -> None:
        self.assertEqual(self.hf_tokenizer.get_vocab_size(), self.mc_tokenizer.vocab_size)

    def test_full_vocab_decode_single_token_matches(self) -> None:
        """Decoding every single token ID (skip_special_tokens=True) produces the same string."""
        mismatches = []
        for token_id in range(self.mc_tokenizer.vocab_size):
            hf_decoded = self.hf_tokenizer.decode([token_id], skip_special_tokens=True)
            mc_decoded = self.mc_tokenizer.decode([token_id], skip_special_tokens=True)
            if hf_decoded != mc_decoded:
                mismatches.append((token_id, hf_decoded, mc_decoded))
        self.assertEqual(mismatches, [], f"Found {len(mismatches)} decode mismatches (first 10): {mismatches[:10]}")

    def test_special_tokens_ids(self) -> None:
        for token_str, attr in {"<s>": "bos", "</s>": "eos", "<unk>": "unk", "<pad>": "pad"}.items():
            hf_id = self.hf_tokenizer.token_to_id(token_str)
            mc_id = getattr(self.mc_tokenizer, f"{attr}_token_id")
            self.assertIsNotNone(hf_id, f"HF tokenizer missing {token_str}")
            self.assertIsNotNone(mc_id, f"MC tokenizer missing {attr}_token_id")
            self.assertEqual(hf_id, mc_id, f"{token_str} ID mismatch: HF={hf_id} MC={mc_id}")

    # ── Encode ──────────────────────────────────────────────────────────

    def test_encode(self) -> None:
        for text in _TEST_STRINGS:
            hf_ids = self.hf_tokenizer.encode(text).ids
            mc_ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Encoding mismatch for {text!r}")

    def test_encode_long_text(self) -> None:
        long_text = "The quick brown fox jumps over the lazy dog. " * 100
        hf_ids = self.hf_tokenizer.encode(long_text).ids
        mc_ids = self.mc_tokenizer.encode(long_text, add_special_tokens=False)
        self.assertEqual(hf_ids, mc_ids)
        self.assertGreater(len(hf_ids), 100, "Long text should produce many tokens")

    def test_encode_multilingual(self) -> None:
        texts = [
            "日本語のテスト",  # Japanese
            "Привет мир",  # Russian
            "مرحبا بالعالم",  # Arabic
            "你好世界",  # Chinese
            "한국어 테스트",  # Korean
            "Ñoño español",  # Spanish with diacritics
            "Ελληνικά",  # Greek
        ]
        for text in texts:
            hf_ids = self.hf_tokenizer.encode(text).ids
            mc_ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Multilingual encoding mismatch for {text!r}")

    def test_encode_code_snippets(self) -> None:
        snippets = [
            "import torch\nmodel = torch.nn.Linear(10, 20)",
            "for i in range(100):\n    print(f'{i=}')",
            "class Foo:\n    def __init__(self):\n        self.x = 42",
            "// C++ comment\nint main() { return 0; }",
            "SELECT * FROM users WHERE id = 1;",
            '{"key": "value", "nested": {"a": [1, 2, 3]}}',
        ]
        for text in snippets:
            hf_ids = self.hf_tokenizer.encode(text).ids
            mc_ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            self.assertEqual(hf_ids, mc_ids, f"Code encoding mismatch for {text!r}")

    # ── Decode ──────────────────────────────────────────────────────────

    def test_decode(self) -> None:
        """Decode token IDs (no special tokens) — both backends produce the same string."""
        for text in _TEST_STRINGS:
            ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            hf_decoded = self.hf_tokenizer.decode(ids)
            mc_decoded = self.mc_tokenizer.decode(ids, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode mismatch for {text!r}")

    def test_decode_with_special_token_ids(self) -> None:
        """Decode sequences that contain BOS/EOS IDs — skip_special_tokens strips them equally."""
        bos_id = self.hf_tokenizer.token_to_id("<s>")
        eos_id = self.hf_tokenizer.token_to_id("</s>")
        for text in _TEST_STRINGS:
            ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            ids_with_special = [bos_id] + ids + [eos_id]

            hf_decoded = self.hf_tokenizer.decode(ids_with_special, skip_special_tokens=True)
            mc_decoded = self.mc_tokenizer.decode(ids_with_special, skip_special_tokens=True)
            self.assertEqual(hf_decoded, mc_decoded, f"Decode skip BOS+EOS mismatch for {text!r}")

    def test_encode_decode_roundtrip(self) -> None:
        """Encode then decode should recover the original text in both backends."""
        for text in _TEST_STRINGS:
            if not text:
                continue
            hf_ids = self.hf_tokenizer.encode(text).ids
            hf_roundtrip = self.hf_tokenizer.decode(hf_ids)
            mc_roundtrip = self.mc_tokenizer.decode(hf_ids, skip_special_tokens=True)
            self.assertEqual(hf_roundtrip, text, f"HF roundtrip failed for {text!r}")
            self.assertEqual(mc_roundtrip, text, f"MC roundtrip failed for {text!r}")

    # ── Token-level ─────────────────────────────────────────────────────

    def test_per_token_decode_matches(self) -> None:
        """Decoding each token individually should produce the same string in both backends."""
        for text in _TEST_STRINGS:
            ids = self.mc_tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            for token_id in ids:
                hf_decoded = self.hf_tokenizer.decode([token_id], skip_special_tokens=True)
                mc_decoded = self.mc_tokenizer.decode([token_id], skip_special_tokens=True)
                self.assertEqual(hf_decoded, mc_decoded, f"Per-token decode mismatch for id={token_id} in {text!r}")


if __name__ == "__main__":
    unittest.main()
