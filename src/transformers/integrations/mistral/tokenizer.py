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

"""Conversion between Mistral tekken tokenizers and HuggingFace tokenizer formats."""

import base64
import json
from functools import lru_cache
from typing import Any

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...convert_slow_tokenizer import bytes_to_unicode
from ...utils import logging


logger = logging.get_logger(__name__)

_MAP_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


class MistralConverter:
    """Converter from Mistral tekken BPE vocab to a HuggingFace `tokenizers.Tokenizer`.

    Supports two construction modes:
    - Direct: pass a pre-parsed `vocab` dict (e.g. from `mistral-common`).
    - From file: use `from_tekken_file()` to parse a raw `tekken.json` file.
    """

    def __init__(
        self,
        vocab: dict | None = None,
        pattern: str = r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
        add_prefix_space: bool = False,
        additional_special_tokens: list[AddedToken] | None = None,
        **kwargs,
    ):
        """Initialize a MistralConverter.

        Args:
            vocab (`dict` or `None`): Pre-parsed vocabulary mapping. Use
                ``from_tekken_file`` for file-based construction instead.
            pattern (`str`): Regex pattern for pre-tokenization.
            add_prefix_space (`bool`): Whether to add a leading space.
            additional_special_tokens (`list` or `None`): Extra special tokens.
        """
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens
        self._precomputed_vocab: dict[str, int] | None = None
        self._precomputed_merges: list[tuple[str, str]] | None = None
        self._tekken_metadata: dict[str, Any] | None = None

    @property
    def tekken_metadata(self) -> dict[str, Any]:
        """Non-vocabulary metadata from the original ``tekken.json`` file.

        Raises:
            AttributeError: If the instance was not created via ``from_tekken_file``.
        """
        if self._tekken_metadata is None:
            raise AttributeError(
                "`tekken_metadata` is only accessible when instance is created by `from_tekken_file` method."
            )
        return self._tekken_metadata

    @classmethod
    def from_tekken_file(
        cls,
        vocab_file: str,
        add_prefix_space: bool = False,
    ) -> "MistralConverter":
        """Parse a raw `tekken.json` file and return a ready-to-use converter.

        Reads the file, extracts the regex pattern and special tokens, then
        pre-computes `vocab` and `merges` with correct index offsets (special
        tokens occupy the first indices).

        Args:
            vocab_file (`str`): Path to a `tekken.json` file.
            add_prefix_space (`bool`): Whether to add a prefix space during tokenization.

        Returns:
            `MistralConverter`: A ready-to-use converter with pre-computed vocab and merges.
        """
        with open(vocab_file, encoding="utf-8") as f:
            untyped = json.load(f)

        pattern = untyped["config"]["pattern"]

        additional_special_tokens = [AddedToken(k["token_str"], special=True) for k in untyped["special_tokens"]]
        bpe_ranks_raw = untyped["vocab"]
        num_special = len(additional_special_tokens)

        bpe_ranks = [base64.b64decode(k["token_bytes"]) for k in bpe_ranks_raw]
        bpe_ranks_dict = {token: rank for rank, token in enumerate(bpe_ranks)}

        vocab, merges = cls._extract_merges(bpe_ranks_dict)

        # Offset vocab indices to account for special tokens occupying the first slots
        vocab = {k: v + num_special for k, v in vocab.items()}
        for idx, token in enumerate(additional_special_tokens):
            vocab[token.content] = idx

        instance = cls(
            vocab=None,
            pattern=pattern,
            add_prefix_space=add_prefix_space,
            additional_special_tokens=additional_special_tokens,
        )
        # Store pre-computed vocab and merges so tokenizer() can use them directly
        instance._precomputed_vocab = vocab
        instance._precomputed_merges = merges

        # Preserve tekken.json metadata so it can be reconstructed on save.
        instance._tekken_metadata = {k: v for k, v in untyped.items() if k != "vocab"}
        # Store which vocab entries had token_str=null so save_as_tekken can
        # restore them instead of unconditionally decoding from bytes.
        instance._tekken_metadata["_null_token_str_bytes"] = [
            entry["token_bytes"] for entry in bpe_ranks_raw if entry.get("token_str") is None
        ]

        return instance

    @staticmethod
    def _extract_merges(bpe_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract a unicode vocab and BPE merge list from byte-level BPE ranks.

        For each multi-byte token, tries all binary splits ``(token[:i], token[i:])``
        and keeps those where both halves exist in the vocabulary. Splits are sorted
        locally by ``(rank_left, rank_right)`` and globally by merged-token rank.

        Args:
            bpe_ranks (`dict[bytes, int]`): Mapping of byte-level tokens to their
                integer ranks in the BPE vocabulary.

        Returns:
            `tuple[dict[str, int], list[tuple[str, str]]]`: A pair of
            ``(vocab, merges)`` where vocab maps unicode token strings to ranks
            and merges is an ordered list of BPE merge pairs.
        """
        byte_encoder = bytes_to_unicode()

        @lru_cache
        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        vocab: dict[str, int] = {}
        all_merges: list[tuple[bytes, bytes, int]] = []

        for token, rank in bpe_ranks.items():
            vocab[token_bytes_to_string(token)] = rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]))
            all_merges.extend(local)

        all_merges = sorted(all_merges, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in all_merges]
        return vocab, merges

    def extract_vocab_merges_from_model(self, vocab: dict) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract vocab and merges from a pre-parsed vocab dict.

        Args:
            vocab (`dict`): Token-to-index mapping from the tekken vocabulary.

        Returns:
            `tuple[dict[str, int], list[tuple[str, str]]]`: A pair of
            ``(vocab_dict, merges_list)`` ready for a ``tokenizers.BPE`` model.
        """
        filtered = {}
        special_entries = {}
        for idx, (token, rank) in enumerate(vocab.items()):
            if token not in self.additional_special_tokens:
                filtered[token] = rank
            else:
                special_entries[token] = idx

        result_vocab, merges = self._extract_merges(filtered)

        for token, idx in special_entries.items():
            result_vocab[token] = idx

        return result_vocab, merges

    def tokenizer(self) -> Tokenizer:
        """Build a raw `tokenizers.Tokenizer` with BPE model (no pre/post-processing)."""
        if self._precomputed_vocab is not None:
            vocab_scores, merges = self._precomputed_vocab, self._precomputed_merges
        else:
            vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        """Build a fully configured `tokenizers.Tokenizer` with pre-tokenizer and decoder."""
        tokenizer = self.tokenizer()
        tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
            [
                pre_tokenizers.Split(Regex(self.pattern), behavior="isolated", invert=False),
                pre_tokenizers.ByteLevel(add_prefix_space=self.add_prefix_space, use_regex=False),
            ]
        )
        tokenizer.decoder = decoders.ByteLevel()
        tokenizer.add_special_tokens(self.additional_special_tokens)

        tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)

        return tokenizer
