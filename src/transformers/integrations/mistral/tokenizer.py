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
import os
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE

from ...convert_slow_tokenizer import bytes_to_unicode
from ...tokenization_utils_tokenizers import PreTrainedTokenizerFast
from ...utils import cached_file, logging, requires_backends
from ...utils.import_utils import is_mistral_common_available


if TYPE_CHECKING:
    from ...models.pixtral.processing_pixtral import PixtralProcessor

logger = logging.get_logger(__name__)


def resolve_mistral_format(
    pretrained_model_name_or_path: str | os.PathLike,
    mistral_format: bool | None = None,
    **cache_kwargs,
) -> tuple[bool, str | None]:
    """Resolve whether to use `MistralCommonBackend` for tokenization.

    Probes for `tekken.json` in the checkpoint directory and checks whether
    `mistral-common` is installed to determine the appropriate tokenizer backend.

    Args:
        pretrained_model_name_or_path (`str` or `os.PathLike`):
            This can be either:

            - a string, the *model id* of a pretrained model hosted on huggingface.co.
            - a path to a *directory* containing model files.
        mistral_format (`bool`, *optional*):
            Tri-state control for tokenizer backend selection:

            - `True` — force `MistralCommonBackend` (raises `ImportError` if `mistral-common`
              is not installed, raises `OSError` if `tekken.json` is not found).
            - `False` — force standard HuggingFace tokenizer.
            - `None` — auto-detect based on `tekken.json` presence and `mistral-common`
              availability.
        **cache_kwargs:
            Forwarded to [`~utils.cached_file`] (e.g. `cache_dir`, `force_download`,
            `local_files_only`, `revision`, `token`).

    Returns:
        `tuple[bool, str | None]`: A tuple of `(use_mistral_format, tekken_file_path)` where
        `use_mistral_format` indicates whether `MistralCommonBackend` should be used, and
        `tekken_file_path` is the resolved path to `tekken.json` (or `None`).

    Raises:
        ImportError: If `mistral_format=True` and `mistral-common` is not installed.
        OSError: If `mistral_format=True` and `tekken.json` cannot be found.
    """
    if mistral_format is False:
        return (False, None)

    if mistral_format is True:
        if not is_mistral_common_available():
            raise ImportError(
                "mistral_format=True requires `mistral-common`. Install it with: pip install mistral-common"
            )
        tekken_file = cached_file(
            pretrained_model_name_or_path,
            "tekken.json",
            _raise_exceptions_for_missing_entries=False,
            _raise_exceptions_for_connection_errors=False,
            **cache_kwargs,
        )
        if tekken_file is None:
            raise OSError(
                f"Cannot find 'tekken.json' at '{pretrained_model_name_or_path}'. "
                "Set `mistral_format=False` to use standard HuggingFace files instead."
            )
        return (True, tekken_file)

    # mistral_format is None: auto-detect
    if not is_mistral_common_available():
        return (False, None)

    tekken_file = cached_file(
        pretrained_model_name_or_path,
        "tekken.json",
        _raise_exceptions_for_missing_entries=False,
        _raise_exceptions_for_connection_errors=False,
        **cache_kwargs,
    )
    return (tekken_file is not None, tekken_file)


_MAP_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


class MistralConverter:
    """Converter from Mistral tekken BPE vocab to a HuggingFace `tokenizers.Tokenizer`."""

    def __init__(self, vocab_file: str, add_prefix_space: bool = False, **kwargs):
        """Parse a raw `tekken.json` file into a ready-to-use converter.

        Matches `mistral_common`'s `Tekkenizer.from_file` vocab and special-token layout.

        Args:
            vocab_file (`str`): Path to a `tekken.json` file.
            add_prefix_space (`bool`): Whether to add a leading space during tokenization.
        """
        self._parse_tekken_file(vocab_file, add_prefix_space)

    def _parse_tekken_file(self, vocab_file: str, add_prefix_space: bool) -> None:
        """Parse a tekken.json file and set all instance attributes.

        Args:
            vocab_file (`str`): Path to a `tekken.json` file.
            add_prefix_space (`bool`): Whether to add a leading space during tokenization.
        """
        with open(vocab_file, encoding="utf-8") as f:
            untyped = json.load(f)

        config = untyped["config"]
        pattern = config["pattern"]

        vocab_size = config.get("default_vocab_size")
        num_special_tokens = config.get("default_num_special_tokens")

        filler_template = "<SPECIAL_{id}>"

        special_tokens_dicts = untyped.get("special_tokens")
        if special_tokens_dicts is None:
            # Old tekken format has no special_tokens key; use mistral-common's defaults.
            requires_backends(self, ["mistral-common"])
            from mistral_common.tokens.tokenizers.tekken import Tekkenizer

            filler_template = getattr(Tekkenizer, "SPECIAL_TOKEN_TEMPLATE", filler_template)
            special_tokens_dicts = list(Tekkenizer.DEPRECATED_SPECIAL_TOKENS)

        special_tokens_dicts = [
            {"rank": entry["rank"], "token_str": str(getattr(entry["token_str"], "value", entry["token_str"]))}
            for entry in special_tokens_dicts
        ]

        if num_special_tokens is None:
            num_special_tokens = len(special_tokens_dicts)

        special_filler = [
            {"rank": i, "token_str": filler_template.format(id=i)}
            for i in range(len(special_tokens_dicts), num_special_tokens)
        ]
        special_tokens_dicts = special_tokens_dicts + special_filler

        additional_special_tokens = [AddedToken(entry["token_str"], special=True) for entry in special_tokens_dicts]

        # Keep full pre-truncation list for metadata (needed by save_as_tekken).
        bpe_ranks_raw_all = untyped["vocab"]
        # Drop padded vocab: keep only the real tokens (matches mistral-common).
        bpe_ranks_raw = bpe_ranks_raw_all
        if vocab_size is not None:
            inner_vocab_size = vocab_size - num_special_tokens
            bpe_ranks_raw = bpe_ranks_raw[:inner_vocab_size]

        bpe_ranks = [base64.b64decode(k["token_bytes"]) for k in bpe_ranks_raw]
        bpe_ranks_dict = {token: rank for rank, token in enumerate(bpe_ranks)}

        vocab, merges = self._extract_merges(bpe_ranks_dict)

        vocab = {k: v + num_special_tokens for k, v in vocab.items()}
        for entry in special_tokens_dicts:
            vocab[entry["token_str"]] = entry["rank"]

        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens
        self._precomputed_vocab = vocab
        self._precomputed_merges = merges

        # Preserve tekken.json metadata so it can be reconstructed on save.
        self.tekken_metadata = {k: v for k, v in untyped.items() if k != "vocab"}
        # Store which vocab entries had token_str=null so save_as_tekken can
        # restore them instead of unconditionally decoding from bytes.
        self.tekken_metadata["_null_token_str_bytes"] = [
            entry["token_bytes"] for entry in bpe_ranks_raw_all if entry.get("token_str") is None
        ]

    @staticmethod
    def _extract_merges(bpe_ranks: dict[bytes, int]) -> tuple[dict[str, int], list[tuple[str, str]]]:
        """Extract a unicode vocab and ordered BPE merge list from byte-level BPE ranks."""
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

    def tokenizer(self) -> Tokenizer:
        """Build a raw `tokenizers.Tokenizer` with BPE model (no pre/post-processing)."""
        tokenizer = Tokenizer(BPE(self._precomputed_vocab, self._precomputed_merges, fuse_unk=False))
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


def convert_tekken_tokenizer(tokenizer_file: str) -> PreTrainedTokenizerFast:
    """Build a `PreTrainedTokenizerFast` from a Mistral `tekken.json` file.

    Args:
        tokenizer_file (`str`): Path to the `tekken.json` vocabulary file.

    Returns:
        Configured fast tokenizer with BPE model and special token mappings.
    """
    converter = MistralConverter(vocab_file=tokenizer_file, add_prefix_space=False)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=converter.converted(),
        tekken_metadata=converter.tekken_metadata,
        **_MAP_SPECIALS,
    )
    return fast


def _unicode_to_bytes() -> dict[str, int]:
    """Invert `bytes_to_unicode()` to map unicode chars back to byte values."""
    return {v: k for k, v in bytes_to_unicode().items()}


def _bpe_token_to_bytes(token_str: str, decoder: dict[str, int]) -> bytes:
    """Convert a BPE unicode token string back to raw bytes."""
    return bytes(decoder[ch] for ch in token_str)


def save_as_tekken(
    tokenizer: PreTrainedTokenizerFast,
    save_directory: str | Path,
) -> Path:
    """Reconstruct a `tekken.json` file from an HF tokenizer with stored tekken metadata.

    The tokenizer must have been originally loaded from a `tekken.json` file and
    must carry a `tekken_metadata` key in its `init_kwargs` (set automatically by
    `convert_tekken_tokenizer`).

    Args:
        tokenizer (`PreTrainedTokenizerFast`): HF fast tokenizer with `tekken_metadata` in init_kwargs.
        save_directory (`str | Path`): Directory to write the `tekken.json` file to.

    Returns:
        Path to the written `tekken.json` file.

    Raises:
        ValueError: If the tokenizer does not carry tekken metadata.
    """
    metadata = getattr(tokenizer, "tekken_metadata", None) or tokenizer.init_kwargs.get("tekken_metadata")
    if metadata is None:
        raise ValueError(
            "Tokenizer does not carry `tekken_metadata`. "
            "It was not loaded from a tekken.json file or metadata was lost."
        )

    save_directory = Path(save_directory)
    save_directory.mkdir(parents=True, exist_ok=True)

    decoder = _unicode_to_bytes()

    hf_vocab: dict[str, int] = tokenizer.get_vocab()

    # Separate special tokens (low ids) from BPE tokens.
    special_tokens_metadata = metadata.get("special_tokens", [])
    special_token_strs = {st["token_str"] for st in special_tokens_metadata}

    bpe_entries: list[tuple[int, str]] = []
    for token_str, token_id in hf_vocab.items():
        if token_str in special_token_strs:
            continue
        bpe_entries.append((token_id, token_str))

    bpe_entries.sort(key=lambda x: x[0])

    null_token_str_bytes = set(metadata.get("_null_token_str_bytes", []))

    vocab_list: list[dict] = []
    for rank, (_token_id, token_str) in enumerate(bpe_entries):
        try:
            raw_bytes = _bpe_token_to_bytes(token_str, decoder)
        except KeyError:
            raw_bytes = token_str.encode("utf-8")
        token_bytes_b64 = base64.b64encode(raw_bytes).decode("ascii")
        vocab_list.append(
            {
                "rank": rank,
                "token_bytes": token_bytes_b64,
                "token_str": None
                if token_bytes_b64 in null_token_str_bytes
                else raw_bytes.decode("utf-8", errors="replace"),
            }
        )

    tekken_data: dict = {}
    tekken_data["vocab"] = vocab_list
    tekken_data["special_tokens"] = special_tokens_metadata
    if "config" in metadata:
        tekken_data["config"] = metadata["config"]
    if "version" in metadata:
        tekken_data["version"] = metadata["version"]
    if "type" in metadata:
        tekken_data["type"] = metadata["type"]
    for optional_key in ("image", "audio", "multimodal"):
        if optional_key in metadata and metadata[optional_key] is not None:
            tekken_data[optional_key] = metadata[optional_key]

    output_path = save_directory / "tekken.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(tekken_data, f, ensure_ascii=False)

    return output_path


def convert_tekken_image_processor(
    tokenizer_file: str,
    params_file: str,
    chat_template: str | None = None,
) -> "PixtralProcessor":
    """Build a `PixtralProcessor` from a tekken tokenizer file and a native `params.json`.

    Args:
        tokenizer_file (`str`): Path to the `tekken.json` vocabulary file.
        params_file (`str`): Path to the native `params.json` config file.
        chat_template (`str | None`): Optional chat template string for the processor.

    Returns:
        Configured `PixtralProcessor` with tokenizer and image processor.

    Raises:
        ValueError: If `params_file` does not contain a `vision_encoder` key.
    """
    with open(params_file, encoding="utf-8") as f:
        params = json.load(f)

    vision_config = params.get("vision_encoder")
    if vision_config is None:
        raise ValueError(
            f"'vision_encoder' key not found in {params_file}. "
            "This model does not appear to be a vision-language model and does not need a processor. "
            "Use `convert_tekken_tokenizer` for text-only models instead."
        )

    # Lazy imports: processing_pixtral imports from integrations.mistral at module level,
    # so importing it here avoids a circular dependency. Placed after validation to avoid
    # triggering heavy imports (torchvision) for text-only models that will fail anyway.
    from ...models.pixtral.image_processing_pixtral import PixtralImageProcessor
    from ...models.pixtral.processing_pixtral import PixtralProcessor

    patch_size = vision_config["patch_size"]
    max_image_size = vision_config.get("max_image_size", vision_config["image_size"])
    spatial_merge_size = vision_config.get("spatial_merge_size", 2)

    if is_mistral_common_available():
        from ...tokenization_mistral_common import MistralCommonBackend

        tokenizer = MistralCommonBackend(tokenizer_path=tokenizer_file)
    else:
        tokenizer = convert_tekken_tokenizer(tokenizer_file)

    image_processor = PixtralImageProcessor(
        patch_size=patch_size,
        size={"longest_edge": max_image_size},
    )

    processor = PixtralProcessor(
        tokenizer=tokenizer,
        image_processor=image_processor,
        image_token="[IMG]",
        image_break_token="[IMG_BREAK]",
        image_end_token="[IMG_END]",
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        chat_template=chat_template,
    )

    return processor
