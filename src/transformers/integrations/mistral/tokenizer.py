import base64
import json
from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING, Any

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tqdm import tqdm

from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast


if TYPE_CHECKING:
    from transformers.models.pixtral.processing_pixtral import PixtralProcessor

MAP_SPECIALS = {
    "bos_token": "<s>",
    "eos_token": "</s>",
    "pad_token": "<pad>",
    "unk_token": "<unk>",
}


class MistralConverter:
    r"""Converter from Mistral tekken BPE vocab to a HuggingFace `tokenizers.Tokenizer`.

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
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens
        self._precomputed_vocab: dict[str, int] | None = None
        self._precomputed_merges: list[tuple[str, str]] | None = None
        self._tekken_metadata: dict[str, Any] | None = None

    @property
    def tekken_metadata(self) -> dict[str, Any]:
        assert self._tekken_metadata is not None, (
            "Only accessible when instance is created by `from_tekken_file` method."
        )
        return self._tekken_metadata

    @classmethod
    def from_tekken_file(
        cls,
        vocab_file: str,
        add_prefix_space: bool = False,
    ) -> "MistralConverter":
        r"""Parse a raw `tekken.json` file and return a ready-to-use converter.

        Reads the file, extracts the regex pattern and special tokens, then
        pre-computes `vocab` and `merges` with correct index offsets (special
        tokens occupy the first indices).

        Args:
            vocab_file: Path to a `tekken.json` file.
            add_prefix_space: Whether to add a prefix space during tokenization.
        """
        with open(vocab_file, encoding="utf-8") as f:
            untyped = json.load(f)

        pattern = untyped["config"]["pattern"]

        additional_special_tokens = [AddedToken(k["token_str"], special=True) for k in untyped["special_tokens"]]
        bpe_ranks_raw = untyped["vocab"]
        byte_encoder = bytes_to_unicode()

        @lru_cache
        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        local_tuples: list[tuple[str, str, str]] = []
        merges: list[tuple[str, str]] = []
        vocab: dict[str, int] = {}
        for idx, token in enumerate(additional_special_tokens):
            vocab[token.content] = idx
        num_special = len(additional_special_tokens)

        bpe_ranks = [base64.b64decode(k["token_bytes"]) for k in bpe_ranks_raw]
        rank_set = set(bpe_ranks)
        token_to_rank = {token: rank for rank, token in enumerate(bpe_ranks)}
        for rank, token in enumerate(tqdm(bpe_ranks, desc="Converting tekken.json to tokenizer.json")):
            vocab[token_bytes_to_string(token)] = num_special + rank
            if len(token) == 1:
                continue
            local = []
            for index in range(1, len(token)):
                piece_l, piece_r = token[:index], token[index:]
                if piece_l in rank_set and piece_r in rank_set and (piece_l + piece_r) in rank_set:
                    local.append((piece_l, piece_r, rank))
            local = sorted(local, key=lambda x: (token_to_rank[x[0]], token_to_rank[x[1]]))
            local_tuples.extend(local)
        local_tuples = sorted(local_tuples, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in local_tuples]

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

        return instance

    def extract_vocab_merges_from_model(self, vocab: dict) -> tuple[dict[str, int], list[tuple[str, str]]]:
        r"""Extract vocab and merges from a pre-parsed vocab dict."""
        bpe_ranks = vocab
        byte_encoder = bytes_to_unicode()

        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

        merges: list[tuple[str, str]] = []
        result_vocab: dict[str, int] = {}
        for idx, (token, rank) in enumerate(bpe_ranks.items()):
            if token not in self.additional_special_tokens:
                result_vocab[token_bytes_to_string(token)] = idx
                if len(token) == 1:
                    continue
                local = []
                for index in range(1, len(token)):
                    piece_l, piece_r = token[:index], token[index:]
                    if piece_l in bpe_ranks and piece_r in bpe_ranks and (piece_l + piece_r) in bpe_ranks:
                        local.append((piece_l, piece_r, rank))
                local = sorted(local, key=lambda x: (bpe_ranks[x[0]], bpe_ranks[x[1]]))
                merges.extend(local)
            else:
                result_vocab[token] = idx
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]
        return result_vocab, merges

    def tokenizer(self) -> Tokenizer:
        r"""Build a raw `tokenizers.Tokenizer` with BPE model (no pre/post-processing)."""
        if self._precomputed_vocab:
            vocab_scores, merges = self._precomputed_vocab, self._precomputed_merges
        else:
            vocab_scores, merges = self.extract_vocab_merges_from_model(self.vocab)
        tokenizer = Tokenizer(BPE(vocab_scores, merges, fuse_unk=False))
        if hasattr(tokenizer.model, "ignore_merges"):
            tokenizer.model.ignore_merges = True
        return tokenizer

    def converted(self) -> Tokenizer:
        r"""Build a fully configured `tokenizers.Tokenizer` with pre-tokenizer and decoder."""
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
    converter = MistralConverter.from_tekken_file(vocab_file=tokenizer_file, add_prefix_space=False)
    fast = PreTrainedTokenizerFast(
        tokenizer_object=converter.converted(),
        tekken_metadata=converter.tekken_metadata,
        **MAP_SPECIALS,
    )
    return fast


def _unicode_to_bytes() -> dict[str, int]:
    r"""Invert `bytes_to_unicode()` to map unicode chars back to byte values."""
    return {v: k for k, v in bytes_to_unicode().items()}


def _bpe_token_to_bytes(token_str: str, decoder: dict[str, int]) -> bytes:
    r"""Convert a BPE unicode token string back to raw bytes."""
    return bytes(decoder[ch] for ch in token_str)


def save_as_tekken(
    tokenizer: PreTrainedTokenizerFast,
    save_directory: str | Path,
) -> Path:
    r"""Reconstruct a `tekken.json` file from an HF tokenizer with stored tekken metadata.

    The tokenizer must have been originally loaded from a `tekken.json` file and
    must carry a `tekken_metadata` key in its `init_kwargs` (set automatically by
    `convert_tekken_tokenizer`).

    Args:
        tokenizer: HF fast tokenizer with `tekken_metadata` in init_kwargs.
        save_directory: Directory to write the `tekken.json` file to.

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

    vocab_list: list[dict] = []
    for rank, (_token_id, token_str) in enumerate(bpe_entries):
        try:
            raw_bytes = _bpe_token_to_bytes(token_str, decoder)
        except KeyError:
            raw_bytes = token_str.encode("utf-8")
        vocab_list.append(
            {
                "rank": rank,
                "token_bytes": base64.b64encode(raw_bytes).decode("ascii"),
                "token_str": raw_bytes.decode("utf-8", errors="replace"),
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


def convert_tekken_processor(
    tokenizer_file: str,
    params_file: str,
    chat_template: str | None = None,
) -> "PixtralProcessor":
    from transformers.models.pixtral.image_processing_pixtral import PixtralImageProcessor
    from transformers.models.pixtral.processing_pixtral import PixtralProcessor

    with open(params_file, encoding="utf-8") as f:
        params = json.load(f)

    vision_config = params.get("vision_encoder")
    if vision_config is None:
        raise ValueError(
            f"'vision_encoder' key not found in {params_file}. "
            "This model does not appear to be a vision-language model and does not need a processor. "
            "Use `convert_tekken_tokenizer` for text-only models instead."
        )

    patch_size = vision_config["patch_size"]
    max_image_size = vision_config.get("max_image_size", vision_config["image_size"])
    spatial_merge_size = vision_config.get("spatial_merge_size", 2)

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
