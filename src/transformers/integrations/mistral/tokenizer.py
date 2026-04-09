import base64
import json
from functools import lru_cache
from typing import TYPE_CHECKING

from tokenizers import AddedToken, Regex, Tokenizer, decoders, pre_tokenizers, processors
from tokenizers.models import BPE
from tqdm import tqdm

from transformers.convert_slow_tokenizer import bytes_to_unicode
from transformers.tokenization_utils_tokenizers import PreTrainedTokenizerFast
from transformers.utils.import_utils import is_mistral_common_available, requires


if is_mistral_common_available():
    from mistral_common.tokens.tokenizers.base import SpecialTokens
    from mistral_common.tokens.tokenizers.mistral import MistralTokenizer

if TYPE_CHECKING:
    from transformers.models.pixtral.processing_pixtral import PixtralProcessor


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
        additional_special_tokens: list[str] | None = None,
        **kwargs,
    ):
        self.vocab = vocab
        self.pattern = pattern
        self.add_prefix_space = add_prefix_space
        self.additional_special_tokens = additional_special_tokens

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
        additional_special_tokens = [
            AddedToken(k["token_str"], special=k["is_control"]) for k in untyped["special_tokens"]
        ]
        bpe_ranks_raw = untyped["vocab"]
        byte_encoder = bytes_to_unicode()

        @lru_cache
        def token_bytes_to_string(b: bytes) -> str:
            return "".join([byte_encoder[ord(char)] for char in b.decode("latin-1")])

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
            merges.extend(local)
        merges = sorted(merges, key=lambda val: val[2])
        merges = [(token_bytes_to_string(val[0]), token_bytes_to_string(val[1])) for val in merges]

        instance = cls(
            vocab=None,
            pattern=pattern,
            add_prefix_space=add_prefix_space,
            additional_special_tokens=additional_special_tokens,
        )
        # Store pre-computed vocab and merges so tokenizer() can use them directly
        instance._precomputed_vocab = vocab
        instance._precomputed_merges = merges
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
        if hasattr(self, "_precomputed_vocab"):
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


@requires(backends=("mistral-common",))
def convert_tekken_tokenizer(tokenizer_file: str):
    """Convert a "tekken" tokenizer to a fast Tokenizer."""
    # Load directly using their lib
    mistral_tokenizer = MistralTokenizer.from_file(tokenizer_file)

    # Extract vocab and special tokens
    vocab = mistral_tokenizer.instruct_tokenizer.tokenizer._tekken_token2id_nospecial
    sorted_tokens = sorted(mistral_tokenizer.instruct_tokenizer.tokenizer._all_special_tokens, key=lambda x: x["rank"])
    all_special = [token["token_str"] for token in sorted_tokens]

    specials_tokens = {token: idx for idx, token in enumerate(all_special)}

    specials_tokens.update(vocab)
    vocab = specials_tokens

    # TODO(juliendenize): expose this in mistral-common to avoid accessing private attributes
    # and improve maintainability
    pattern = mistral_tokenizer.instruct_tokenizer.tokenizer._model._pat_str

    # Convert
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=MistralConverter(
            vocab=vocab, additional_special_tokens=all_special, pattern=pattern
        ).converted()
    )

    # Post-process
    tokenizer.add_special_tokens({"additional_special_tokens": all_special})

    MAP_SPECAL = {
        "bos_token": SpecialTokens.bos.value,
        "eos_token": SpecialTokens.eos.value,
        "pad_token": SpecialTokens.pad.value,
        "unk_token": SpecialTokens.unk.value,
    }

    for special_key, special_token in MAP_SPECAL.items():
        if special_token in all_special:
            tokenizer.add_special_tokens({special_key: special_token})

    return tokenizer


@requires(backends=("mistral-common",))
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
        image_token=SpecialTokens.img.value,
        image_break_token=SpecialTokens.img_break.value,
        image_end_token=SpecialTokens.img_end.value,
        patch_size=patch_size,
        spatial_merge_size=spatial_merge_size,
        chat_template=chat_template,
    )

    return processor
