# Copyright 2024 The HuggingFace Team. All rights reserved.
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
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from parameterized import parameterized

from transformers.testing_utils import require_mistral_common, require_vision
from transformers.utils import is_vision_available

from ...test_processing_common import ProcessorTesterMixin, url_to_local_path


if is_vision_available():
    from transformers import PixtralProcessor


def _build_mistral_common_tokenizer():
    """Build a real MistralCommonBackend with all special tokens required by mistral-common."""
    import base64
    import json

    from mistral_common.tokens.tokenizers.base import SpecialTokens

    from transformers.tokenization_mistral_common import MistralCommonBackend

    # Collect all special tokens defined by mistral-common
    special_tokens = [{"rank": i, "token_str": st.value, "is_control": True} for i, st in enumerate(SpecialTokens)]
    num_special = len(special_tokens)
    vocab_size = 256 + num_special

    # Build BPE vocab from single raw bytes
    vocab_list = []
    for rank in range(256):
        raw_byte = bytes([rank])
        vocab_list.append({"rank": rank, "token_bytes": base64.b64encode(raw_byte).decode("ascii"), "token_str": None})

    tekken_data = {
        "vocab": vocab_list,
        "special_tokens": special_tokens,
        "config": {
            "pattern": r"""(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+""",
            "num_vocab_tokens": 256,
            "default_vocab_size": vocab_size,
            "default_num_special_tokens": num_special,
            "version": "v3",
        },
        "version": 1,
        "type": "tekken",
    }

    with tempfile.TemporaryDirectory() as tmpdir:
        tekken_path = Path(tmpdir) / "tekken.json"
        with open(tekken_path, "w", encoding="utf-8") as f:
            json.dump(tekken_data, f, ensure_ascii=False)

        # Tokenizer caches raw bytes at init, so cleanup is safe after construction
        tokenizer = MistralCommonBackend(tokenizer_path=tekken_path)
    return tokenizer


@require_vision
class PixtralProcessorTest(ProcessorTesterMixin, unittest.TestCase):
    processor_class = PixtralProcessor
    model_id = "mistral-community/pixtral-12b"

    @classmethod
    def _setup_test_attributes(cls, processor):
        cls.url_0 = url_to_local_path(
            "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/transformers/tasks/australia.jpg"
        )
        cls.image_0 = np.random.randint(255, size=(3, 876, 1300), dtype=np.uint8)
        cls.url_1 = "http://images.cocodataset.org/val2017/000000039769.jpg"
        cls.image_1 = np.random.randint(255, size=(3, 480, 640), dtype=np.uint8)
        cls.image_2 = np.random.randint(255, size=(3, 1024, 1024), dtype=np.uint8)
        cls.image_token = processor.image_token

    @parameterized.expand([(1, "pt"), (2, "pt")])
    @unittest.skip("Not tested before, to investigate")
    def test_apply_chat_template_image(self, batch_size, return_tensors):
        pass

    def test_image_token_filling(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        # Important to check with non square image
        image = torch.randint(0, 2, (3, 500, 316))
        expected_image_tokens = 640
        image_token_index = 10

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": "What is shown in this image?"},
                ],
            },
        ]
        inputs = processor(
            text=[processor.apply_chat_template(messages)],
            images=[image],
            return_tensors="pt",
        )
        image_tokens = (inputs["input_ids"] == image_token_index).sum().item()
        self.assertEqual(expected_image_tokens, image_tokens)

    def test_from_pretrained_subfolder_tokenizer(self):
        processor = PixtralProcessor.from_pretrained("hf-internal-testing/tiny-flux2", subfolder="tokenizer")
        self.assertIsInstance(processor, PixtralProcessor)
        self.assertIsNotNone(processor.tokenizer)

    def test_processor_with_single_image(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = "USER: [IMG]\nWhat's the content of the image? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=self.image_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=self.url_0, return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to "USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the content of the image? ASSISTANT:"
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test passing inputs as a single list
        inputs_image = processor(text=prompt_string, images=[self.image_0], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

        # Test as nested single list
        inputs_image = processor(text=prompt_string, images=[[self.image_0]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([1, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510,  1058,  1032,    10,    10,    12,    10,    10,    13,  1010, 7493,  1681,  1278,  4701,  1307,  1278,  3937,  1063,  1349,  4290, 16002, 41150,  1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_single_list(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:"

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=[self.image_0, self.image_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 1)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
                    )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=[self.url_0, self.url_1], return_tensors="pt")
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 1)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in as a nested list
        inputs_url = processor(text=prompt_string, images=[[self.image_0, self.image_1]], return_tensors="pt")
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([2, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_url["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    def test_processor_with_multiple_images_multiple_lists(self):
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG][IMG]\nWhat's the difference between these two images? ASSISTANT:",
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ]
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0, self.image_1], [self.image_2]]

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 2)
        self.assertIsInstance(inputs_image["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_image["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing in a url
        inputs_url = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_url)
        self.assertTrue(len(inputs_url["input_ids"]) == 2)
        self.assertIsInstance(inputs_url["input_ids"], torch.Tensor)
        self.assertIsInstance(inputs_image["pixel_values"], torch.Tensor)
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        input_ids = inputs_url["input_ids"]
        self.assertEqual(
            input_ids[0].tolist(),
            # Equivalent to ["USER: [IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END][IMG][IMG][IMG_BREAK][IMG][IMG][IMG_END]\nWhat's the difference between these two images? ASSISTANT:"]
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

        # Test passing as a single flat list
        inputs_image = processor(
            text=prompt_string, images=[self.image_0, self.image_1, self.image_2], return_tensors="pt", padding=True
        )
        self.assertTrue(inputs_image["pixel_values"].shape == torch.Size([3, 3, 32, 32]))

        # fmt: off
        self.assertEqual(
            inputs_image["input_ids"][0].tolist(),
            [21510, 1058, 1032, 10, 10, 12, 10, 10, 13, 10, 10, 12, 10, 10, 13, 1010, 7493, 1681, 1278, 6592, 2396, 2576, 2295, 8061, 1063, 1349, 4290, 16002, 41150, 1058]
        )
        # fmt: on

    @require_mistral_common
    def test_apply_chat_template_with_mistral_common_backend(self):
        """PixtralProcessor.apply_chat_template delegates to MistralCommonBackend and produces real tokens."""

        processor = self.processor_class.from_pretrained(self.tmpdirname)

        mc_tokenizer = _build_mistral_common_tokenizer()

        processor.tokenizer = mc_tokenizer

        conversation = [{"role": "user", "content": "Hello"}]

        result_str = processor.apply_chat_template(conversation, tokenize=False)
        self.assertIsInstance(result_str, str)
        self.assertIn("Hello", result_str)

        result_dict = processor.apply_chat_template(conversation, tokenize=True, return_dict=True)
        self.assertIn("input_ids", result_dict)
        self.assertTrue(len(result_dict["input_ids"]) > 0)

        result_pt = processor.apply_chat_template(conversation, tokenize=True, return_dict=True, return_tensors="pt")
        self.assertIn("input_ids", result_pt)
        self.assertIsInstance(result_pt["input_ids"], torch.Tensor)
        self.assertTrue(result_pt["input_ids"].numel() > 0)

        result_ids = processor.apply_chat_template(conversation, tokenize=True, return_dict=False)
        self.assertIsInstance(result_ids, list)
        self.assertTrue(len(result_ids) > 0)

    def test_processor_returns_full_length_batches(self):
        # to avoid https://github.com/huggingface/transformers/issues/34204
        processor = self.processor_class.from_pretrained(self.tmpdirname)
        prompt_string = [
            "USER: [IMG]\nWhat's the content of the image? ASSISTANT:",
        ] * 5
        processor.tokenizer.pad_token = "</s>"
        image_inputs = [[self.image_0]] * 5

        # Make small for checking image token expansion
        processor.image_processor.size = {"longest_edge": 30}
        processor.image_processor.patch_size = {"height": 2, "width": 2}

        # Test passing in an image
        inputs_image = processor(text=prompt_string, images=image_inputs, return_tensors="pt", padding=True)
        self.assertIn("input_ids", inputs_image)
        self.assertTrue(len(inputs_image["input_ids"]) == 5)
        self.assertTrue(len(inputs_image["pixel_values"]) == 5)
