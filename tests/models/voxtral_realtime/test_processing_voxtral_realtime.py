# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

import unittest
import warnings

import numpy as np

from transformers import AutoProcessor
from transformers.testing_utils import require_mistral_common, require_torch
from transformers.utils import is_mistral_common_available, is_soundfile_available, is_torch_available


if is_torch_available():
    import torch

if is_mistral_common_available():
    from mistral_common.protocol.transcription.request import StreamingMode, TranscriptionRequest
    from mistral_common.tokens.tokenizers.audio import Audio


@require_mistral_common
@require_torch
@unittest.skipUnless(is_soundfile_available(), "test requires soundfile")
class VoxtralRealtimeProcessorTest(unittest.TestCase):
    checkpoint_name = "mistralai/Voxtral-Mini-4B-Realtime-2602"

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.processor = AutoProcessor.from_pretrained(cls.checkpoint_name)

    def test_online_streaming_matches_legacy_audio_request(self):
        """In online streaming mode the processor no longer passes audio to `mistral_common`'s
        `TranscriptionRequest` (that path is deprecated and removed in mistral-common 1.13.0). The prefill
        audio array is rebuilt locally instead. This locks in that the new behavior produces identical
        tokens and audio features to the legacy audio-in-request path.
        """
        processor = self.processor
        sampling_rate = processor.feature_extractor.sampling_rate

        rng = np.random.default_rng(42)
        audio = (rng.standard_normal(sampling_rate) * 0.1).clip(-1.0, 1.0).astype(np.float32)

        # New behavior: no audio is passed to the request, so no deprecation/future warning is emitted.
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            new_encoding = processor(audio, is_streaming=True, is_first_audio_chunk=True, return_tensors="pt")
        streaming_warnings = [
            w
            for w in caught
            if issubclass(w.category, (DeprecationWarning, FutureWarning)) and "streaming" in str(w.message).lower()
        ]
        self.assertEqual(
            streaming_warnings,
            [],
            f"Unexpected streaming warning(s): {[str(w.message) for w in streaming_warnings]}",
        )

        # Legacy behavior: audio passed directly to the request (the deprecated path we migrated away from).
        audio_obj = Audio(audio_array=audio, sampling_rate=sampling_rate, format="wav")
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                legacy_request = TranscriptionRequest(
                    audio=audio_obj.to_base64("wav"),
                    streaming=StreamingMode.ONLINE,
                    language=None,
                )
                legacy_tokenized = processor.tokenizer.tokenizer.encode_transcription(legacy_request)
        except AssertionError as error:
            self.skipTest(f"Legacy audio-in-request path is no longer supported by mistral-common: {error}")

        legacy_audio_arrays = [el.audio_array for el in legacy_tokenized.audios]
        legacy_features = processor.feature_extractor(
            legacy_audio_arrays,
            center=True,
            sampling_rate=sampling_rate,
            padding=True,
            truncation=False,
            return_tensors="pt",
        )

        self.assertEqual(new_encoding["input_ids"].tolist(), [legacy_tokenized.tokens])
        torch.testing.assert_close(new_encoding["input_features"], legacy_features["input_features"])
