"""Unit tests for the Classify + Detect cache (Plan v3, Priority 5).

Layers 1-2 are purpose-agnostic by design, so the same document re-run under a
different purpose must not pay for them twice — only the free policy lookup may
differ. The cache key is the normalized image, so a hit is a guarantee rather
than a guess. Offline: the OCR and LLM calls are stubbed.
"""
import unittest
from unittest.mock import patch

from PIL import Image

from ocr import NormalizedImage, OcrWord
from redact import (
    DETECT_CACHE,
    clear_detect_cache,
    load_presets,
    run_plan_v2,
)


def _word(wid, text, x=10, y=10, w=60, h=30):
    return OcrWord(wid, text, x, y, w, h, 95)


WORDS = [
    _word("w0000", "ARJUN"),
    _word("w0001", "MEHTA", 90),
    _word("w0002", "Aadhaar", 10, 50),
    _word("w0003", "123456789012", 120, 50, w=150),
]
LABELS = [
    {"id": "w0000", "type": "name"},
    {"id": "w0001", "type": "name"},
    {"id": "w0002", "type": "label_text"},
    {"id": "w0003", "type": "aadhaar_number"},
]


class TestDetectCache(unittest.TestCase):
    def setUp(self):
        clear_detect_cache()
        self.classify_calls = 0
        self.detect_calls = 0

    def tearDown(self):
        clear_detect_cache()

    def _run(self, image, purpose="id_verification"):
        norm = NormalizedImage(image=image, scale=1.0, upscaled=False)

        def _classify(words):
            self.classify_calls += 1
            return "aadhaar_card"

        def _detect(words, document_type="other"):
            self.detect_calls += 1
            return [dict(entry) for entry in LABELS]

        with patch("redact.normalize_image", return_value=norm), patch(
            "redact.extract_words", return_value=WORDS
        ), patch("llm.classify", side_effect=_classify), patch(
            "llm.detect_batched", side_effect=_detect
        ), patch("llm.validate_llm", return_value=([], "clean")):
            return run_plan_v2(image, purpose, load_presets()[purpose])

    def test_second_purpose_reuses_the_cached_classify_and_detect(self):
        image = Image.new("RGB", (300, 120), (255, 255, 255))
        first = self._run(image, "id_verification")
        second = self._run(image, "proof_of_address")
        self.assertEqual((self.classify_calls, self.detect_calls), (1, 1))
        self.assertEqual(len(DETECT_CACHE), 1)
        self.assertFalse(
            any("cached" in w for w in first.warnings), first.warnings
        )
        self.assertTrue(
            any("cached" in w for w in second.warnings), second.warnings
        )
        # Same fragments, so the identifier is handled identically either way...
        self.assertEqual(first.document_type, second.document_type)
        self.assertEqual(second.partial_ids, [])
        self.assertIn("w0003", second.llm_redact_ids)  # redacted, not partial

    def test_a_different_image_is_not_a_cache_hit(self):
        self._run(Image.new("RGB", (300, 120), (255, 255, 255)))
        self._run(Image.new("RGB", (300, 120), (0, 0, 0)))
        self.assertEqual((self.classify_calls, self.detect_calls), (2, 2))
        self.assertEqual(len(DETECT_CACHE), 2)

    def test_cache_is_bounded(self):
        for shade in range(12):
            self._run(Image.new("RGB", (300, 120), (shade, shade, shade)))
        self.assertLessEqual(len(DETECT_CACHE), 8)

    def test_cached_labels_are_not_aliased_across_runs(self):
        image = Image.new("RGB", (300, 120), (255, 255, 255))
        first = self._run(image)
        cached = DETECT_CACHE[_key(image)][1]
        self.assertEqual(cached, LABELS)  # raw labels, anchors applied per run
        self.assertNotIn("w0003", first.visible_ids)


def _key(image):
    return next(iter(DETECT_CACHE))


if __name__ == "__main__":
    unittest.main()
