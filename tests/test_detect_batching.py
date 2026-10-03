"""Offline tests for Detect batching and its fault tolerance (Plan v2, Layer 2).

The real model is never called: `llm.detect` is replaced, so these tests pin the
sharding contract (deterministic, id-exact, bounded) and the degradation rule
(a batch that will not parse is split; only the unparseable leaf fails closed).
"""
import unittest
from unittest.mock import patch

from llm import OllamaError, OllamaParseError, detect_batched, LAST_DETECT_FAILED_IDS
from ocr import OcrWord


def _words(n, poison_at=None):
    words = []
    for i in range(n):
        text = "POISON" if i == poison_at else f"tok{i:02d}"
        words.append(OcrWord(f"w{i:04d}", text, 10 + 20 * i, 10 * (i // 4), 40, 18, 95))
    return words


def _labeller(batch_size_seen):
    def fake_detect(batch, document_type=None):
        batch_size_seen.append(len(batch))
        return [{"id": w.id, "type": "other"} for w in batch]
    return fake_detect


class TestDetectBatching(unittest.TestCase):
    def test_batches_are_bounded_and_every_id_labeled_once(self):
        words = _words(60)
        seen: list[int] = []
        with patch("llm.detect", side_effect=_labeller(seen)):
            labels = detect_batched(words, batch_size=25)
        self.assertTrue(all(n <= 25 for n in seen))
        self.assertEqual(sorted(e["id"] for e in labels), sorted(w.id for w in words))
        self.assertFalse(LAST_DETECT_FAILED_IDS)

    def test_failing_batch_is_split_so_the_rest_survives(self):
        words = _words(12, poison_at=3)
        seen: list[int] = []

        def flaky(batch, document_type=None):
            seen.append(len(batch))
            if any(w.text == "POISON" for w in batch):
                raise OllamaParseError("truncated JSON")
            return [{"id": w.id, "type": "other"} for w in batch]

        with patch("llm.detect", side_effect=flaky):
            labels = detect_batched(words, batch_size=12)
        # The whole document is not lost: only the small poisoned leaf is.
        self.assertTrue(set(LAST_DETECT_FAILED_IDS).issubset({w.id for w in words}))
        self.assertLessEqual(len(LAST_DETECT_FAILED_IDS), 5)
        self.assertNotIn("w0003", [e["id"] for e in labels])
        self.assertEqual(len(labels), len(words) - len(LAST_DETECT_FAILED_IDS))
        # Latency guard: the failing 12-fragment batch is halved, never retried
        # wholesale at full size more than once.
        self.assertEqual(max(seen), 12)
        self.assertIn(6, seen)

    def test_failed_ids_reset_between_runs(self):
        LAST_DETECT_FAILED_IDS.extend(["stale"])
        with patch("llm.detect", side_effect=_labeller([])):
            detect_batched(_words(4), batch_size=25)
        self.assertEqual(LAST_DETECT_FAILED_IDS, [])

    def test_connection_error_propagates_so_the_pipeline_fails_closed(self):
        with patch("llm.detect", side_effect=OllamaError("server down")):
            with self.assertRaises(OllamaError):
                detect_batched(_words(30), batch_size=25)


if __name__ == "__main__":
    unittest.main()
