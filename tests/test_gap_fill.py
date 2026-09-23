"""Unit tests for fail-closed id accounting + validator override."""
import unittest

from llm import LLMDecision
from ocr import OcrWord
from redact import apply_validator


def _words():
    return [
        OcrWord("w0001", "ARJUN", 0, 0, 50, 20, 95),
        OcrWord("w0002", "MEHTA", 60, 0, 50, 20, 95),
        OcrWord("w0003", "5010023456789012", 0, 30, 120, 20, 92),
        OcrWord("w0004", "9876543210", 0, 60, 90, 20, 96),
        OcrWord("w0005", "SALARY", 0, 90, 60, 20, 95),
    ]


class TestFailClosed(unittest.TestCase):
    def test_unmentioned_ids_fail_closed(self):
        decision = LLMDecision(
            document_type="bank_statement",
            redact_ids=["w0003"],
            keep_ids=["w0001"],
        )
        corrected, hits, unfilled = apply_validator(_words(), decision)
        self.assertEqual(set(unfilled), {"w0002", "w0004", "w0005"})
        for wid in ("w0002", "w0004", "w0005"):
            self.assertIn(wid, corrected.redact_ids)

    def test_unknown_ids_dropped(self):
        decision = LLMDecision(
            document_type="other",
            redact_ids=["w9999", "w0001"],
            keep_ids=["w0003"],
        )
        corrected, hits, unfilled = apply_validator(_words(), decision)
        self.assertNotIn("w9999", corrected.redact_ids)
        self.assertNotIn("w9999", corrected.keep_ids)
        self.assertIn("w0001", corrected.redact_ids)

    def test_validator_overrides_keep(self):
        decision = LLMDecision(
            document_type="bank_statement",
            redact_ids=["w0003"],
            keep_ids=["w0001", "w0004"],  # LLM wrongly kept the phone number
        )
        corrected, hits, unfilled = apply_validator(_words(), decision)
        hit_ids = {h.word_id for h in hits}
        self.assertIn("w0004", hit_ids)
        self.assertIn("w0004", corrected.redact_ids)
        self.assertNotIn("w0004", corrected.keep_ids)

    def test_full_coverage_partition(self):
        decision = LLMDecision(
            document_type="other",
            redact_ids=["w0003"],
            keep_ids=["w0001", "w0002"],
        )
        corrected, hits, unfilled = apply_validator(_words(), decision)
        assigned = set(corrected.redact_ids) | set(corrected.keep_ids)
        self.assertEqual(assigned, {w.id for w in _words()})
        self.assertFalse(set(corrected.redact_ids) & set(corrected.keep_ids))


if __name__ == "__main__":
    unittest.main()
