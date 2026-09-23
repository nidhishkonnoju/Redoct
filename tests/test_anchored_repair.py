"""Unit tests for the deterministic label-anchored keep repair."""
import unittest

from llm import LLMDecision
from ocr import OcrWord
from redact import repair_anchored_keeps


def _line(words, y):
    """Build words left-to-right on one visual line at height y."""
    out, x = [], 0
    for text in words:
        out.append(OcrWord(f"w{x:04d}{y}", text, x, y, 10 * len(text), 20, 95))
        x += 10 * len(text) + 10
    return out


def _decision(words):
    return LLMDecision(
        document_type="bank_statement",
        redact_ids=[w.id for w in words],  # LLM redacted everything
        keep_ids=[],
    )


class TestAnchoredRepair(unittest.TestCase):
    def test_salary_amount_on_anchored_line_is_kept(self):
        words = _line(["SALARY", "CREDIT", "-", "INFOTECH", "LTD", "85,000.00"], 0)
        corrected, repaired = repair_anchored_keeps(
            words, _decision(words), "proof_of_income"
        )
        kept = {w.text for w in words if w.id in set(corrected.keep_ids)}
        self.assertIn("85,000.00", kept)
        self.assertTrue(repaired)

    def test_account_holder_name_is_kept(self):
        words = _line(["Account", "Holder:", "ARJUN", "MEHTA"], 0)
        corrected, _ = repair_anchored_keeps(
            words, _decision(words), "proof_of_income"
        )
        kept = {w.text for w in words if w.id in set(corrected.keep_ids)}
        self.assertIn("ARJUN", kept)
        self.assertIn("MEHTA", kept)

    def test_pii_never_restored(self):
        words = _line(["Account", "Holder:", "ARJUN", "5010023456789012"], 0)
        corrected, _ = repair_anchored_keeps(
            words, _decision(words), "proof_of_income"
        )
        self.assertNotIn("5010023456789012", {w.id for w in words if w.id in set(corrected.keep_ids)})
        account = next(w for w in words if w.text.startswith("5010"))
        self.assertIn(account.id, corrected.redact_ids)

    def test_third_party_name_line_skipped(self):
        words = _line(["Father's", "Name:", "RAKESH", "MEHTA"], 0)
        corrected, repaired = repair_anchored_keeps(
            words, _decision(words), "proof_of_income"
        )
        self.assertEqual(repaired, [])
        self.assertEqual(corrected.keep_ids, [])

    def test_dates_not_restored_on_salary_line(self):
        words = _line(["SALARY", "CREDIT", "02-Apr-2026", "85,000.00"], 0)
        corrected, _ = repair_anchored_keeps(
            words, _decision(words), "proof_of_income"
        )
        kept = {w.text for w in words if w.id in set(corrected.keep_ids)}
        self.assertIn("85,000.00", kept)
        self.assertNotIn("02-Apr-2026", kept)

    def test_other_purposes_are_untouched(self):
        words = _line(["SALARY", "CREDIT", "85,000.00"], 0)
        decision = _decision(words)
        corrected, repaired = repair_anchored_keeps(
            words, decision, "id_verification"
        )
        self.assertEqual(repaired, [])
        self.assertEqual(corrected.keep_ids, [])
        self.assertEqual(
            set(corrected.redact_ids), {w.id for w in words}
        )


if __name__ == "__main__":
    unittest.main()
