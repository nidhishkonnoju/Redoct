"""Unit tests for the date rule chain (Plan v3 Priority 2B).

Two rules have to agree for a date to be handled correctly:

- `transaction_date` is a KEEP type in every preset (a statement's dates are not
  sensitive), and
- `anchor_date_labels()` demotes a `date_of_birth` label to `transaction_date`
  only when the row is a ledger row with no birth-date anchor, while the
  `date_of_birth` context rule force-redacts every date near a birth-date label.

The tests below pin both directions: a ledger date stays visible, and a real
date of birth cannot escape just because the model mislabelled it. All offline.
"""
import unittest

from ocr import OcrWord
from policy import apply_policy
from redact import (
    anchor_date_labels,
    apply_validator,
    load_presets,
    validate_context_fragments,
)
from llm import LLMDecision


def _line(y, words, start=0, h=20):
    """Build one OCR line: words laid out left-to-right at baseline y."""
    out, x = [], 10
    for i, text in enumerate(words):
        out.append(OcrWord(f"w{start + i:04d}", text, x, y, 10 * len(text), h, 95))
        x += 10 * len(text) + 8
    return out


class TestDateAnchoring(unittest.TestCase):
    def setUp(self):
        self.words = (
            _line(10, ["02-Apr-2026", "SALARY", "CREDIT", "85,000.00"])
            + _line(40, ["05-Apr-2026", "UPI/REF/982134/Rent", "24,500.00"], start=4)
            + _line(70, ["Date", "of", "Birth", "14/08/1999"], start=7)
            + _line(100, ["Statement", "Period:", "01-Apr-2026", "30-Jun-2026"], start=11)
        )

    def _labels(self, spec):
        return [{"id": i, "type": t} for i, t in spec]

    def test_ledger_date_labelled_as_dob_is_demoted(self):
        labels = self._labels([("w0000", "date_of_birth")])  # 02-Apr-2026 txn date
        out, demoted = anchor_date_labels(self.words, labels)
        self.assertEqual(demoted, ["w0000"])
        self.assertEqual(out[0]["type"], "transaction_date")

    def test_real_dob_under_its_label_is_not_demoted(self):
        labels = self._labels([("w0010", "date_of_birth")])  # 14/08/1999
        out, demoted = anchor_date_labels(self.words, labels)
        self.assertEqual(demoted, [])
        self.assertEqual(out[0]["type"], "date_of_birth")

    def test_statement_period_dates_are_not_demoted_or_altered(self):
        # They are already labelled correctly; the rule must leave them alone.
        labels = self._labels([("w0013", "transaction_date")])
        self.assertEqual(anchor_date_labels(self.words, labels), (labels, []))

    def test_non_date_dob_labels_are_left_alone(self):
        # A model can mislabel anything: without a date or a ledger hint in the
        # row there is nothing to demote, so the DOB label (a redact type stands).
        words = _line(10, ["Customer", "ID:", "99887766"])
        labels = self._labels([("w0002", "date_of_birth")])
        self.assertEqual(anchor_date_labels(words, labels), (labels, []))

    def test_no_labels_is_a_noop(self):
        self.assertEqual(anchor_date_labels(self.words, []), ([], []))

    def test_ledger_date_stays_visible_end_to_end(self):
        labels = self._labels([("w0000", "date_of_birth")])
        out, _ = anchor_date_labels(self.words, labels)
        split = apply_policy(out, load_presets()["proof_of_income"])
        self.assertIn("w0000", split["keep_ids"])

    def test_mislabelled_birth_date_is_caught_by_the_context_rule(self):
        # Worst case: the model calls the birth date a transaction_date, which
        # every preset keeps. The label-anchored regex net must still redact it.
        labels = self._labels([("w0010", "transaction_date")])
        split = apply_policy(labels, load_presets()["proof_of_income"])
        self.assertIn("w0010", split["keep_ids"])
        decision = LLMDecision(
            document_type="bank_statement",
            redact_ids=split["redact_ids"],
            keep_ids=split["keep_ids"],
        )
        corrected, hits, _ = apply_validator(self.words, decision)
        self.assertIn("w0010", corrected.redact_ids)
        self.assertNotIn("w0010", corrected.keep_ids)
        self.assertIn("date_of_birth", {h.pattern for h in hits})

    def test_presets_keep_transaction_dates(self):
        for name, preset in load_presets().items():
            self.assertIn("transaction_date", preset["keep_types"], name)
            self.assertNotIn("transaction_date", preset["redact_types"], name)


class TestContextWindowWidening(unittest.TestCase):
    def test_value_printed_above_its_label_is_caught(self):
        # Right-aligned value, label underneath: the model may call the date a
        # transaction_date, so the net has to look one line up as well.
        words = _line(10, ["14/08/1999"]) + _line(40, ["Date", "of", "Birth"], start=1)
        hits = validate_context_fragments(words, set())
        self.assertEqual([h.matched_text for h in hits], ["14/08/1999"])

    def test_dates_far_from_a_label_are_still_untouched(self):
        # A birth date at the bottom of the page must not sweep the ledger above
        # it: only geometrically adjacent lines share the label's window.
        words = (
            _line(10, ["02-Apr-2026", "SALARY", "CREDIT"])
            + _line(40, ["05-Apr-2026", "24,500.00"], start=3)
            + _line(400, ["Date", "of", "Birth", "14/08/1999"], start=5)
        )
        hits = validate_context_fragments(words, set())
        self.assertEqual([h.word_id for h in hits], ["w0008"])


if __name__ == "__main__":
    unittest.main()
