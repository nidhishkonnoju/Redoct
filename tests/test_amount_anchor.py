"""Unit tests for the amount-anchoring rule (Plan v2, Layer 2 -> Layer 3).

`salary_amount` is a KEEP type under Proof of Income, so the deterministic rule
must demote every money value that is not on a salary-anchored line. All offline.
"""
import unittest

from ocr import OcrWord
from redact import anchor_amount_labels
from policy import apply_policy
from redact import load_presets


def _line(y, words, start=0):
    """Build one OCR line: words laid out left-to-right at baseline y."""
    out, x = [], 10
    for i, text in enumerate(words):
        out.append(OcrWord(f"w{start + i:04d}", text, x, y, 10 * len(text), 20, 95))
        x += 10 * len(text) + 8
    return out


class TestAmountAnchoring(unittest.TestCase):
    def setUp(self):
        self.words = (
            _line(10, ["02-Apr-2026", "SALARY", "CREDIT", "85,000.00"])
            + _line(40, ["05-Apr-2026", "UPI/REF/982134/Rent", "Payment", "24,500.00"], start=4)
            + _line(70, ["Closing", "Balance:", "INR", "3,42,118.00"], start=8)
            + _line(100, ["Gross", "Salary", "1,23,456.00"], start=12)
        )

    def _labels(self, spec):
        return [{"id": i, "type": t} for i, t in spec]

    def test_unanchored_amount_demoted_to_other_amount(self):
        labels = self._labels([("w0007", "salary_amount")])  # rent line
        out, demoted = anchor_amount_labels(self.words, labels)
        self.assertEqual(demoted, ["w0007"])
        self.assertEqual(out[0]["type"], "other_amount")

    def test_anchored_amounts_kept_as_salary(self):
        labels = self._labels([
            ("w0003", "salary_amount"),   # SALARY CREDIT line
            ("w0014", "salary_amount"),   # Gross Salary line
        ])
        out, demoted = anchor_amount_labels(self.words, labels)
        self.assertEqual(demoted, [])
        self.assertEqual([e["type"] for e in out], ["salary_amount", "salary_amount"])

    def test_closing_balance_is_not_a_salary(self):
        labels = self._labels([("w0011", "salary_amount")])
        out, demoted = anchor_amount_labels(self.words, labels)
        self.assertEqual(demoted, ["w0011"])
        self.assertEqual(out[0]["type"], "other_amount")

    def test_non_amount_labels_untouched(self):
        labels = self._labels([
            ("w0000", "label_text"), ("w0005", "label_text"), ("w0009", "date_of_birth"),
        ])
        out, demoted = anchor_amount_labels(self.words, labels)
        self.assertEqual(demoted, [])
        self.assertEqual(out, labels)

    def test_no_labels_is_a_noop(self):
        self.assertEqual(anchor_amount_labels(self.words, []), ([], []))

    def test_policy_end_to_end_only_salary_stays_visible(self):
        labels = self._labels([
            ("w0007", "salary_amount"),   # rent -> demoted
            ("w0011", "salary_amount"),   # closing balance -> demoted
            ("w0014", "salary_amount"),   # gross salary -> kept
        ])
        out, demoted = anchor_amount_labels(self.words, labels)
        split = apply_policy(out, load_presets()["proof_of_income"])
        self.assertEqual(split["keep_ids"], ["w0014"])
        self.assertEqual(sorted(split["redact_ids"]), ["w0007", "w0011"])


if __name__ == "__main__":
    unittest.main()
