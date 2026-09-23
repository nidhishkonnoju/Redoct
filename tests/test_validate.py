"""Unit tests for the regex safety net (no Tesseract/Ollama needed)."""
import unittest

from ocr import OcrWord
from redact import (
    VALIDATION_PATTERNS,
    validate_context_fragments,
    validate_fragments,
)


class TestValidatePatterns(unittest.TestCase):
    def test_aadhaar_matches_spaced(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["aadhaar"].search("Aadhaar 1234 5678 9012"))

    def test_aadhaar_matches_unspaced(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["aadhaar"].search("123456789012"))

    def test_pan_matches(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["pan"].search("PAN ABCDE1234F"))

    def test_pan_case_sensitive(self):
        self.assertIsNone(VALIDATION_PATTERNS["pan"].search("abcde1234f"))

    def test_phone_matches(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["phone"].search("Call 9876543210"))

    def test_phone_rejects_landline(self):
        self.assertIsNone(VALIDATION_PATTERNS["phone"].search("Office 0221234567"))

    def test_ifsc_matches(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["ifsc"].search("IFSC HDFC0001234"))

    def test_long_digits_matches_account_number(self):
        self.assertIsNotNone(VALIDATION_PATTERNS["long_digits"].search("A/c 5010023456789012"))

    def test_email_matches(self):
        self.assertIsNotNone(
            VALIDATION_PATTERNS["email"].search("write to arjun@example.com")
        )

    def test_dates_do_not_trip_any_pattern(self):
        for text in ("22/09/2026", "01-Apr-2026", "30-Jun-2026"):
            for name, pattern in VALIDATION_PATTERNS.items():
                self.assertIsNone(pattern.search(text), f"{name} matched {text!r}")

    def test_amounts_do_not_trip(self):
        for text in ("85,000.00", "INR 85000", "24,500.00", "3,214.00"):
            for name, pattern in VALIDATION_PATTERNS.items():
                self.assertIsNone(pattern.search(text), f"{name} matched {text!r}")


def _line(words, y=0):
    out, x = [], 0
    for text in words:
        out.append(OcrWord(f"c{x:04d}{y}", text, x, y, 10 * len(text), 20, 95))
        x += 10 * len(text) + 10
    return out


class TestContextRules(unittest.TestCase):
    def test_dob_under_label_is_caught(self):
        words = _line(["Date", "of", "Birth", "14/08/1999"])
        hits = validate_context_fragments(words, set())
        self.assertEqual([h.pattern for h in hits], ["date_of_birth"])
        self.assertEqual(hits[0].matched_text, "14/08/1999")

    def test_bare_transaction_date_is_not_caught(self):
        words = _line(["02-Apr-2026", "SALARY", "CREDIT", "85,000.00"])
        self.assertEqual(validate_context_fragments(words, set()), [])

    def test_anchored_date_already_redacted_is_skipped(self):
        words = _line(["Date", "of", "Birth", "14/08/1999"])
        dob = next(w for w in words if w.text == "14/08/1999")
        self.assertEqual(validate_context_fragments(words, {dob.id}), [])

    def test_dob_label_on_its_own_line_is_caught(self):
        words = _line(["Date of Birth", "14/08/1999"])
        hits = validate_context_fragments(words, set())
        self.assertEqual([h.matched_text for h in hits], ["14/08/1999"])

    def test_label_line_then_value_on_next_line_is_caught(self):
        # Real layout: label at y=520 (small font), value at y=555 (large font),
        # so group_lines keeps them apart and the rule must look one line down.
        words = _line(["Date", "of", "Birth"], y=520) + _line(["14/08/1999"], y=555)
        hits = validate_context_fragments(words, set())
        self.assertEqual([h.matched_text for h in hits], ["14/08/1999"])


class TestValidateFragments(unittest.TestCase):
    def test_skips_already_redacted(self):
        words = [OcrWord("w0001", "9876543210", 0, 0, 10, 10, 95)]
        self.assertEqual(validate_fragments(words, {"w0001"}), [])

    def test_flags_kept_phone(self):
        words = [OcrWord("w0001", "9876543210", 0, 0, 10, 10, 95)]
        hits = validate_fragments(words, set())
        hit_patterns = {h.pattern for h in hits}
        self.assertIn("phone", hit_patterns)
        self.assertTrue(all(h.word_id == "w0001" for h in hits))


if __name__ == "__main__":
    unittest.main()
