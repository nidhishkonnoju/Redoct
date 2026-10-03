"""Unit tests for the name-anchoring rule (Plan v2, Layer 2 -> Layer 3).

`name` is a KEEP type for three of four presets. The model calls merchants
`name`, so a name on a ledger row must be demoted. The line layouts below are
copied from the real sample docs (see the OCR line dump), including the PAN card
case where the "Name" label and its value sit on separate rows.
"""
import unittest

from ocr import OcrWord
from policy import apply_policy
from redact import (
    anchor_name_labels,
    anchor_third_party_names,
    load_presets,
)


def _line(y, words, start=0):
    out, x = [], 10
    for i, text in enumerate(words):
        out.append(OcrWord(f"w{start + i:04d}", text, x, y, 9 * len(text), 18, 95))
        x += 9 * len(text) + 8
    return out


# Real layouts: bank statement rows, PAN card, salary slip.
STATEMENT = (
    _line(10, ["Account", "Holder:", "ARJUN", "MEHTA"])                     # w0000-3
    + _line(40, ["02-Apr-2026", "SALARY", "CREDIT", "-", "INFOTECH",
                 "SOLUTIONS", "PVT", "LTD", "85,000.00"], start=4)         # w0004-12
    + _line(70, ["15-Apr-2026", "Card", "Purchase", "BIGBASKET",
                 "3,214.00"], start=13)                                    # w0013-17
    + _line(100, ["Closing", "Balance:", "INR", "3,42,118.00"], start=18)  # w0018-21
)
PAN = (
    _line(10, ["Name", "PHOTO"])                                            # w0000-1
    + _line(40, ["ARJUN", "MEHTA"], start=2)                               # w0002-3
    + _line(70, ["Father's", "Name"], start=4)                              # w0004-5
    + _line(100, ["RAKESH", "MEHTA"], start=6)                              # w0006-7
)
SLIP = (
    _line(10, ["Employee", "Name:", "ARJUN", "MEHTA", "Employee", "ID:",
               "EMP20451"])                                                 # w0000-6
    + _line(40, ["GROSS", "SALARY", "85,000"], start=7)                     # w0007-9
)


def _labels(spec):
    return [{"id": i, "type": t} for i, t in spec]


class TestNameAnchoring(unittest.TestCase):
    def test_merchant_on_a_ledger_row_is_demoted(self):
        out, demoted = anchor_name_labels(
            STATEMENT, _labels([("w0016", "name")])  # BIGBASKET
        )
        self.assertEqual(demoted, ["w0016"])
        self.assertEqual(out[0]["type"], "other")

    def test_account_holder_name_on_its_own_row_is_kept(self):
        out, demoted = anchor_name_labels(
            STATEMENT, _labels([("w0002", "name"), ("w0003", "name")])
        )
        self.assertEqual(demoted, [])
        self.assertEqual([e["type"] for e in out], ["name", "name"])

    def test_pan_card_name_value_row_is_kept(self):
        # The value row ("ARJUN MEHTA") carries no label word and no amount.
        out, demoted = anchor_name_labels(
            PAN, _labels([("w0002", "name"), ("w0003", "name")])
        )
        self.assertEqual(demoted, [])
        self.assertEqual([e["type"] for e in out], ["name", "name"])

    def test_employee_id_is_not_read_as_a_money_value(self):
        # "EMP20451" must not make the employee-name row look like a ledger row.
        out, demoted = anchor_name_labels(
            SLIP, _labels([("w0002", "name"), ("w0003", "name")])
        )
        self.assertEqual(demoted, [])
        self.assertEqual([e["type"] for e in out], ["name", "name"])

    def test_other_types_are_untouched(self):
        labels = _labels([("w0016", "label_text"), ("w0012", "salary_amount")])
        out, demoted = anchor_name_labels(STATEMENT, labels)
        self.assertEqual(demoted, [])
        self.assertEqual(out, labels)

    def test_policy_redacts_the_counterparty_but_keeps_the_holder(self):
        labels = _labels([("w0002", "name"), ("w0016", "name")])
        out, demoted = anchor_name_labels(STATEMENT, labels)
        split = apply_policy(out, load_presets()["proof_of_income"])
        self.assertEqual(split["keep_ids"], ["w0002"])
        self.assertEqual(split["redact_ids"], ["w0016"])


class TestThirdPartyNameAnchoring(unittest.TestCase):
    """A relative's name the model called `name` must not survive as `name`.

    `name` is a keep type *and* a purpose-critical type in every preset, so a
    `name` label on the father's value both prints it and shields it from the
    audit guardrail. The anchor settles it deterministically.
    """

    def test_father_value_on_the_next_row_is_demoted(self):
        # Card layout (real PAN / voter OCR): the label gets its own row and the
        # value sits on the row below — including the label word itself, which
        # the model also calls `name` and which is never the value.
        labels = _labels([("w0005", "name"), ("w0006", "name"), ("w0007", "name")])
        out, demoted = anchor_third_party_names(PAN, labels)
        self.assertEqual(demoted, ["w0006", "w0007"])
        self.assertEqual([e["type"] for e in out], ["name", "father_name", "father_name"])

    def test_same_row_label_and_value(self):
        # Marksheet layout: "Father's Name: RAKESH MEHTA" in one row.
        words = _line(10, ["Father's", "Name:", "RAKESH", "MEHTA"])
        labels = _labels([(f"w{i:04d}", "name") for i in range(4)])
        out, demoted = anchor_third_party_names(words, labels)
        self.assertEqual(demoted, ["w0002", "w0003"])
        self.assertEqual([e["type"] for e in out],
                         ["name", "name", "father_name", "father_name"])

    def test_subject_name_row_is_untouched(self):
        labels = _labels([("w0002", "name"), ("w0003", "name")])
        out, demoted = anchor_third_party_names(PAN, labels)
        self.assertEqual(demoted, [])
        self.assertEqual(out, labels)

    def test_spouse_and_guardian_labels_are_covered(self):
        for anchor in ("Spouse", "Guardian"):
            with self.subTest(anchor=anchor):
                words = _line(10, [anchor, "Name"]) + _line(40, ["SUNITA", "MEHTA"],
                                                            start=2)
                labels = _labels([("w0002", "name"), ("w0003", "name")])
                out, demoted = anchor_third_party_names(words, labels)
                self.assertEqual(demoted, ["w0002", "w0003"])
                self.assertEqual([e["type"] for e in out],
                                 ["father_name", "father_name"])

    def test_non_name_types_are_untouched(self):
        labels = _labels([("w0006", "label_text"), ("w0007", "date_of_birth")])
        out, demoted = anchor_third_party_names(PAN, labels)
        self.assertEqual(demoted, [])
        self.assertEqual(out, labels)

    def test_no_name_labels_short_circuits(self):
        out, demoted = anchor_third_party_names(PAN, [])
        self.assertEqual((out, demoted), ([], []))

    def test_policy_redacts_the_father_but_keeps_the_subject(self):
        labels = _labels([("w0002", "name"), ("w0003", "name"),
                          ("w0006", "name"), ("w0007", "name")])
        out, _ = anchor_third_party_names(PAN, labels)
        split = apply_policy(out, load_presets()["id_verification"])
        self.assertEqual(split["keep_ids"], ["w0002", "w0003"])
        self.assertEqual(split["redact_ids"], ["w0006", "w0007"])


if __name__ == "__main__":
    unittest.main()
