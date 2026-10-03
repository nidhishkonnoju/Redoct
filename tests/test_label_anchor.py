"""Unit tests for the value-label anchor (Plan v3, Layer 2 -> Layer 3).

The Detect prompt teaches the model the field vocabulary, but on layouts it has
not seen it reads a label and its value as one text block: the address becomes
`label_text` (a KEEP type, so it would print under ID Verification) and the EPIC
number becomes `phone_number` (so `id_verification`'s "show the last 4" never
happens). The document's own printed label is the deterministic tie-breaker, and
these tests pin both directions: what gets re-typed, and what must never be.

Layouts are copied from the real OCR of `voter_id_card.png` / `pan_card.png`.
"""
import unittest

from ocr import OcrWord
from policy import apply_policy
from redact import anchor_label_values, load_presets


def _line(y, words, start=0):
    out, x = [], 10
    for i, text in enumerate(words):
        out.append(OcrWord(f"w{start + i:04d}", text, x, y, 9 * len(text), 18, 95))
        x += 9 * len(text) + 8
    return out


def _labels(spec):
    return [{"id": i, "type": t} for i, t in spec]


# voter_id_card.png: the label on its own row, the value on the next one.
VOTER = (
    _line(10, ["Elector's", "Photo", "Identity", "Card"])              # w0000-3
    + _line(40, ["Elector's", "Name"])                                 # w0004-5
    + _line(70, ["ARJUN", "MEHTA"], start=6)                           # w0006-7
    + _line(100, ["Elector's", "Photo", "Identity", "Card", "No."], start=8)  # w0008-12
    + _line(130, ["ABC1234567"], start=13)                             # w0013
    + _line(160, ["Address"], start=14)                                # w0014
    + _line(190, ["Flat", "12,", "MG", "Road,", "Bengaluru", "560034"], start=15)
)


class TestValueLabelAnchor(unittest.TestCase):
    def test_identifier_under_its_own_label_is_retyped(self):
        # The model called the EPIC number `phone_number`: hidden (so no leak),
        # but `id_verification` promises a last-4 reveal, which needs the right
        # type.
        labels = _labels([("w0013", "phone_number")])
        out, retyped = anchor_label_values(VOTER, labels)
        self.assertEqual(retyped, ["w0013"])
        self.assertEqual(out[0]["type"], "voter_id_number")

    def test_address_block_labelled_label_text_is_retyped(self):
        labels = _labels([
            ("w0015", "label_text"), ("w0016", "label_text"),
            ("w0017", "label_text"), ("w0018", "label_text"),
            ("w0019", "label_text"), ("w0020", "label_text"),
        ])
        out, retyped = anchor_label_values(VOTER, labels)
        self.assertEqual(retyped, [f"w00{i}" for i in range(15, 21)])
        self.assertEqual({e["type"] for e in out}, {"address"})

    def test_confident_non_identifier_labels_are_never_overridden(self):
        labels = _labels([
            ("w0006", "name"), ("w0007", "name"),      # next to "Elector's Name"
            ("w0013", "date_of_birth"),                # a wrong-but-confident date
            ("w0015", "salary_amount"),
        ])
        out, retyped = anchor_label_values(VOTER, labels)
        self.assertEqual(retyped, [])
        self.assertEqual(out, labels)

    def test_label_phrase_words_are_not_retyped(self):
        # "No." and "Card" belong to the label; they must not become PII types.
        labels = _labels([("w0012", "label_text"), ("w0003", "label_text")])
        out, retyped = anchor_label_values(VOTER, labels)
        self.assertEqual(retyped, [])
        self.assertEqual(out, labels)

    def test_unlabelled_fragments_are_left_to_the_fail_closed_default(self):
        # The address row IS the value of the "Address" label, but the model
        # never labelled those ids at all: adding a label entry would silently
        # turn a fail-closed redaction into a disclosure, so it must not happen.
        out, retyped = anchor_label_values(VOTER, [])
        self.assertEqual((out, retyped), ([], []))

    def test_pan_label_maps_the_pan_value(self):
        pan = (
            _line(10, ["Permanent", "Account", "Number"])              # w0000-2
            + _line(40, ["ABCDE1234F"], start=3)                       # w0003
        )
        out, retyped = anchor_label_values(pan, _labels([("w0003", "ifsc_code")]))
        self.assertEqual(retyped, ["w0003"])
        self.assertEqual(out[0]["type"], "pan_number")

    def test_id_verification_now_partially_masks_the_voter_number(self):
        labels = _labels([("w0013", "phone_number")])
        out, _ = anchor_label_values(VOTER, labels)
        split = apply_policy(out, load_presets()["id_verification"])
        self.assertEqual(split["partial_ids"], ["w0013"])
        self.assertEqual(split["keep_ids"], [])
        self.assertEqual(split["redact_ids"], [])

    def test_id_verification_hides_the_address_but_proof_of_address_shows_it(self):
        labels = _labels([("w0015", "label_text"), ("w0019", "label_text")])
        out, _ = anchor_label_values(VOTER, labels)
        id_split = apply_policy(out, load_presets()["id_verification"])
        self.assertEqual(id_split["redact_ids"], ["w0015", "w0019"])
        addr_split = apply_policy(out, load_presets()["proof_of_address"])
        self.assertEqual(addr_split["keep_ids"], ["w0015", "w0019"])


if __name__ == "__main__":
    unittest.main()
