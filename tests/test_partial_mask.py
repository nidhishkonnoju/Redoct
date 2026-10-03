"""Unit tests for partial masking (Plan v3 Priority 1) — the third outcome.

The policy has three outcomes per field type: keep (fully visible), redact
(black box, nothing printed) and partial (black box **with** a template-masked
value printed over it). Three invariants make the third outcome safe, and each
one is pinned below:

1. `policy.mask_value()` is the ONLY producer of printed text and it fails
   closed (returns None) whenever the reveal cannot be honoured, so a spec
   mistake degrades to a full redaction instead of a raw leak;
2. the black box is drawn for partial ids too — the mask is paint on a covered
   fragment, never an unredacted fragment;
3. a partial fragment is not "visible": it is kept out of the LLM audit input,
   and the label-anchored context net may still downgrade it to a full redact.

All offline: no Tesseract, no Ollama, no network.
"""
import unittest
from unittest.mock import patch

from PIL import Image

from llm import LLMDecision
from ocr import NormalizedImage, OcrWord
from policy import apply_policy, mask_value, partial_spec_map
from redact import (
    apply_validator,
    build_partial_masks,
    load_presets,
    render_redaction,
    run_plan_v2,
)

AADHAAR = "1234 5678 9012"


def _word(wid, text, x=10, y=10, w=None, h=30, conf=95):
    return OcrWord(wid, text, x, y, w if w is not None else 10 * len(text), h, conf)


def _labels(spec):
    return [{"id": i, "type": t} for i, t in spec]


class TestMaskValue(unittest.TestCase):
    SPEC = {"reveal": "last4", "format": "XXXX XXXX {last4}"}

    def test_reveals_only_the_last_four_characters(self):
        masked = mask_value(AADHAAR, self.SPEC)
        self.assertEqual(masked, "XXXX XXXX 9012")
        for hidden in ("1234", "5678"):
            self.assertNotIn(hidden, masked)

    def test_first_side_reveal(self):
        masked = mask_value(
            "ABCPM1234C", {"reveal": "first2", "format": "{first2}XXXXXXX"}
        )
        self.assertEqual(masked, "ABXXXXXXX")

    def test_value_shorter_than_the_reveal_fails_closed(self):
        self.assertIsNone(mask_value("123", self.SPEC))
        self.assertIsNone(mask_value("12345678", self.SPEC))  # only 4 would be hidden

    def test_missing_or_unknown_reveal_fails_closed(self):
        self.assertIsNone(mask_value(AADHAAR, {"format": "XXXX XXXX {last4}"}))
        self.assertIsNone(
            mask_value(AADHAAR, {"reveal": "middle3", "format": "XX{last4}"})
        )
        self.assertIsNone(mask_value("", self.SPEC))
        self.assertIsNone(mask_value(None, self.SPEC))
        self.assertIsNone(mask_value(AADHAAR, None))

    def test_template_must_carry_exactly_the_reveal_placeholder(self):
        self.assertIsNone(mask_value(AADHAAR, {"reveal": "last4", "format": "XXXX XXXX"}))
        self.assertIsNone(
            mask_value(AADHAAR, {"reveal": "last4", "format": "XXXX {first4} {last4}"})
        )
        # The wrong end of the value is worse than no reveal at all.
        self.assertIsNone(
            mask_value(AADHAAR, {"reveal": "last4", "format": "XXXX {first4}"})
        )

    def test_template_may_not_embed_literal_value_characters(self):
        self.assertIsNone(
            mask_value(AADHAAR, {"reveal": "last4", "format": "ABCD XXXX {last4}"})
        )
        self.assertIsNone(
            mask_value(AADHAAR, {"reveal": "last4", "format": "1234 XXXX {last4}"})
        )


class TestThreeWayPolicy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def test_partial_beats_redact_and_keep(self):
        preset = {
            "keep_types": ["aadhaar_number"],
            "redact_types": ["aadhaar_number"],
            "partial_types": {
                "aadhaar_number": {"reveal": "last4", "format": "XXXXXXXX{last4}"}
            },
        }
        out = apply_policy([{"id": "a", "type": "aadhaar_number"}], preset)
        self.assertEqual(out["partial_ids"], ["a"])
        self.assertEqual(out["keep_ids"], [])
        self.assertEqual(out["redact_ids"], [])

    def test_every_id_lands_in_exactly_one_bucket(self):
        for pkey, preset in self.presets.items():
            frags = _labels([("a", "aadhaar_number"), ("b", "name"), ("c", "address")])
            out = apply_policy(frags, preset)
            buckets = [out["redact_ids"], out["keep_ids"], out["partial_ids"]]
            assigned = set().union(*buckets)
            self.assertEqual(assigned, {"a", "b", "c"}, pkey)
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    self.assertFalse(set(buckets[i]) & set(buckets[j]), pkey)

    def test_no_preset_declares_a_type_as_both_kept_and_masked(self):
        for pkey, preset in self.presets.items():
            masked = set(partial_spec_map(preset))
            # A type is masked *instead of* kept or redacted, so it must not be
            # in the keep list, and it must never be purpose-critical (those are
            # the fields the purpose exists to reveal in full).
            self.assertFalse(masked & set(preset["keep_types"]), pkey)
            self.assertFalse(masked & set(preset["purpose_critical_types"]), pkey)
            for ftype, spec in partial_spec_map(preset).items():
                # A declared spec must be bounded: it discloses only its own
                # reveal window, and it fails closed on anything shorter.
                self.assertEqual(spec.get("reveal"), "last4", f"{pkey}/{ftype}")
                self.assertIsNone(mask_value("123", spec), f"{pkey}/{ftype}")
                masked = mask_value("123456789012", spec)
                self.assertIsNotNone(masked, f"{pkey}/{ftype}")
                self.assertTrue(masked.endswith("9012"), f"{pkey}/{ftype}")
                self.assertNotIn("1234", masked, f"{pkey}/{ftype}")

    def test_id_verification_masks_identifiers_instead_of_hiding_them(self):
        preset = self.presets["id_verification"]
        out = apply_policy(
            _labels([
                ("a", "aadhaar_number"), ("p", "pan_number"),
                ("v", "voter_id_number"), ("d", "date_of_birth"), ("n", "name"),
            ]),
            preset,
        )
        self.assertEqual(out["partial_ids"], ["a", "p", "v"])
        self.assertEqual(out["redact_ids"], ["d"])
        self.assertEqual(out["keep_ids"], ["n"])


class TestValidatorWithPartial(unittest.TestCase):
    def setUp(self):
        self.words = [
            _word("w0000", "ARJUN", 10, 10),
            _word("w0001", AADHAAR, 250, 10, w=170),
            _word("w0002", "9876543210", 10, 60, w=110),
        ]
        self.preset = load_presets()["id_verification"]

    def test_partial_ids_are_not_unfilled(self):
        decision = LLMDecision(redact_ids=[], keep_ids=["w0000"], partial_ids=["w0001"])
        corrected, _, unfilled = apply_validator(self.words, decision)
        self.assertNotIn("w0001", unfilled)
        self.assertEqual(unfilled, ["w0002"])
        self.assertEqual(corrected.partial_ids, ["w0001"])

    def test_type_regex_net_does_not_reclassify_a_partial_fragment(self):
        # The aadhaar pattern matches w0001's raw text, but the fragment is
        # pre-masked in the output, so the net must leave it partial.
        decision = LLMDecision(redact_ids=[], keep_ids=[], partial_ids=["w0001"])
        corrected, hits, _ = apply_validator(self.words, decision)
        self.assertEqual(corrected.partial_ids, ["w0001"])
        self.assertNotIn("w0001", corrected.redact_ids)
        self.assertNotIn("w0001", {h.word_id for h in hits})

    def test_context_net_still_applies_when_partial(self):
        # A masked birth date is still a birth date: the label-anchored rule
        # outranks the partial outcome and forces a plain full redaction.
        words = [
            _word("d0", "Date", 10, 10), _word("d1", "of", 70, 10),
            _word("d2", "Birth", 110, 10), _word("d3", "14/08/1999", 190, 10),
        ]
        decision = LLMDecision(redact_ids=[], keep_ids=[], partial_ids=["d3"])
        corrected, hits, _ = apply_validator(words, decision)
        self.assertEqual(corrected.partial_ids, [])
        self.assertIn("d3", corrected.redact_ids)
        self.assertIn("date_of_birth", {h.pattern for h in hits})

    def test_redact_outranks_partial_on_a_duplicated_id(self):
        decision = LLMDecision(redact_ids=["w0001"], keep_ids=[], partial_ids=["w0001"])
        corrected, _, unfilled = apply_validator(self.words, decision)
        self.assertIn("w0001", corrected.redact_ids)
        self.assertEqual(corrected.partial_ids, [])
        self.assertNotIn("w0001", unfilled)

    def test_keep_and_partial_are_disjoint_in_the_corrected_decision(self):
        decision = LLMDecision(keep_ids=["w0001"], partial_ids=["w0001"])
        corrected, _, _ = apply_validator(self.words, decision)
        self.assertEqual(corrected.keep_ids, [])
        self.assertEqual(corrected.partial_ids, ["w0001"])

    def test_build_partial_masks_uses_the_preset_spec(self):
        masks, unmaskable = build_partial_masks(
            self.words, _labels([("w0001", "aadhaar_number")]), ["w0001"], self.preset
        )
        self.assertEqual(masks, {"w0001": "XXXX XXXX 9012"})
        self.assertEqual(unmaskable, [])

    def test_short_value_is_reported_unmaskable(self):
        words = [_word("s0", "12345")]
        masks, unmaskable = build_partial_masks(
            words, _labels([("s0", "aadhaar_number")]), ["s0"], self.preset
        )
        self.assertEqual(masks, {})
        self.assertEqual(unmaskable, ["s0"])

    def test_id_absent_from_the_document_is_reported_unmaskable(self):
        masks, unmaskable = build_partial_masks(
            self.words, _labels([("ghost", "aadhaar_number")]), ["ghost"], self.preset
        )
        self.assertEqual(masks, {})
        self.assertEqual(unmaskable, ["ghost"])


class TestMaskRendering(unittest.TestCase):
    def _render(self, word, mask, size=(600, 120)):
        img = Image.new("RGB", size, (255, 255, 255))
        out = render_redaction(img, [word], [], partial_masks={word.id: mask})
        return img, out

    def test_masked_value_is_boxed_and_printed_over_the_box(self):
        word = _word("a0", AADHAAR, 10, 10, w=220, h=40)
        _, out = self._render(word, "XXXX XXXX 9012")
        box = (word.x - 5, word.y - 5, word.x + word.w + 5, word.y + word.h + 5)
        pixels = [
            out.getpixel((x, y))
            for x in range(box[0], box[2])
            for y in range(box[1], box[3])
        ]
        self.assertTrue(any(min(p) > 150 for p in pixels), "mask text not drawn")
        self.assertTrue(any(max(p) < 60 for p in pixels), "black box missing")
        # The box is opaque: nothing of the original value survives at the edges.
        self.assertEqual(out.getpixel((box[0], box[1])), (0, 0, 0))
        self.assertEqual(out.getpixel((box[2] - 1, box[3] - 1)), (0, 0, 0))

    def test_unprintable_mask_leaves_the_black_box_empty(self):
        # A box too small for the floor font must stay blank — never unboxed,
        # never clipped into a partial leak.
        word = _word("a1", AADHAAR, 10, 10, w=6, h=6)
        _, out = self._render(word, "XXXX XXXX 9012")
        pixels = [
            out.getpixel((x, y))
            for x in range(5, word.x + word.w + 5)
            for y in range(5, word.y + word.h + 5)
        ]
        self.assertTrue(pixels)
        self.assertTrue(all(max(p) < 60 for p in pixels))

    def test_full_redaction_still_has_no_text(self):
        word = _word("a2", "Flat 12, MG Road", 10, 10, w=200, h=40)
        img = Image.new("RGB", (600, 120), (255, 255, 255))
        out = render_redaction(img, [word], [word.id])
        pixels = [
            out.getpixel((x, y))
            for x in range(5, word.x + word.w + 5)
            for y in range(5, word.y + word.h + 5)
        ]
        self.assertTrue(pixels)
        self.assertTrue(all(max(p) < 60 for p in pixels))

    def test_original_image_is_not_mutated(self):
        word = _word("a3", AADHAAR, 10, 10, w=220, h=40)
        img, out = self._render(word, "XXXX XXXX 9012")
        self.assertIsNot(img, out)
        self.assertEqual(img.getpixel((20, 20)), (255, 255, 255))


class TestPipelinePartialWiring(unittest.TestCase):
    """End-to-end (stubbed LLM/OCR): partial ids never reach the audit."""

    def setUp(self):
        self.words = [
            _word("w0000", "ARJUN", 10, 10, w=60),
            _word("w0001", "MEHTA", 90, 10, w=60),
            _word("w0002", "Aadhaar", 10, 50, w=80),
            _word("w0003", AADHAAR, 110, 50, w=170, h=30),
            _word("w0004", "Flat", 10, 90, w=40),
            _word("w0005", "MG", 60, 90, w=30),
            _word("w0006", "Road", 100, 90, w=50),
            _word("w0007", "9876543210", 10, 130, w=110),
        ]
        self.labels = _labels([
            ("w0000", "name"), ("w0001", "name"), ("w0002", "label_text"),
            ("w0003", "aadhaar_number"), ("w0004", "address"), ("w0005", "address"),
            ("w0006", "address"), ("w0007", "phone_number"),
        ])
        self.audit_inputs = []

    def _run(self):
        norm = NormalizedImage(
            image=Image.new("RGB", (400, 200), (255, 255, 255)), scale=1.0,
            upscaled=False,
        )

        def _audit(visible, visible_ids, label):
            self.audit_inputs.append(list(visible_ids))
            return [], "clean"

        with patch("redact.normalize_image", return_value=norm), patch(
            "redact.extract_words", return_value=self.words
        ), patch("llm.classify", return_value="aadhaar_card"), patch(
            "llm.detect_batched", return_value=self.labels
        ), patch("llm.validate_llm", side_effect=_audit):
            return run_plan_v2(
                Image.new("RGB", (400, 200), (255, 255, 255)),
                "id_verification",
                load_presets()["id_verification"],
            )

    def test_partial_fragment_is_never_sent_to_the_audit(self):
        result = self._run()
        self.assertEqual(result.partial_ids, ["w0003"])
        self.assertNotIn("w0003", result.llm_redact_ids)
        self.assertEqual(self.audit_inputs, [["w0000", "w0001", "w0002"]])
        self.assertTrue(
            any("partially masked" in w for w in result.warnings), result.warnings
        )

    def test_partial_fragment_really_gets_a_box_and_a_print_in_the_output(self):
        result = self._run()
        pixels = [
            result.output_image.getpixel((x, y))
            for x in range(105, 285)
            for y in range(45, 85)
        ]
        self.assertTrue(any(max(p) < 60 for p in pixels), "black box missing")
        self.assertTrue(any(min(p) > 150 for p in pixels), "mask text missing")
        self.assertFalse(
            any("not covered by drawn boxes" in w for w in result.warnings),
            result.warnings,
        )

    def test_unmaskable_partial_falls_back_to_full_redaction(self):
        # Same wiring, but the value is too short for a safe reveal: the
        # fragment must move to redact_ids (never into keep, never printed).
        self.words[3] = _word("w0003", "12345", 110, 50, w=60, h=30)
        result = self._run()
        self.assertEqual(result.partial_ids, [])
        self.assertIn("w0003", result.llm_redact_ids)
        self.assertTrue(
            any("could not be masked" in w for w in result.warnings), result.warnings
        )


if __name__ == "__main__":
    unittest.main()
