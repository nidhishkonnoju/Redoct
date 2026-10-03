"""Unit tests for the Plan v3 line protocol (Priority 0).

These run entirely offline: they drive the parsers and the layer functions with
a stubbed `_call_generate`, so no Ollama and no GPU are needed. The protocol
exists because a 3B model truncating a nested JSON object mid-string
(`Unterminated string ... column 2744`) used to lose a whole batch; a truncated
line-based reply still parses line by line.
"""
import unittest
from unittest.mock import patch

import llm
import prompts
from ocr import OcrWord


def _words(count, text="X"):
    return [
        OcrWord(f"w{i:04d}", f"{text}{i}", 10, 10 * i, 20, 10, 95)
        for i in range(1, count + 1)
    ]


class TestLabelLines(unittest.TestCase):
    def setUp(self):
        self.words = _words(3)
        self.forward, self.backward = llm.build_id_maps(self.words)

    def parse(self, raw):
        return llm._parse_labels(raw, self.backward)

    def test_clean_lines(self):
        out = self.parse("a00: name\na01: account_number\na02: label_text")
        self.assertEqual(
            [(e["id"], e["type"]) for e in out],
            [("w0001", "name"), ("w0002", "account_number"), ("w0003", "label_text")],
        )

    def test_truncated_reply_keeps_the_lines_it_printed(self):
        # The Plan v3 failure mode: the model runs out of budget mid-answer.
        # JSON lost everything; the line protocol keeps two good labels, and the
        # mid-word value still resolves because `account_num` has a unique prefix.
        out = self.parse("a00: name\na01: account_num")
        self.assertEqual(len(out), 2)
        self.assertEqual(out[1]["type"], "account_number")

    def test_ambiguous_truncated_value_fails_closed_to_other(self):
        # `trans` opens both transaction_date (a KEEP type) and transaction_line
        # (a REDACT type), so guessing would decide visibility on a coin flip.
        # Unmatched values are `other` — and when *every* value is unmatched the
        # parser reports a parse failure so the batch splitter can retry smaller.
        self.assertIsNone(llm._match_type("trans"))
        self.assertIsNone(llm._match_type("o"))
        # `transac` is longer but still opens both ledger types: no guess.
        self.assertIsNone(llm._match_type("transac"))
        with self.assertRaises(ValueError):
            self.parse("a00: trans\na01: o")

    def test_unique_prefix_of_a_keep_type_is_not_guessed_from_two_letters(self):
        # 3 characters is below the rescue threshold: only `other` can be a
        # two-letter answer, and no prefix that short is trusted.
        self.assertIsNotNone(llm._match_type("salary_amo"))
        self.assertIsNone(llm._match_type("sal"))

    def test_markdown_and_bullet_noise_is_tolerated(self):
        raw = "- **a00**: name\n* a01 = address\n1. `a02` -> phone_number\n"
        self.assertEqual(
            [e["type"] for e in self.parse(raw)],
            ["name", "address", "phone_number"],
        )

    def test_chatty_value_still_matches_the_longest_type(self):
        # "other_amount" must win over "other" for a value containing both.
        out = self.parse("a00: the type is other_amount\na01: it is other")
        self.assertEqual([e["type"] for e in out], ["other_amount", "other"])

    def test_unknown_ids_and_unknown_types_fail_closed(self):
        raw = "a00: name\nZZ: address\nb99: nonsense_type"
        out = self.parse(raw)
        self.assertEqual([(e["id"], e["type"]) for e in out], [("w0001", "name")])

    def test_all_unrecognized_values_raise_so_the_batch_can_split(self):
        # Echoing the fragment text back is the dangerous degenerate case: it
        # must look like a parse failure, not like "every fragment is other".
        with self.assertRaises(ValueError):
            self.parse("a00: X1\na01: X2\na02: X3")

    def test_empty_and_prose_output(self):
        self.assertEqual(self.parse("I cannot help with that."), [])

    def test_json_reply_still_parses_as_a_fallback(self):
        raw = '{"labels": [{"id": "a00", "type": "name"}]}'
        self.assertEqual(
            [(e["id"], e["type"]) for e in self.parse(raw)], [("w0001", "name")]
        )

    def test_duplicate_id_keeps_the_first_label(self):
        out = self.parse("a00: name\na00: address")
        self.assertEqual([(e["id"], e["type"]) for e in out], [("w0001", "name")])


class TestAuditLines(unittest.TestCase):
    def test_flags_and_reason(self):
        flagged, reasoning = llm._parse_audit(
            "FLAGGED: w0002, w0007\nREASON: card number is visible"
        )
        self.assertEqual(flagged, ["w0002", "w0007"])
        self.assertEqual(reasoning, "card number is visible")

    def test_none_means_no_flags(self):
        for body in ("none", "NONE", "-", "[]"):
            flagged, _ = llm._parse_audit(f"FLAGGED: {body}\nREASON: all clear")
            self.assertEqual(flagged, [], body)

    def test_json_reply_still_parses_as_a_fallback(self):
        flagged, reasoning = llm._parse_audit(
            '{"flagged_ids": ["w0003"], "reasoning": "dob"}'
        )
        self.assertEqual(flagged, ["w0003"])
        self.assertEqual(reasoning, "dob")

    def test_missing_flagged_line_is_a_parse_failure(self):
        for raw in ("I see no problems.", "REASON: looks fine"):
            with self.assertRaises(ValueError, msg=raw):
                llm._parse_audit(raw)


class TestLayerPromptsAndCalls(unittest.TestCase):
    def test_detect_request_never_asks_for_json(self):
        with patch.object(llm, "_call_generate", return_value="a00: name\na01: name") as call:
            out = llm.detect(_words(2))
        system, user = call.call_args.args[0], call.call_args.args[1]
        self.assertEqual([e["type"] for e in out], ["name", "name"])
        for text in (system, user):
            self.assertNotIn("Respond ONLY as JSON", text)
        self.assertIn("ONE LINE PER FRAGMENT", user)
        self.assertIn("<id>: <type>", user)

    def test_preserve_note_is_added_for_known_document_types_only(self):
        statement = prompts.detect_system("bank_statement")
        self.assertIn(prompts.DETECT_PRESERVE_NOTES["bank_statement"], statement)
        self.assertEqual(prompts.detect_system("other"), prompts.DETECT_BASE_SYSTEM)
        self.assertEqual(prompts.detect_system("nonsense"), prompts.DETECT_BASE_SYSTEM)

    def test_prompt_vocabulary_cannot_drift_from_the_parser_vocabulary(self):
        for field_type in prompts.FIELD_TYPES:
            self.assertIn(field_type, prompts.DETECT_VOCABULARY)
        self.assertEqual(set(llm.FIELD_TYPES), set(prompts.FIELD_TYPES))

    def test_prompt_teaches_the_same_birth_date_labels_as_the_regex_net(self):
        import redact

        self.assertEqual(tuple(redact.DOB_ANCHORS), tuple(prompts.DOB_ANCHORS))
        system = prompts.detect_system("other").lower()
        for anchor in prompts.DOB_ANCHORS:
            self.assertIn(anchor.lower(), system, anchor)

    def test_detect_passes_the_document_type_note_to_the_request(self):
        with patch.object(llm, "_call_generate", return_value="a00: name") as call:
            llm.detect(_words(1), document_type="bank_statement")
        self.assertIn(
            prompts.DETECT_PRESERVE_NOTES["bank_statement"], call.call_args.args[0]
        )

    def test_missing_ids_are_left_unlabelled_for_the_pipeline_to_fail_closed(self):
        with patch.object(llm, "_call_generate", return_value="a00: name"):
            out = llm.detect(_words(2))
        self.assertEqual([e["id"] for e in out], ["w0001"])

    def test_detect_retries_then_raises(self):
        with patch.object(llm, "_call_generate", return_value="a00: X1"):
            with self.assertRaises(llm.OllamaParseError):
                llm.detect(_words(2))

    def test_audit_request_carries_the_line_rules_and_filters_foreign_ids(self):
        words = _words(2)
        with patch.object(llm, "_call_generate", return_value="FLAGGED: w0001, w9999") as call:
            flagged, _ = llm.validate_llm(words, [w.id for w in words], "Proof of Income")
        self.assertEqual(flagged, ["w0001"])
        self.assertIn("FLAGGED:", call.call_args.args[1])
        self.assertIn("REASON:", call.call_args.args[1])

    def test_layer_call_log_records_a_name_per_layer(self):
        # `_call_generate` is what stamps the layer name into LAST_LAYER_CALLS,
        # so the contract to pin here is the name each layer passes down.
        with patch.object(llm, "_call_generate", return_value="a00: name") as call:
            llm.detect(_words(1))
        self.assertEqual(call.call_args.kwargs["layer"], "detect")
        with patch.object(llm, "_call_generate", return_value="bank_statement") as call:
            llm.classify(_words(1))
        self.assertEqual(call.call_args.kwargs["layer"], "classify")
        with patch.object(llm, "_call_generate", return_value="FLAGGED: none") as call:
            llm.validate_llm(_words(1), ["w0001"], "Proof of Income")
        self.assertEqual(call.call_args.kwargs["layer"], "audit")

    def test_real_generate_appends_the_layer_name(self):
        calls: list[dict] = []
        llm.LAST_LAYER_CALLS.clear()
        fake_resp = type("R", (), {"status_code": 200, "json": lambda self: {"response": "ok"}})()
        with patch.object(llm.requests, "post", return_value=fake_resp):
            llm._call_generate("sys", "user", layer="detect")
            llm._call_generate("sys", "user", layer="audit")
        calls = list(llm.LAST_LAYER_CALLS)
        llm.LAST_LAYER_CALLS.clear()
        self.assertEqual([c["layer"] for c in calls], ["detect", "audit"])


if __name__ == "__main__":
    unittest.main()
