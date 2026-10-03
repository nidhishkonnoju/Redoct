"""Unit tests for the deterministic policy lookup (Plan v2, Layer 3).

No services needed — every field type x every preset runs offline.
"""
import unittest

from llm import FIELD_TYPES
from policy import apply_policy
from redact import load_presets


def _label(prefix, ftype, n=3):
    return [{"id": f"{prefix}{i}", "type": ftype} for i in range(n)]


class TestPolicyLookup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def test_every_field_type_has_a_decision_in_every_preset(self):
        for pkey, preset in self.presets.items():
            frags = [{"id": f"t{i}", "type": t} for i, t in enumerate(FIELD_TYPES)]
            out = apply_policy(frags, preset)
            buckets = [out["redact_ids"], out["keep_ids"], out["partial_ids"]]
            assigned = set().union(*buckets)
            self.assertEqual(assigned, {f["id"] for f in frags}, pkey)
            # ...and exactly one decision each (keep / partial / redact).
            for i in range(len(buckets)):
                for j in range(i + 1, len(buckets)):
                    self.assertFalse(set(buckets[i]) & set(buckets[j]), pkey)

    def test_unknown_type_redacts_everywhere(self):
        for preset in self.presets.values():
            out = apply_policy([{"id": "x", "type": "not_a_real_type"}], preset)
            self.assertIn("x", out["redact_ids"])

    def test_redact_beats_keep_on_conflict(self):
        preset = {"keep_types": ["name"], "redact_types": ["name"]}
        out = apply_policy([{"id": "x", "type": "name"}], preset)
        self.assertIn("x", out["redact_ids"])

    def test_order_stable_and_deduped(self):
        preset = {"keep_types": ["name"], "redact_types": []}
        frags = [{"id": "a", "type": "name"}, {"id": "a", "type": "name"},
                 {"id": "b", "type": "other"}]
        out = apply_policy(frags, preset)
        self.assertEqual(out["keep_ids"], ["a"])
        self.assertEqual(out["redact_ids"], ["b"])


class TestFatherNameRegression(unittest.TestCase):
    """Bug 1 regression: father_name must redact in every preset that lists it."""

    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def test_father_name_redacts_where_listed(self):
        listed = [k for k, p in self.presets.items() if "father_name" in p["redact_types"]]
        self.assertTrue(listed)
        for pkey in listed:
            out = apply_policy([{"id": "f1", "type": "father_name"}], self.presets[pkey])
            self.assertIn("f1", out["redact_ids"], pkey)

    def test_father_and_person_names_split_correctly(self):
        preset = self.presets["proof_of_income"]
        out = apply_policy([
            {"id": "n1", "type": "name"},
            {"id": "f1", "type": "father_name"},
        ], preset)
        self.assertIn("n1", out["keep_ids"])
        self.assertIn("f1", out["redact_ids"])


class TestLabelVsValue(unittest.TestCase):
    """Bug 2 sibling: labels and values must never conflate."""

    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def test_label_stays_visible_value_hides(self):
        preset = self.presets["id_verification"]  # keeps label_text, redacts date_of_birth
        out = apply_policy([
            {"id": "l1", "type": "label_text"},
            {"id": "v1", "type": "date_of_birth"},
        ], preset)
        self.assertIn("l1", out["keep_ids"])
        self.assertIn("v1", out["redact_ids"])


class TestAuditGuardrail(unittest.TestCase):
    """Plan v2 Layer 5b: policy beats the fresh-context audit on critical fields."""

    @classmethod
    def setUpClass(cls):
        cls.presets = load_presets()

    def test_purpose_critical_types_are_kept_by_every_preset(self):
        for pkey, preset in self.presets.items():
            self.assertIn("purpose_critical_types", preset, pkey)
            critical = set(preset["purpose_critical_types"])
            self.assertTrue(critical, pkey)
            self.assertLessEqual(critical, set(preset["keep_types"]), pkey)

    def test_blocks_flag_on_purpose_critical_field(self):
        from redact import audit_guardrail

        actionable, blocked = audit_guardrail(
            ["n1", "l1"], ["n1", "l1", "n2"], {"n1"}
        )
        self.assertEqual(blocked, ["n1"])
        self.assertEqual(actionable, ["l1"])

    def test_drops_ids_that_are_not_visible(self):
        from redact import audit_guardrail

        actionable, blocked = audit_guardrail(["ghost"], ["l1"], set())
        self.assertEqual(actionable, [])
        self.assertEqual(blocked, [])

    def test_dedupes_and_keeps_order(self):
        from redact import audit_guardrail

        actionable, blocked = audit_guardrail(
            ["b", "a", "b", "z"], ["a", "b", "z"], {"z"}
        )
        self.assertEqual(actionable, ["b", "a"])
        self.assertEqual(blocked, ["z"])


if __name__ == "__main__":
    unittest.main()
