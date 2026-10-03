"""Unit tests for geometry-based row reconstruction (Plan v3, shared utility).

`reconstruct_rows()` is the fix for the merged-OCR-column miss: the row-level
rules must not trust Tesseract's own line/paragraph/block numbering, because a
ledger whose date and amount land in a different OCR line is exactly how a
counterparty name leaked earlier. These tests pin the contract with synthetic
bounding boxes only — no Tesseract, no Ollama.
"""
import unittest

from ocr import OcrWord, reconstruct_rows, row_for_fragment, row_text


def _w(wid, text, x, y, h=20, w=10, block=1, par=1, line=1):
    return OcrWord(wid, text, x, y, w * len(text), h, 95, block, par, line)


class TestReconstructRows(unittest.TestCase):
    def test_same_visual_row_merges_despite_different_tesseract_grouping(self):
        # Merged columns: tesseract reports each fragment in its own block and
        # line, so line_num says "three rows" while the boxes say "one row".
        words = [
            _w("w1", "Card", 10, 100, block=1, par=1, line=1),
            _w("w2", "Purchase", 90, 100, block=3, par=2, line=4),
            _w("w3", "BIGBASKET", 260, 100, block=5, par=3, line=9),
        ]
        rows = reconstruct_rows(words)
        self.assertEqual(len(rows), 1)
        self.assertEqual([w.text for w in rows[0]], ["Card", "Purchase", "BIGBASKET"])

    def test_separate_rows_stay_separate(self):
        words = [
            _w("w1", "15-Apr-2026", 10, 100),
            _w("w2", "3,214.00", 400, 100),
            _w("w3", "20-Jun-2026", 10, 160),
            _w("w4", "640.00", 400, 160),
        ]
        rows = reconstruct_rows(words)
        self.assertEqual(len(rows), 2)
        self.assertEqual([w.id for w in rows[0]], ["w1", "w2"])
        self.assertEqual([w.id for w in rows[1]], ["w3", "w4"])

    def test_row_is_sorted_left_to_right(self):
        words = [
            _w("w1", "BIGBASKET", 500, 100),
            _w("w2", "3,214.00", 900, 100),
            _w("w3", "Purchase", 200, 100),
        ]
        rows = reconstruct_rows(words)
        self.assertEqual([w.text for w in rows[0]], ["Purchase", "BIGBASKET", "3,214.00"])

    def test_binds_a_row_that_the_global_median_grouping_splits(self):
        # A 11px baseline drift between two 20px-tall fragments on one visual
        # row: a global-median tolerance (10px here) rejects it, the per-pair
        # tolerance scales with the fragment and accepts it. Binding more
        # fragments to a row can only *demote* more counterparties, never fewer,
        # so the generous side is the fail-closed side.
        from ocr import group_lines

        words = [_w("w1", "MERCHANT", 10, 100, h=20), _w("w2", "500.00", 500, 111, h=20)]
        self.assertEqual([len(r) for r in group_lines(words)], [1, 1])
        rows = reconstruct_rows(words)
        self.assertEqual([len(r) for r in rows], [2])

    def test_label_and_value_merge_when_font_sizes_differ(self):
        # Big-font value in a small-font row: same visual row, 2x height.
        words = [_w("w1", "Date", 10, 100, h=14), _w("w2", "14/08/1999", 120, 96, h=30)]
        rows = reconstruct_rows(words)
        self.assertEqual(len(rows), 1)

    def test_tolerance_ratio_is_configurable(self):
        words = [_w("w1", "A", 10, 100, h=20), _w("w2", "B", 100, 111, h=20)]
        self.assertEqual(len(reconstruct_rows(words)), 1)
        self.assertEqual(len(reconstruct_rows(words, y_tolerance_ratio=0.1)), 2)

    def test_rows_are_top_to_bottom(self):
        words = [_w("w1", "second", 10, 200), _w("w2", "first", 10, 100)]
        rows = reconstruct_rows(words)
        self.assertEqual([row_text(r) for r in rows], ["first", "second"])

    def test_empty_input(self):
        self.assertEqual(reconstruct_rows([]), [])

    def test_row_for_fragment_finds_the_owning_row(self):
        words = [
            _w("w1", "15-Apr-2026", 10, 100),
            _w("w2", "BIGBASKET", 300, 100),
            _w("w3", "Closing", 10, 900),
        ]
        rows = reconstruct_rows(words)
        self.assertEqual(
            [w.id for w in row_for_fragment(words[1], rows)], ["w1", "w2"]
        )
        self.assertEqual([w.id for w in row_for_fragment(words[2], rows)], ["w3"])
        orphan = _w("w9", "not-in-rows", 10, 500)
        self.assertEqual(row_for_fragment(orphan, rows), [])

    def test_row_text_visual_order_and_upper(self):
        words = [_w("w1", "closing", 400, 100), _w("w2", "Balance", 10, 100)]
        row = reconstruct_rows(words)[0]
        self.assertEqual(row_text(row), "Balance closing")
        self.assertEqual(row_text(row, upper=True), "BALANCE CLOSING")


if __name__ == "__main__":
    unittest.main()
