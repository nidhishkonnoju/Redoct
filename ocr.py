"""Tesseract OCR wrapper: image -> list of OcrWord with bounding boxes.

Coordinates are expressed in the *normalized* image space (see
normalize_image), which is the same space the redaction renderer draws in —
one coordinate space end to end.
"""
from __future__ import annotations

import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import pytesseract
from PIL import Image

from config import (
    OCR_MAX_SIDE_PX,
    OCR_MIN_CONF,
    OCR_MIN_SIDE_PX,
    OCR_TESSERACT_CONFIG,
    TESSERACT_CMD,
)

_COMMON_PATHS = (
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
    r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
)


@dataclass
class NormalizedImage:
    image: Image.Image
    scale: float      # normalized = original * scale
    upscaled: bool


@dataclass
class OcrWord:
    id: str
    text: str
    x: int
    y: int
    w: int
    h: int
    conf: float
    # Tesseract's own grouping. Kept for diagnostics only: the row-level rules
    # rebuild rows from geometry (see reconstruct_rows) because merged ledger
    # columns make these numbers unreliable. Default 0 for synthetic words.
    block_num: int = 0
    par_num: int = 0
    line_num: int = 0


def configure_tesseract() -> str:
    """Resolve the tesseract binary path and point pytesseract at it."""
    if TESSERACT_CMD:
        path = TESSERACT_CMD
    else:
        path = shutil.which("tesseract") or ""
        if not path:
            for candidate in _COMMON_PATHS:
                if os.path.exists(candidate):
                    path = candidate
                    break
    if not path or not os.path.exists(path):
        raise RuntimeError(
            "Tesseract OCR binary not found. Install it with:\n"
            "  winget install --id UB-Mannheim.TesseractOCR -e\n"
            "or set the REDACT_TESSERACT environment variable to tesseract.exe."
        )
    pytesseract.pytesseract.tesseract_cmd = path
    return path


def normalize_image(img: Image.Image) -> NormalizedImage:
    """Scale so OCR has enough pixels but coordinates stay bounded."""
    w, h = img.size
    min_side, max_side = min(w, h), max(w, h)
    scale = 1.0
    if min_side < OCR_MIN_SIDE_PX:
        scale = OCR_MIN_SIDE_PX / min_side
    if max_side * scale > OCR_MAX_SIDE_PX:
        scale = OCR_MAX_SIDE_PX / max_side
    if scale == 1.0:
        return NormalizedImage(image=img.copy(), scale=1.0, upscaled=False)
    new_size = (max(1, round(w * scale)), max(1, round(h * scale)))
    resized = img.resize(new_size, Image.LANCZOS)
    return NormalizedImage(image=resized, scale=scale, upscaled=scale > 1.0)


def reconstruct_rows(
    words: list[OcrWord], y_tolerance_ratio: float = 0.6
) -> list[list[OcrWord]]:
    """Group fragments into logical rows from geometry alone (Plan v3).

    Tesseract's own block/paragraph/line numbering merges or splits columns of a
    ledger row unpredictably, and every row-level rule must not inherit that
    mistake: a counterparty name whose date and amount landed in a different
    OCR line is exactly how a leak slipped through earlier. Rows are rebuilt
    from the bounding boxes instead — fragments in reading order top-to-bottom,
    a fragment joining the current row when its vertical center sits within
    `max(height) * y_tolerance_ratio` of the row's previous fragment. The ratio
    is generous on purpose because a label and its value often differ in font
    size on the same visual row.

    Each row is returned sorted left-to-right, so `row_text()` reads in visual
    order. Pure geometry: no OCR service, no font metrics.
    """
    if not words:
        return []
    ordered = sorted(words, key=lambda w: w.y + w.h / 2)
    rows: list[list[OcrWord]] = []
    current: list[OcrWord] = []
    for frag in ordered:
        if not current:
            current = [frag]
            continue
        ref = current[-1]
        tolerance = max(frag.h, ref.h) * y_tolerance_ratio
        if abs(_center(frag) - _center(ref)) <= tolerance:
            current.append(frag)
        else:
            rows.append(current)
            current = [frag]
    if current:
        rows.append(current)
    for row in rows:
        row.sort(key=lambda w: w.x)
    return rows


def _center(word: OcrWord) -> float:
    return word.y + word.h / 2


def row_for_fragment(frag: OcrWord, rows: list[list[OcrWord]]) -> list[OcrWord]:
    """Return the reconstructed row holding `frag` (matched by id), else []."""
    for row in rows:
        if any(w.id == frag.id for w in row):
            return row
    return []


def row_text(row: list[OcrWord], upper: bool = False) -> str:
    """Space-joined row text in visual order; `upper` for keyword matching."""
    text = " ".join(w.text for w in row)
    return text.upper() if upper else text


def group_lines(words: list[OcrWord]) -> list[list[OcrWord]]:
    """Group words into reading-order lines by vertical center proximity."""
    if not words:
        return []
    heights = sorted(w.h for w in words)
    median_h = heights[len(heights) // 2]
    tolerance = max(median_h / 2, 6)
    ordered = sorted(words, key=lambda w: (w.y + w.h / 2, w.x))
    lines: list[list[OcrWord]] = []
    for word in ordered:
        center = word.y + word.h / 2
        for line in lines:
            line_center = sum(w.y + w.h / 2 for w in line) / len(line)
            if abs(center - line_center) <= tolerance:
                line.append(word)
                break
        else:
            lines.append([word])
    for line in lines:
        line.sort(key=lambda w: w.x)
    return lines


def _int_field(data: dict, key: str, index: int) -> int:
    """Read a tesseract int column defensively (missing/short columns -> 0)."""
    try:
        return int(data[key][index])
    except (KeyError, TypeError, ValueError, IndexError):
        return 0


def extract_words(img: Image.Image) -> list[OcrWord]:
    """Run tesseract and return confident, non-blank words with boxes."""
    configure_tesseract()
    data = pytesseract.image_to_data(
        img, output_type=pytesseract.Output.DICT, config=OCR_TESSERACT_CONFIG
    )
    words: list[OcrWord] = []
    for i, text in enumerate(data["text"]):
        text = (text or "").strip()
        try:
            conf = float(data["conf"][i])
        except (TypeError, ValueError):
            conf = -1.0
        if not text or conf < OCR_MIN_CONF:
            continue
        n = len(words) + 1
        words.append(
            OcrWord(
                id=f"w{n:04d}",
                text=text,
                x=int(data["left"][i]),
                y=int(data["top"][i]),
                w=int(data["width"][i]),
                h=int(data["height"][i]),
                conf=conf,
                block_num=_int_field(data, "block_num", i),
                par_num=_int_field(data, "par_num", i),
                line_num=_int_field(data, "line_num", i),
            )
        )
    return words


if __name__ == "__main__":
    from config import SAMPLE_DOCS_DIR

    path = SAMPLE_DOCS_DIR / "bank_statement.png"
    print(f"Loading {path} ...")
    source = Image.open(path).convert("RGB")
    norm = normalize_image(source)
    if norm.upscaled:
        print(f"Upscaled x{norm.scale:.2f} -> {norm.image.size}")
    found = extract_words(norm.image)
    print(f"{len(found)} words:")
    for wd in found:
        print(f"{wd.id}  conf={wd.conf:5.1f}  ({wd.x},{wd.y},{wd.w},{wd.h})  {wd.text!r}")
