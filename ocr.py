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
