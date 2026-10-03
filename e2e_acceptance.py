"""End-to-end acceptance probe (Plan v3 Priorities 1 + 4).

Runs the REAL pipeline (Tesseract + Ollama, no stubs) over every synthetic
sample document and verifies the privacy contract the whole layered design
exists to guarantee — on the *rendered pixels*, not on the pipeline's own
claims:

1. **PII covered** — every fragment the regex / context net recognises as PII is
   behind a drawn box (full redact or partial mask), never readable. This check
   is label-free: it re-derives the PII set from the OCR fragments themselves,
   so it cannot be fooled by a mislabelled fragment.
2. **Privacy floor** — every covered fragment's box is opaque black in the
   output image. A partial fragment's box additionally carries exactly one
   preset-derived mask: the string is recomputed here from the preset's own
   `partial_types` spec via `policy.mask_value()`, must reveal the last four
   characters and must not contain the hidden ones.
3. **Purpose delivered** (reported, never gated) — the values the stated purpose
   exists to reveal are still readable. A 3B model can legitimately mislabel a
   field, so a *missing* purpose value is reported as ⚠ rather than failing the
   run; a *privacy* failure is the hard gate.

Usage:
    python e2e_acceptance.py                  # every case
    python e2e_acceptance.py marksheet voter  # only cases whose name matches
    python e2e_acceptance.py --list

Exit code: 0 when every case passes, 1 otherwise.
"""
from __future__ import annotations

import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

from PIL import Image

import llm
from policy import mask_value, partial_spec_map
from redact import (
    REDACT_PADDING_PX,
    load_presets,
    run_plan_v2,
    validate_context_fragments,
    validate_fragments,
)

ROOT = Path(__file__).resolve().parent
SAMPLES = ROOT / "sample_docs"

LIGHT_PIXEL_FLOOR = 150  # mask text is (235, 235, 235); boxes are (0, 0, 0)
MASK_BOX_MAX_INK = 0.5   # a mask never paints half the box (the box stays opaque)


@dataclass
class Case:
    """One (document, purpose) acceptance case."""

    document: str
    purpose: str
    hidden: tuple[str, ...] = ()   # literals that MUST be behind a box
    visible: tuple[str, ...] = ()  # literals the purpose wants readable
    note: str = ""

    @property
    def name(self) -> str:
        return f"{self.document} + {self.purpose}"


CASES: tuple[Case, ...] = (
    Case(
        "bank_statement", "proof_of_income",
        hidden=("5010023456789012", "HDFC0001234", "9876543210",
                "arjun.mehta@example.com"),
        visible=("ARJUN", "85,000.00"),
        note="account/IFSC/mobile/email hidden, name + salary shown (the branch "
             "address is the bank's, not the holder's, so it may stay)",
    ),
    Case(
        "pan_card", "id_verification",
        hidden=("14/08/1999", "RAKESH", "ABCDE1234F"),
        visible=("ARJUN",),
        note="PAN is partially masked (last 4), DOB + father hidden",
    ),
    Case(
        "salary_slip", "proof_of_income",
        hidden=("5010023456789012", "KA/BNG/204518/000", "9812345670",
                "AJMPM4471K"),
        visible=("ARJUN", "85,000"),
        note="bank a/c + PF + phone + PAN-in-remarks hidden",
    ),
    Case(
        "voter_id_card", "id_verification",
        hidden=("14/08/1999", "RAKESH", "ABC1234567", "Flat"),
        visible=("ARJUN",),
        note="EPIC number partially masked (last 4) — new layout, new spec",
    ),
    Case(
        "voter_id_card", "proof_of_address",
        hidden=("14/08/1999", "RAKESH", "ABC1234567"),
        visible=("Flat", "Bengaluru"),
        note="name + address shown, identifiers hidden/partial",
    ),
    Case(
        "marksheet", "education_proof",
        hidden=("2023CS1042", "GENERAL", "RAKESH", "14/08/1999", "9812345670"),
        visible=("BENGALURU", "8.72"),
        note="roll number/category/father/DOB hidden, CGPA shown",
    ),
)


def _matching(words, literal: str):
    needle = literal.lower()
    return [w for w in words if needle in w.text.lower()]


def _ink(img: Image.Image, word, pad: int = REDACT_PADDING_PX) -> tuple[int, int]:
    """Count (dark, light) pixels inside a fragment's padded box."""
    x0, y0 = max(word.x - pad, 0), max(word.y - pad, 0)
    x1 = min(word.x + word.w + pad, img.width)
    y1 = min(word.y + word.h + pad, img.height)
    dark = light = 0
    for x in range(x0, x1):
        for y in range(y0, y1):
            if min(img.getpixel((x, y))) > LIGHT_PIXEL_FLOOR:
                light += 1
            else:
                dark += 1
    return dark, light


def _check_mask(text_by_id: dict[str, str], preset: dict, result) -> list[str]:
    """Every printed mask must be derivable from the preset and bounded."""
    fails: list[str] = []
    specs = list(partial_spec_map(preset).values())
    for wid, mask in sorted(result.partial_masks.items()):
        raw = text_by_id.get(wid, "")
        allowed = {mask_value(raw, spec) for spec in specs}
        if mask not in allowed:
            fails.append(
                f"partial mask for {wid} ({mask!r}) is not a preset-derived "
                f"mask of {raw!r}"
            )
            continue
        flat_mask = re.sub(r"\s+", "", mask)
        flat_raw = re.sub(r"\s+", "", raw)
        if not flat_mask.endswith(flat_raw[-4:]):
            fails.append(
                f"partial mask for {wid} ({mask!r}) does not reveal the last 4 "
                f"characters of {raw!r}"
            )
        prefix = flat_raw[:-4]
        if prefix and prefix in flat_mask:
            fails.append(
                f"partial mask for {wid} ({mask!r}) still contains hidden "
                "characters of the raw value"
            )
    return fails


def _check_case(case: Case, presets: dict) -> tuple[list[str], list[str], dict]:
    """Return (hard failures, informational notes, stats) for one case."""
    fails: list[str] = []
    notes: list[str] = []
    path = SAMPLES / f"{case.document}.png"
    preset = presets[case.purpose]
    img = Image.open(path).convert("RGB")
    llm.LAST_LAYER_CALLS.clear()
    started = time.perf_counter()
    result = run_plan_v2(img, case.purpose, preset)
    elapsed = time.perf_counter() - started
    layers: dict[str, float] = {}
    for call in llm.LAST_LAYER_CALLS:
        layers[call["layer"]] = layers.get(call["layer"], 0.0) + call["total_s"]

    words = result.words
    text_by_id = {w.id: w.text for w in words}
    covered = (
        set(result.llm_redact_ids) | set(result.unfilled_ids) | set(result.partial_ids)
    )
    visible = set(result.visible_ids)

    # 1. Label-free PII invariant: anything the deterministic nets call PII must
    #    sit behind a box, whatever the model labelled it.
    pii_ids = {h.word_id for h in validate_fragments(words, set())}
    pii_ids |= {h.word_id for h in validate_context_fragments(words, set())}
    for wid in sorted(pii_ids - covered):
        fails.append(f"PII fragment left readable: {text_by_id.get(wid, wid)!r}")

    # 2. Pixel probe: the box is the privacy floor, so verify it was drawn.
    partial = set(result.partial_masks)
    for word in words:
        if word.id not in covered:
            continue
        dark, light = _ink(result.output_image, word)
        if word.id in partial:
            if not dark:
                fails.append(f"partial box not drawn for {word.text!r}")
            if not light:
                fails.append(
                    f"partial mask not printed for {word.text!r} "
                    f"({result.partial_masks[word.id]!r})"
                )
            elif light > MASK_BOX_MAX_INK * (dark + light):
                fails.append(f"partial box not opaque for {word.text!r}")
        elif light:
            fails.append(f"redacted box not opaque for {word.text!r}")

    # 3. The literals this case is about.
    for literal in case.hidden:
        matches = _matching(words, literal)
        if not matches:
            fails.append(f"stale expectation: {literal!r} is not in the OCR text")
            continue
        for word in matches:
            if word.id in visible:
                fails.append(f"expected-hidden value is readable: {literal!r}")
            elif word.id not in covered:
                fails.append(f"expected-hidden value is uncovered: {literal!r}")

    fails.extend(_check_mask(text_by_id, preset, result))

    for literal in case.visible:
        if any(w.id in visible for w in _matching(words, literal)):
            notes.append(f"OK  {literal!r} readable")
        else:
            notes.append(f"!!  {literal!r} not readable (purpose value hidden)")

    for warning in result.warnings:
        if "not covered by drawn boxes" in warning:
            fails.append(f"pipeline self-check: {warning}")

    stats = {
        "elapsed": elapsed,
        "layers": layers,
        "llm_s": round(sum(layers.values()), 1),
        "calls": len(llm.LAST_LAYER_CALLS),
        "document_type": result.document_type,
        "visible": len(visible),
        "masked": len(partial),
        "redacted": len(set(result.llm_redact_ids)),
        "pii": len(pii_ids),
        "masks": [result.partial_masks[i] for i in sorted(partial)],
        "warnings": result.warnings,
    }
    return fails, notes, stats


def main(argv: list[str]) -> int:
    if "--list" in argv:
        for case in CASES:
            print(f"{case.name}  ({case.note})")
        return 0
    timings = "--timings" in argv
    filters = [a for a in argv if not a.startswith("-")]
    cases = [
        c for c in CASES
        if not filters or any(f.lower() in c.name.lower() for f in filters)
    ]
    if not cases:
        print(f"no case matches {filters}")
        return 1

    presets = load_presets()
    failed = 0
    for case in cases:
        fails, notes, stats = _check_case(case, presets)
        print(
            f"[{'PASS' if not fails else 'FAIL'}] {case.name:<42} "
            f"{stats['elapsed']:6.1f}s  type={str(stats['document_type']):<16} "
            f"visible={stats['visible']:<3} masked={stats['masked']:<2} "
            f"redacted={stats['redacted']:<3} pii={stats['pii']}"
        )
        failed += bool(fails)
        if timings:
            breakdown = ", ".join(
                f"{layer} {seconds:.1f}s" for layer, seconds in sorted(stats["layers"].items())
            )
            print(
                f"        timing: llm {stats['llm_s']}s in {stats['calls']} call(s) "
                f"[{breakdown}]; ocr+render {stats['elapsed'] - stats['llm_s']:.1f}s"
            )
        if stats["masks"]:
            print(f"        masks: {', '.join(stats['masks'])}")
        for note in notes:
            print(f"        {note}")
        if case.note:
            print(f"        expected: {case.note}")
        for warning in stats["warnings"]:
            print(f"        ! {warning}")
        for fail in fails:
            print(f"        x {fail}")

    print(
        f"\n{len(cases) - failed}/{len(cases)} case(s) passed -> "
        f"{'PII visible=none' if not failed else 'SEE FAILURES ABOVE'}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
