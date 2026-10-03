"""Deterministic purpose-policy lookup (Plan v2, Layer 3).

The LLM's Detect layer only ever labels what each fragment IS — a name, an
account number, a salary figure. Whether a fragment stays visible is decided
here, with zero LLM calls: a plain membership test against the preset's
keep/partial/redact type lists.

Fail-closed by construction: any fragment whose type is unknown, missing, or
simply not listed under keep_types lands in redact_ids. The ONLY way a
fragment stays visible is explicit keep_types membership.

Plan v3 (Priority 1) adds a third outcome, `partial`, between keep and
redact: the fragment is still covered by a black box (the privacy floor, never
skipped) and a *template-masked* value is printed over it. Everything that
decides *which* outcome applies, and *what* string may be printed, is pure
logic in this module so it ports to Android unchanged (README, "Porting to
Android").
"""
from __future__ import annotations

import re

# A partial reveal must hide at least this many characters of the original
# value. A spec that would print a value almost verbatim (short number, wrong
# template) returns None from `mask_value()` and the caller falls back to a
# plain full redaction — the demo's "last 4" promise never turns into "all".
PARTIAL_MIN_HIDDEN = 4

_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z0-9_]+)\}")
_REVEAL_RE = re.compile(r"^(first|last)(\d+)$")


def partial_spec_map(preset: dict) -> dict[str, dict]:
    """`partial_types` as {field type: spec}, ignoring malformed entries."""
    raw = preset.get("partial_types") or {}
    if not isinstance(raw, dict):
        return {}
    return {
        str(key).strip(): spec
        for key, spec in raw.items()
        if str(key).strip() and isinstance(spec, dict)
    }


def apply_policy(labeled_fragments: list[dict], preset: dict) -> dict:
    """Split labeled fragments into redact vs keep vs partial ids.

    labeled_fragments: [{"id": <full ocr id>, "type": <field type>}, ...]
    preset: {label, keep_types, redact_types, partial_types?}

    Returns {"redact_ids": [...], "keep_ids": [...], "partial_ids": [...]} —
    order-stable, de-duplicated, and no id ever appears twice.

    Precedence, most revealing first:
      1. `partial_types`  — a masked, partial reveal (beats redact: a fragment
         the preset chose to partially show is masked, not hidden);
      2. `keep_types`     — shown in full, unless also in redact_types
         (redact beats keep on conflict);
      3. everything else  — redact, including unknown and missing types.

    Fragments with no label entry are NOT handled here (the caller fail-closes
    them via the valid-id set) — this function is a pure mapping.
    """
    keep_types = set(preset.get("keep_types", []))
    redact_types = set(preset.get("redact_types", []))
    partial_types = set(partial_spec_map(preset))
    redact_ids: list[str] = []
    keep_ids: list[str] = []
    partial_ids: list[str] = []
    seen: set[str] = set()
    for frag in labeled_fragments:
        fid = frag.get("id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        ftype = frag.get("type")
        if ftype in partial_types:
            partial_ids.append(fid)
        elif ftype in keep_types and ftype not in redact_types:
            keep_ids.append(fid)
        else:
            redact_ids.append(fid)
    return {
        "redact_ids": redact_ids,
        "keep_ids": keep_ids,
        "partial_ids": partial_ids,
    }


def mask_value(raw_text: str, spec: dict) -> str | None:
    """Apply a `partial_types` spec to a fragment's OCR text.

    Returns the string to print over the black box, or None when the spec or
    the value cannot honour the reveal safely. None means *fail closed*: the
    caller draws a plain full redaction instead of printing anything.

    Rejected (return None), deliberately:
      - a missing/unknown `reveal` ("last4"/"first2"), an empty value, or a
        value too short to hide `PARTIAL_MIN_HIDDEN` characters;
      - a `format` without exactly one placeholder, or with the *wrong*
        placeholder (`{first2}` on a `last4` spec would print the wrong end);
      - a `format` that still contains a fragment of the real value — the
        template is a mask shape (X's), never data.
    """
    if not isinstance(spec, dict):
        return None
    text = str(raw_text or "").strip()
    reveal = str(spec.get("reveal", "")).strip().lower()
    match = _REVEAL_RE.match(reveal)
    if not text or not match:
        return None
    side, count = match.group(1), int(match.group(2))
    if count <= 0 or len(text) <= count + PARTIAL_MIN_HIDDEN:
        return None
    revealed = text[:count] if side == "first" else text[-count:]
    template = spec.get("format")
    if not isinstance(template, str):
        return None
    placeholders = _PLACEHOLDER_RE.findall(template)
    if len(placeholders) != 1 or placeholders[0].lower() != f"{side}{count}":
        return None
    # Leftover letters/digits in the template would be printed verbatim: a
    # template may only be a mask shape plus its one placeholder.
    residue = _PLACEHOLDER_RE.sub("X", template)
    if re.search(r"[A-Za-z0-9]", residue.replace("X", "").replace("x", "")):
        return None
    return template.replace("{" + placeholders[0] + "}", revealed)

