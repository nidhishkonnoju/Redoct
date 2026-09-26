"""Deterministic purpose-policy lookup (Plan v2, Layer 3).

The LLM's Detect layer only ever labels what each fragment IS — a name, an
account number, a salary figure. Whether a fragment stays visible is decided
here, with zero LLM calls: a plain membership test against the preset's
keep/redact type lists.

Fail-closed by construction: any fragment whose type is unknown, missing, or
simply not listed under keep_types lands in redact_ids. The ONLY way a
fragment stays visible is explicit keep_types membership.
"""
from __future__ import annotations


def apply_policy(labeled_fragments: list[dict], preset: dict) -> dict:
    """Split labeled fragments into redact vs keep ids.

    labeled_fragments: [{"id": <full ocr id>, "type": <field type>}, ...]
    preset: {"label": ..., "keep_types": [...], "redact_types": [...]}

    Returns {"redact_ids": [...], "keep_ids": [...]} — order-stable,
    de-duplicated. A type listed in redact_types always loses to keep_types.
    Fragments with no label entry are NOT handled here (the caller fail-closes
    them via the valid-id set) — this function is a pure mapping.
    """
    keep_types = set(preset.get("keep_types", []))
    redact_types = set(preset.get("redact_types", []))
    redact_ids: list[str] = []
    keep_ids: list[str] = []
    seen: set[str] = set()
    for frag in labeled_fragments:
        fid = frag.get("id")
        if not fid or fid in seen:
            continue
        seen.add(fid)
        ftype = frag.get("type")
        if ftype in keep_types and ftype not in redact_types:
            keep_ids.append(fid)
        else:
            redact_ids.append(fid)
    return {"redact_ids": redact_ids, "keep_ids": keep_ids}
