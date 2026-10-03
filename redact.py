"""Redaction rendering + regex safety net + pipeline orchestration."""
from __future__ import annotations

import json
import hashlib
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from config import (
    MASK_FONT_PATH,
    MASK_FONT_RATIO,
    MASK_MIN_FONT_PX,
    MASK_TEXT_COLOR,
    MASK_TEXT_PAD_PX,
    PRESETS_PATH,
    REDACT_PADDING_PX,
)
import llm
from llm import LLMDecision, OllamaError, OllamaParseError
from ocr import (
    OcrWord,
    extract_words,
    group_lines,
    normalize_image,
    reconstruct_rows,
    row_for_fragment,
    row_text,
)
from prompts import DOB_ANCHORS
from policy import apply_policy, mask_value, partial_spec_map

# PRD section 6 safety net + two intentional extensions: long digit runs catch
# bank account numbers (no pattern-specific regex below would match), and email
# addresses are PII the PRD's list simply omitted.
VALIDATION_PATTERNS: dict[str, re.Pattern[str]] = {
    "aadhaar": re.compile(r"\b\d{4}\s?\d{4}\s?\d{4}\b"),
    "pan": re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"),
    "phone": re.compile(r"\b[6-9]\d{9}\b"),
    "ifsc": re.compile(r"\b[A-Z]{4}0[A-Z0-9]{6}\b"),
    "long_digits": re.compile(r"\b\d{10,19}\b"),
    "email": re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.]{2,}\b"),
}

# Context-anchored rules: a bare date is not sensitive (statement periods and
# transaction dates must stay), but the same token under a "Date of Birth"
# label is. Anchoring on the label is what makes the rule safe to apply.
# `DOB_ANCHORS` lives in `prompts.py` because the Detect prompt teaches the model
# exactly this label list — one definition shared by the prompt, this regex net
# and `anchor_date_labels()` means the three cannot drift apart.
_DATE_TOKEN_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
CONTEXT_RULES: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "date_of_birth": (DOB_ANCHORS, _DATE_TOKEN_RE),
}

# How far a neighbouring OCR line may sit from an anchored label line and still
# count as "the same visual region". Measured between line centers, scaled by the
# document's median fragment height so it holds for both a 300 dpi scan and a
# phone photo; the floor keeps tiny-font pages from collapsing the window.
CONTEXT_LINE_GAP_RATIO = 2.5
CONTEXT_LINE_MIN_GAP = 30

# --- Detect cache (Plan v3 Priority 5) ---------------------------------------
# Classify + Detect are purpose-agnostic by design (Plan v2), so re-running the
# SAME document under a different purpose has no reason to pay for them again:
# only the free policy lookup (and the regex net / render) depend on the purpose.
# The key is the normalized image itself, so a hit means byte-identical
# fragments and labels; the entries are tiny (a few hundred ids + type names).
DETECT_CACHE: dict[str, tuple[str, list[dict]]] = {}
DETECT_CACHE_MAX = 8


def _median_height(words: list[OcrWord]) -> float:
    heights = sorted(w.h for w in words)
    return float(heights[len(heights) // 2]) if heights else 0.0


def _line_center(line: list[OcrWord]) -> float:
    return sum(w.y + w.h / 2 for w in line) / len(line)


def _near_line(line: list[OcrWord], other: list[OcrWord], median_h: float) -> bool:
    """True when two OCR lines are close enough to share a label/value region."""
    if not line or not other:
        return False
    limit = max(CONTEXT_LINE_GAP_RATIO * median_h, CONTEXT_LINE_MIN_GAP)
    return abs(_line_center(line) - _line_center(other)) <= limit


@dataclass
class ValidationHit:
    word_id: str
    pattern: str
    matched_text: str


@dataclass
class PipelineResult:
    output_image: Image.Image
    document_type: str
    llm_redact_ids: list[str]
    validator_redact_ids: list[str]
    unfilled_ids: list[str]
    validator_hits: list[ValidationHit]
    reasoning: str
    elapsed_s: float
    warnings: list[str] = field(default_factory=list)
    # Plan v3 Priority 1: ids drawn as a black box with a masked reveal over it.
    partial_ids: list[str] = field(default_factory=list)
    # The exact strings printed over those boxes (never LLM output, never the
    # raw fragment) — surfaced so the UI and the acceptance probe can show/verify
    # what was revealed.
    partial_masks: dict[str, str] = field(default_factory=dict)
    # Ids left fully readable in the output (the audit's view). A partial id is
    # NOT visible: a reader sees a mask, not the fragment.
    visible_ids: list[str] = field(default_factory=list)
    # The OCR fragments this run was built from, for pixel-level verification.
    words: list[OcrWord] = field(default_factory=list)


def load_presets(path: Path | None = None) -> dict:
    """Read and sanity-check presets.json (plain function; app.py caches it)."""
    path = Path(path or PRESETS_PATH)
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"presets.json not found at {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"presets.json is not valid JSON: {exc}") from exc
    for key, preset in data.items():
        if "label" not in preset:
            raise RuntimeError(f"preset '{key}' missing 'label'")
        if not ({"keep", "redact"} <= set(preset) or {"keep_types", "redact_types"} <= set(preset)):
            raise RuntimeError(
                f"preset '{key}' needs keep/redact or keep_types/redact_types"
            )
        # Plan v3 Priority 1: a partial spec is the *only* thing that may print
        # text over a black box, so a malformed one must fail at load time rather
        # than silently degrade at render time.
        for ftype, spec in (preset.get("partial_types") or {}).items():
            if not isinstance(spec, dict) or "reveal" not in spec or "format" not in spec:
                raise RuntimeError(
                    f"preset '{key}' partial type '{ftype}' needs both 'reveal' "
                    "and 'format'"
                )
            kept = set(preset.get("keep_types") or preset.get("keep") or [])
            critical = set(preset.get("purpose_critical_types") or [])
            if ftype in kept or ftype in critical:
                raise RuntimeError(
                    f"preset '{key}' type '{ftype}' cannot be both kept (or "
                    "purpose-critical) and partially masked"
                )
    return data


def validate_fragments(
    ocr_words: list[OcrWord], candidate_redact_ids: set[str]
) -> list[ValidationHit]:
    """Scan every fragment NOT slated for redaction against PII patterns."""
    hits: list[ValidationHit] = []
    for word in ocr_words:
        if word.id in candidate_redact_ids:
            continue
        for pattern_name, pattern in VALIDATION_PATTERNS.items():
            match = pattern.search(word.text)
            if match:
                hits.append(
                    ValidationHit(
                        word_id=word.id,
                        pattern=pattern_name,
                        matched_text=match.group(0),
                    )
                )
    return hits


def validate_context_fragments(
    ocr_words: list[OcrWord], candidate_redact_ids: set[str]
) -> list[ValidationHit]:
    """Context-anchored rules: label and value must be in the same visual region.

    Layouts routinely put the label on one OCR line and its value on the next
    (larger font, lower baseline), so a label anchors its own line plus the
    lines immediately above and below in reading order. Including the line
    *above* is Plan v3's fail-closed widening: a right-aligned value can print
    before its label (``14/08/1999            Date of Birth``), and if the model
    then calls that date a ``transaction_date`` — which presets now KEEP — the
    only thing standing between a birth date and the output image would be this
    net. The cost is bounded and visible: one line of over-redaction next to an
    explicit birth-date label. The label itself keeps the rule safe — a bare
    date anywhere else in the document is never touched. Only neighbouring lines
    that are physically close count, so a birth date at the bottom of a page
    cannot blanket-redact the ledger above it.
    """
    hits: list[ValidationHit] = []
    lines = group_lines(ocr_words)
    median_h = _median_height(ocr_words)
    for index, line in enumerate(lines):
        text = " ".join(w.text for w in line).upper()
        window: list[OcrWord] = []
        if index > 0 and _near_line(line, lines[index - 1], median_h):
            window += lines[index - 1]
        window += line
        if index + 1 < len(lines) and _near_line(line, lines[index + 1], median_h):
            window += lines[index + 1]
        for field_name, (anchors, pattern) in CONTEXT_RULES.items():
            if not any(anchor in text for anchor in anchors):
                continue
            for word in window:
                if word.id in candidate_redact_ids:
                    continue
                match = pattern.search(word.text)
                if match:
                    hits.append(
                        ValidationHit(
                            word_id=word.id,
                            pattern=field_name,
                            matched_text=match.group(0),
                        )
                    )
    return hits


def apply_validator(
    ocr_words: list[OcrWord], decision: LLMDecision
) -> tuple[LLMDecision, list[ValidationHit], list[str]]:
    """Fail-closed id accounting + regex override (three outcomes).

    Returns (corrected_decision, validator_hits, unfilled_ids):
    - fragments the LLM never mentioned are forced to redact,
    - fragments the regex net flags while marked keep are forced to redact,
    - every fragment id ends up in exactly one of keep/partial/redact.

    Precedence inside a conflicting decision: REDACT > PARTIAL > KEEP.

    The *type* regex net (aadhaar/pan/phone/…) never touches a `partial` id:
    a partial fragment is deliberately covered by a box with a masked value on
    top, so its raw text is not what the reader sees, and its reveal is bounded
    by `policy.mask_value()`. The *context* net (label-anchored dates) stays
    authoritative over everything — a partially-masked date of birth is still a
    birth date, so that rule can downgrade partial to a plain full redaction.
    """
    valid = {w.id for w in ocr_words}
    llm_keep = [i for i in decision.keep_ids if i in valid]
    llm_redact = [i for i in decision.redact_ids if i in valid]
    llm_partial = [i for i in decision.partial_ids if i in valid]
    # A model may echo an id in more than one list (seen with qwen2.5).
    llm_partial = [i for i in llm_partial if i not in set(llm_redact)]
    llm_keep = [
        i for i in llm_keep if i not in set(llm_redact) and i not in set(llm_partial)
    ]
    unfilled = sorted(valid - set(llm_keep) - set(llm_redact) - set(llm_partial))

    # Type patterns: the exclusion set lists fragments already headed for a full
    # redaction, so what remains are the *kept* ones plus the partial ones. A
    # partial fragment is excluded here on purpose — it is pre-masked, and the
    # preset's own `partial_types` spec already bounds what may be printed, so a
    # type hit on its raw text is not evidence of a leak (otherwise no identifier
    # could ever be partially revealed).
    hits = validate_fragments(
        ocr_words, set(llm_redact) | set(unfilled) | set(llm_partial)
    )
    # Context patterns: authoritative over every outcome, partial included, so
    # only full redact + unlabelled fragments are excluded from this rule.
    hits += validate_context_fragments(ocr_words, set(llm_redact) | set(unfilled))
    # De-duplicate: one fragment may trip several patterns.
    unique: dict[str, ValidationHit] = {}
    for hit in hits:
        unique.setdefault(hit.word_id, hit)
    hits = list(unique.values())
    validator_redact = sorted({h.word_id for h in hits})

    corrected = LLMDecision(
        document_type=decision.document_type,
        redact_ids=sorted(set(llm_redact) | set(unfilled) | set(validator_redact)),
        keep_ids=[i for i in llm_keep if i not in set(validator_redact)],
        partial_ids=[i for i in llm_partial if i not in set(validator_redact)],
        reasoning=decision.reasoning,
    )
    return corrected, hits, unfilled


def _merge_line_boxes(
    words: list[OcrWord], padding: int
) -> list[tuple[int, int, int, int]]:
    """Merge words into padded rectangles; runs on one line become one box."""
    if not words:
        return []
    heights = sorted(w.h for w in words)
    median_h = heights[len(heights) // 2]
    sorted_words = sorted(words, key=lambda w: (w.y + w.h / 2, w.x))
    lines: list[list[OcrWord]] = []
    for word in sorted_words:
        center = word.y + word.h / 2
        for line in lines:
            line_center = sum(x.y + x.h / 2 for x in line) / len(line)
            if abs(center - line_center) <= max(median_h / 2, 6):
                line.append(word)
                break
        else:
            lines.append([word])

    boxes: list[tuple[int, int, int, int]] = []
    for line in lines:
        line.sort(key=lambda w: w.x)
        run: list[OcrWord] = []
        for word in line:
            if not run:
                run = [word]
                continue
            prev = run[-1]
            gap = word.x - (prev.x + prev.w)
            if gap <= max(prev.h, word.h):  # contiguous text, not a separate block
                run.append(word)
            else:
                boxes.append(_bounding_box(run, padding))
                run = [word]
        if run:
            boxes.append(_bounding_box(run, padding))
    return boxes


def _bounding_box(words: list[OcrWord], padding: int) -> tuple[int, int, int, int]:
    x0 = min(w.x for w in words) - padding
    y0 = min(w.y for w in words) - padding
    x1 = max(w.x + w.w for w in words) + padding
    y1 = max(w.y + w.h for w in words) + padding
    return x0, y0, x1, y1


def pii_word_ids(words: list[OcrWord]) -> set[str]:
    """Ids whose text matches a PII rule — never restored by any repair."""
    ids = {hit.word_id for hit in validate_fragments(words, set())}
    ids |= {hit.word_id for hit in validate_context_fragments(words, set())}
    return ids


# Anchors for the deterministic repair and for amount anchoring. These are the
# values the stated purpose exists to reveal; the LLM tends to blanket-redact
# them, or (worse) to call every money value a salary.
_SALARY_ANCHORS = (
    "SALARY CREDIT", "SALARY", "GROSS", "NET PAY", "NET SALARY", "BASIC",
    "EARNINGS", "CTC",
)
_NAME_ANCHORS = ("ACCOUNT HOLDER", "EMPLOYEE NAME", "CUSTOMER NAME", "ACCOUNT NAME")
# A line naming a third party is skipped entirely: their name must stay hidden.
# `anchor_third_party_names()` below uses the same list to demote a `name` the
# model put on a relative's value, so label layer and repair layer agree.
_NAME_ANCHOR_EXCLUDE = ("FATHER", "MOTHER", "SPOUSE", "GUARDIAN", "NOMINEE")
_AMOUNT_RE = re.compile(r"[₹$]?[\d,]+(\.\d+)?$")
_NAME_WORD_RE = re.compile(r"^[A-Za-z][A-Za-z.'-]*$")
# Ledger-row evidence for the name rule: a money value plus a date or a transfer
# keyword. The name value of a card/ID often sits on its own line (a PAN card
# prints "Name" and "ARJUN MEHTA" on separate rows), so the row's *content* —
# not the presence of a label word — decides what the name is.
_MONEY_RE = re.compile(r"^[₹$]?\d[\d,]*(\.\d{1,2})?$")
_DATE_RE = re.compile(r"^\d{1,2}[-/][A-Za-z0-9]{1,3}[-/]\d{2,4}$")
_TXN_ROW_HINTS = (
    "PURCHASE", "PAYMENT", "UPI/", "NEFT/", "IMPS/", "ATM/", "DEBIT", "CREDIT",
    "TXN", "WITHDRAWAL", "DEPOSIT",
)


def anchor_amount_labels(
    ocr_words: list[OcrWord],
    labels: list[dict],
    rows: list[list[OcrWord]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement between Layer 2 and Layer 3: anchor amounts.

    A small model reliably tags *every* money value on a statement as
    `salary_amount` (rent, card purchases, the closing balance). `salary_amount`
    is a KEEP type under "Proof of Income", so an unanchored mislabel would
    expose the whole spending pattern — exactly what the purpose must not show.

    An amount is accepted as `salary_amount` only when its own visual row
    carries a salary anchor ("SALARY CREDIT", "GROSS SALARY", …). Anything else
    is demoted to `other_amount`, which every preset redacts. Same anchor list
    the deterministic repair uses, so the label layer and the repair layer
    cannot disagree about which rows are salary rows. Rows come from
    `reconstruct_rows` (geometry, not Tesseract's line numbering), so a merged
    ledger column cannot smuggle an unanchored amount into a salary row.

    Returns (relabelled labels, demoted ids).
    """
    rows = rows if rows is not None else reconstruct_rows(ocr_words)
    anchored: set[str] = set()
    for row in rows:
        if any(anchor in row_text(row, upper=True) for anchor in _SALARY_ANCHORS):
            anchored.update(w.id for w in row)
    demoted: list[str] = []
    out: list[dict] = []
    for entry in labels:
        if entry.get("type") == "salary_amount" and entry.get("id") not in anchored:
            demoted.append(entry["id"])
            out.append({**entry, "type": "other_amount"})
        else:
            out.append(entry)
    return out, sorted(set(demoted))


def anchor_name_labels(
    ocr_words: list[OcrWord],
    labels: list[dict],
    rows: list[list[OcrWord]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement between Layer 2 and Layer 3: anchor names.

    The same model also calls **merchants** `name` ("Card Purchase BIGBASKET"),
    and `name` is a KEEP type for Proof of Income / Proof of Address / ID
    Verification — so an unanchored mislabel exposes where the account holder
    shops, which the purpose does not require.

    The discriminator is the row itself, never the presence of a label word: a
    `name` on a ledger row (a money value **and** either a date or a transfer
    keyword such as PURCHASE, NEFT/ or UPI/) is a counterparty, so it is demoted to
    `other` and every preset redacts it. A name on its own line — including a
    PAN card, whose "Name" label and value are printed on separate rows — keeps
    its `name` label and its purpose-critical protection.

    Returns (relabelled labels, demoted ids).
    """
    rows = rows if rows is not None else reconstruct_rows(ocr_words)
    ledger_ids: set[str] = set()
    for row in rows:
        text = row_text(row, upper=True)
        money = any(_MONEY_RE.match(w.text.strip()) for w in row)
        dated = any(_DATE_RE.match(w.text.strip()) for w in row)
        hinted = any(hint in text for hint in _TXN_ROW_HINTS)
        if money and (dated or hinted):
            ledger_ids.update(w.id for w in row)
    demoted: list[str] = []
    out: list[dict] = []
    for entry in labels:
        if entry.get("type") == "name" and entry.get("id") in ledger_ids:
            demoted.append(entry["id"])
            out.append({**entry, "type": "other"})
        else:
            out.append(entry)
    return out, sorted(set(demoted))


# Words that make up a label phrase itself ("Father's Name", "Address",
# "Photo"). They are never the value a label anchors, so rules that map a label
# to its value must skip them.
_LABEL_PHRASE_WORDS = frozenset(
    {"FATHER", "FATHERS", "MOTHER", "MOTHERS", "SPOUSE", "SPOUSES", "GUARDIAN",
     "GUARDIANS", "NOMINEE", "NAME", "RELATION", "S/O", "D/O", "W/O", "SON",
     "DAUGHTER", "WIFE", "HUSBAND", "PHOTO", "ADDRESS", "OF", "NO", "NUMBER",
     "CARD", "IDENTITY", "ELECTOR", "ELECTORS"}
)


# Field labels a document prints above (or beside) their value. The model often
# reads the whole block as `label_text` — a KEEP type in every preset — so an
# address would print; on the voter card it also calls the EPIC number
# `phone_number`, which redacts it but silently defeats the preset's own partial
# reveal. Anchoring on the literal label is what keeps the mapping safe: the
# label is what makes the value's field unambiguous, and it is the same evidence
# a human reader uses. Every mapped type is redact-or-partial in all four
# presets, so a mapping can never *reveal* a field a preset hides.
_VALUE_LABEL_TYPES: tuple[tuple[tuple[str, ...], str], ...] = (
    (("PERMANENT ACCOUNT NUMBER", "PAN NO", "PAN NUMBER"), "pan_number"),
    (("AADHAAR", "AADHAR", "UIDAI"), "aadhaar_number"),
    (("ELECTOR'S PHOTO IDENTITY CARD", "VOTER ID", "EPIC"), "voter_id_number"),
    (("ADDRESS",), "address"),
    (("CATEGORY", "CASTE"), "category"),
)
# A fragment is only re-typed when the model left it generic or put another
# identifier/text type on it. A confident non-identifier label — a name, a date
# of birth, a salary figure, a remarks block — is never overridden.
_VALUE_LABEL_RETYPEABLE = frozenset(
    {"label_text", "other", "account_number", "ifsc_code", "phone_number",
     "pf_number", "roll_number", "aadhaar_number", "pan_number",
     "voter_id_number"}
)


def _value_candidates(row: list[OcrWord], anchors: tuple[str, ...]) -> list[str]:
    """Ids of the fragments that follow a label phrase inside its own row.

    A multi-word label ("Permanent Account Number") must not have its own words
    read as the value, so everything up to and including the last word that
    belongs to the label phrase is skipped. A label that ends its row returns
    nothing, and the caller falls back to the row below it.
    """
    letters = [_label_word(anchor) for anchor in anchors]
    last_label = -1
    for index, word in enumerate(row):
        token = _label_word(word.text)
        if not token:
            continue
        if token in _LABEL_PHRASE_WORDS or any(token in anchor for anchor in letters):
            last_label = index
    return [w.id for w in row[last_label + 1:]]


def _overlaps_x(label_row: list[OcrWord], other_row: list[OcrWord]) -> bool:
    """True when a row prints in the same horizontal band as the label row.

    The photo placeholder on a voter card occupies a row of its own between the
    identifier label and the identifier value; without the band check it would be
    read as the label's value.
    """
    if not label_row or not other_row:
        return False
    a0 = min(w.x for w in label_row)
    a1 = max(w.x + w.w for w in label_row)
    b0 = min(w.x for w in other_row)
    b1 = max(w.x + w.w for w in other_row)
    return b0 < a1 and a0 < b1


def anchor_label_values(
    ocr_words: list[OcrWord],
    labels: list[dict],
    rows: list[list[OcrWord]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement: the value under a field label IS that field.

    The Detect prompt teaches the model this vocabulary, but on a layout it has
    not seen it splits the difference — it tags the address block `label_text`
    (printed, because `label_text` is a keep type everywhere) or the identifier
    `phone_number` (hidden, but then `id_verification`'s "show the last 4" never
    happens). The fix is the same evidence every other rule here uses: the
    document prints its own label. A fragment anchored by a known field label —
    in the label's own row, or in the row directly below it, which is how cards
    lay them out — is re-typed to that field.

    Deliberately narrow: only labels whose field type is unambiguous are mapped
    (`_VALUE_LABEL_TYPES`), and only fragments the model left generic or
    identifier-ish are re-typed (`_VALUE_LABEL_RETYPEABLE`), so a confident
    `name` / `date_of_birth` / `salary_amount` / `remarks` label always wins.

    Returns (relabelled labels, re-typed ids).
    """
    rows = rows if rows is not None else reconstruct_rows(ocr_words)
    type_by_id = {e["id"]: e.get("type") for e in labels}
    forced: dict[str, str] = {}
    for index, row in enumerate(rows):
        text = row_text(row, upper=True)
        for anchors, ftype in _VALUE_LABEL_TYPES:
            if not any(anchor in text for anchor in anchors):
                continue
            candidates = _value_candidates(row, anchors)
            if not candidates:
                # The value of a card label sits on the row directly below it,
                # in the same horizontal band and close enough to share a
                # label/value region — the same "visual region" test the context
                # net uses. A photo placeholder row (a different band) and a
                # distant header are both skipped.
                median_h = _median_height(ocr_words)
                for offset in (1, 2):
                    if index + offset >= len(rows):
                        break
                    below = rows[index + offset]
                    if not _near_line(row, below, median_h):
                        break
                    if not _overlaps_x(row, below):
                        continue
                    candidates = _value_candidates(below, anchors)
                    if candidates:
                        break
            for wid in candidates:
                forced.setdefault(wid, ftype)
            break
    retyped = {
        wid: ftype
        for wid, ftype in forced.items()
        if wid in type_by_id
        and type_by_id[wid] in _VALUE_LABEL_RETYPEABLE
        and type_by_id[wid] != ftype
    }
    if not retyped:
        return labels, []
    out = [
        {**entry, "type": retyped[entry["id"]]} if entry.get("id") in retyped else entry
        for entry in labels
    ]
    return out, sorted(retyped)


def anchor_third_party_names(
    ocr_words: list[OcrWord],
    labels: list[dict],
    rows: list[list[OcrWord]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement: a `name` under a third-party label is not the subject's.

    The model labels the *value* of "Father's Name" as `name` on both ID layouts
    (verified against the OCR dump of `pan_card.png` / `voter_id_card.png`), and
    `name` is a keep type — and a purpose-critical one — in every preset. Left
    alone, the relative's name is printed, and the audit guardrail actively
    *protects* it because it is labelled `name`. So the anchor settles it
    deterministically: any `name` fragment in a row anchored by FATHER / MOTHER /
    SPOUSE / GUARDIAN / NOMINEE is demoted to `father_name`, which every preset
    redacts.

    Two layouts are covered, matching the samples: label and value on the same
    row (marksheet-style, "Father's Name: RAKESH MEHTA") and label on its own row
    with the value on the next line (card-style, the PAN/voter layout). Only the
    label's own words are never reported. A name row with no third-party anchor —
    the subject's own name — is untouched.

    Returns (relabelled labels, demoted ids).
    """
    rows = rows if rows is not None else reconstruct_rows(ocr_words)
    name_ids = {e["id"] for e in labels if e.get("type") == "name"}
    if not name_ids:
        return labels, []
    demoted: set[str] = set()
    for index, row in enumerate(rows):
        if not any(bad in row_text(row, upper=True) for bad in _NAME_ANCHOR_EXCLUDE):
            continue
        own = [
            w.id for w in row
            if w.id in name_ids and _label_word(w.text) not in _LABEL_PHRASE_WORDS
        ]
        if not own and index + 1 < len(rows):
            # Card layout: the label sits on its own row, the value below it.
            own = [w.id for w in rows[index + 1] if w.id in name_ids]
        demoted.update(own)
    if not demoted:
        return labels, []
    out = [
        {**entry, "type": "father_name"} if entry.get("id") in demoted else entry
        for entry in labels
    ]
    return out, sorted(demoted)


def _label_word(text: str) -> str:
    """Normalize a fragment for label-phrase comparison ("Father's." -> FATHERS)."""
    return re.sub(r"[^A-Z]", "", text.upper())


def anchor_date_labels(
    ocr_words: list[OcrWord],
    labels: list[dict],
    rows: list[list[OcrWord]] | None = None,
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement: a ledger date is not a date of birth.

    `date_of_birth` is a redact type everywhere, so a mislabel here costs
    nothing; the danger runs the other way. Plan v3 adds `transaction_date` to
    every preset's `keep_types` so a ledger stays legible under
    "Proof of Income", and a 3B model hands out `transaction_date` freely. If a
    bare date in a transaction row were kept *and* labelled `date_of_birth`,
    the demo would look like a leak, so the DOB label is only honoured where
    the row actually declares one.

    A `date_of_birth` label is demoted to `transaction_date` only when the row
    has no DOB anchor *and* looks like a ledger row: an explicit date, or a
    transfer keyword, or a money value. Under a DOB anchor ("Date of Birth",
    "D.O.B", "जन्म") the label stands, and the `date_of_birth` context rule in
    `CONTEXT_RULES` force-hides that date even if a later layer disagrees.

    Returns (relabelled labels, demoted ids).
    """
    rows = rows if rows is not None else reconstruct_rows(ocr_words)
    ledger_ids: set[str] = set()
    for row in rows:
        text = row_text(row, upper=True)
        if any(anchor in text for anchor in DOB_ANCHORS):
            continue
        dated = any(_DATE_RE.match(w.text.strip()) for w in row)
        money = any(_MONEY_RE.match(w.text.strip()) for w in row)
        hinted = any(hint in text for hint in _TXN_ROW_HINTS)
        if dated or (money and hinted) or hinted:
            ledger_ids.update(w.id for w in row)
    demoted: list[str] = []
    out: list[dict] = []
    for entry in labels:
        if entry.get("type") == "date_of_birth" and entry.get("id") in ledger_ids:
            demoted.append(entry["id"])
            out.append({**entry, "type": "transaction_date"})
        else:
            out.append(entry)
    return out, sorted(set(demoted))


def repair_anchored_keeps(
    ocr_words: list[OcrWord],
    decision: LLMDecision,
    purpose_key: str,
    rows: list[list[OcrWord]] | None = None,
) -> tuple[LLMDecision, list[str]]:
    """Deterministic, auditable repair: re-keep purpose-critical labelled values.

    The LLM reliably over-redacts numeric fragments (every amount looks like an
    account number to it), which hides exactly the values a "Proof of Income"
    request exists to show. For rows explicitly anchored by a salary or
    account-holder label, re-keep their value fragments. Two hard guards:

    - ids whose text matches a PII pattern are NEVER restored — the regex net
      always wins, so no repair can ever undress a real identifier;
    - rows naming a third party (father/spouse/…) are skipped, so a relative's
      name cannot leak back into the output.

    Rows come from `reconstruct_rows` so the repair and the anchors share one
    notion of "row"; the `rows` parameter lets the pipeline share a single
    reconstruction across all four row rules instead of rebuilding it.
    """
    if purpose_key != "proof_of_income":
        return decision, []
    rows = rows if rows is not None else reconstruct_rows(ocr_words)

    protected = pii_word_ids(ocr_words)
    redacted = set(decision.redact_ids)
    repaired: list[str] = []
    for row in rows:
        text = row_text(row, upper=True)
        if any(bad in text for bad in _NAME_ANCHOR_EXCLUDE):
            continue
        salary_row = any(a in text for a in _SALARY_ANCHORS)
        name_row = any(a in text for a in _NAME_ANCHORS)
        if not (salary_row or name_row):
            continue
        for word in row:
            if word.id not in redacted or word.id in protected:
                continue
            if salary_row and _AMOUNT_RE.match(word.text):
                repaired.append(word.id)
            elif name_row and _NAME_WORD_RE.match(word.text) and len(word.text) > 1:
                repaired.append(word.id)
    if not repaired:
        return decision, []
    repaired = sorted(set(repaired))
    corrected = LLMDecision(
        document_type=decision.document_type,
        redact_ids=[i for i in decision.redact_ids if i not in set(repaired)],
        keep_ids=decision.keep_ids + repaired,
        partial_ids=decision.partial_ids,
        reasoning=decision.reasoning,
    )
    return corrected, repaired


def _mask_font(size: int):
    """Mask text font, falling back to PIL's bitmap font (same policy as OCR)."""
    try:
        return ImageFont.truetype(MASK_FONT_PATH, size) if MASK_FONT_PATH else \
            ImageFont.load_default()
    except OSError:
        return ImageFont.load_default()


def _fit_mask_text(text: str, box_w: int, box_h: int) -> int:
    """Largest font size whose single line of `text` fits inside the black box.

    The starting size is `MASK_FONT_RATIO` of the box height, and the search
    only ever shrinks from there. If even `MASK_MIN_FONT_PX` does not fit the
    answer is 0 — the caller then leaves the box blank, because a clipped reveal
    is both unreadable and a leak of more than the template promised.
    """
    usable_w = max(1, box_w - 2 * MASK_TEXT_PAD_PX)
    usable_h = max(1, box_h - 2 * MASK_TEXT_PAD_PX)
    lo = MASK_MIN_FONT_PX
    hi = max(MASK_MIN_FONT_PX, int(usable_h * MASK_FONT_RATIO))
    best = 0
    while lo <= hi:
        mid = (lo + hi) // 2
        font = _mask_font(mid)
        left, top, right, bottom = font.getbbox(text)
        if right - left <= usable_w and bottom - top <= usable_h:
            best = mid
            lo = mid + 1
        else:
            hi = mid - 1
    return best


def render_redaction(
    img: Image.Image,
    ocr_words: list[OcrWord],
    redact_ids: list[str],
    padding: int = REDACT_PADDING_PX,
    partial_masks: dict[str, str] | None = None,
) -> Image.Image:
    """Return a copy of `img` with solid black boxes over redact_ids.

    Plan v3 Priority 1 (three-way policy): a `partial` fragment is still
    covered by an opaque black box — the privacy floor is never skipped — and
    the pre-masked value from `policy.mask_value()` is printed *over* that box
    in light text (`MASK_TEXT_COLOR`), shrunk to fit. If the text cannot fit at
    any allowed size, or rendering fails for any reason, the box simply stays
    blank: information only ever flows through this paint path, so a failure
    can never expose the raw value.
    """
    out = img.copy()
    draw = ImageDraw.Draw(out)
    words_by_id = {w.id: w for w in ocr_words}
    masks = {i: t for i, t in (partial_masks or {}).items() if i in words_by_id}
    selected = [words_by_id[i] for i in set(redact_ids) if i in words_by_id]
    partial_words = [words_by_id[i] for i in masks]
    for x0, y0, x1, y1 in _merge_line_boxes(selected + partial_words, padding):
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
    for word_id, masked_text in masks.items():
        word = words_by_id[word_id]
        box = _bounding_box([word], padding)
        size = _fit_mask_text(masked_text, box[2] - box[0], box[3] - box[1])
        if size < MASK_MIN_FONT_PX:
            continue  # leave the box blank rather than clip the reveal
        try:
            font = _mask_font(size)
            left, top, right, bottom = font.getbbox(masked_text)
            draw.text(
                (
                    box[0] + (box[2] - box[0] - (right - left)) // 2 - left,
                    box[1] + (box[3] - box[1] - (bottom - top)) // 2 - top,
                ),
                masked_text,
                font=font,
                fill=MASK_TEXT_COLOR,
            )
        except Exception:
            continue  # fail closed to the bare black box drawn above
    return out


def _fail_closed_decision(words: list[OcrWord], note: str) -> LLMDecision:
    return LLMDecision(
        document_type="other",
        redact_ids=[w.id for w in words],
        keep_ids=[],
        reasoning=note,
    )


def audit_guardrail(
    flagged_ids: list[str],
    visible_ids: list[str],
    protected_ids: set[str],
) -> tuple[list[str], list[str]]:
    """Split an audit's flags into (actionable, blocked) — Plan v2 Layer 5b.

    The deterministic policy wins over the audit for `purpose_critical_types`:
    those are the fragments the stated purpose exists to reveal, so a fresh-
    context LLM must not be able to hide them again (an earlier build let the
    audit re-redact the holder's own name). Flags on fragments that are not
    currently visible are dropped as hallucinated ids.
    """
    visible = set(visible_ids)
    actionable: list[str] = []
    blocked: list[str] = []
    for fid in dict.fromkeys(flagged_ids):
        if fid not in visible:
            continue
        (blocked if fid in protected_ids else actionable).append(fid)
    return actionable, blocked


def build_partial_masks(
    ocr_words: list[OcrWord],
    labels: list[dict],
    partial_ids: list[str],
    preset: dict,
) -> tuple[dict[str, str], list[str]]:
    """Resolve partial ids into the exact strings that may be printed.

    Returns ({id: masked text}, ids that could not be masked). The masked text
    always comes from `policy.mask_value()` — a pure template substitution over
    the preset's own format — never from the LLM and never the raw fragment.

    Any id whose spec or value cannot honour the reveal (too short, wrong
    template, unknown reveal) is returned in the second list so the caller can
    fail *closed* to a plain full redaction instead of printing something.
    """
    specs = partial_spec_map(preset)
    types_by_id = {e.get("id"): e.get("type") for e in labels}
    text_by_id = {w.id: w.text for w in ocr_words}
    masks: dict[str, str] = {}
    unmaskable: list[str] = []
    for pid in partial_ids:
        text = text_by_id.get(pid)
        spec = specs.get(types_by_id.get(pid) or "")
        masked = mask_value(text, spec) if text is not None and spec else None
        if masked:
            masks[pid] = masked
        else:
            unmaskable.append(pid)
    return masks, sorted(set(unmaskable))


def _detect_cache_key(image: Image.Image) -> str:
    """Cache key for Layers 1+2: the exact normalized pixels OCR saw."""
    digest = hashlib.sha256(image.tobytes()).hexdigest()
    return f"{image.width}x{image.height}:{digest}"


def _cache_detect(cache_key: str, document_type: str, labels: list[dict]) -> None:
    """Store a raw Detect result, evicting the oldest entry when full.

    Only the *raw* labels are cached: the deterministic anchors (Layer 2b) run on
    every request, so a cache hit still reports the same corrections and still
    applies them for whatever preset is asked for.
    """
    DETECT_CACHE[cache_key] = (document_type, [dict(entry) for entry in labels])
    while len(DETECT_CACHE) > DETECT_CACHE_MAX:
        DETECT_CACHE.pop(next(iter(DETECT_CACHE)))


def clear_detect_cache() -> None:
    """Drop cached Detect results (the UI's "different purpose" path keeps them)."""
    DETECT_CACHE.clear()


def run_plan_v2(
    img: Image.Image, purpose_key: str, preset: dict, max_audit_rounds: int = 2
) -> PipelineResult:
    """Plan v2 pipeline: OCR -> classify -> detect -> policy -> regex net ->
    LLM audit (<=max_audit_rounds rounds) -> repairs -> render. Any LLM step
    raising OllamaError/OllamaParseError is skipped and the pipeline keeps
    going with fail-closed redaction for that step."""
    warnings: list[str] = []
    t0 = time.perf_counter()
    audit_rounds = 0

    norm = normalize_image(img.convert("RGB"))
    words = extract_words(norm.image)
    if not words:
        raise RuntimeError(
            "No text detected in the image. Try a sharper, well-lit photo."
        )

    # Layers 1+2 are purpose-agnostic, so the same document under a different
    # purpose reuses them (Priority 5): the cache key is the normalized image, so
    # a hit guarantees the same fragments and the same labels were produced.
    rows = reconstruct_rows(words)
    cache_key = _detect_cache_key(norm.image)
    cached = DETECT_CACHE.get(cache_key)
    if cached is not None:
        document_type, labels = cached[0], [dict(entry) for entry in cached[1]]
        warnings.append(
            "Classify + Detect reused the cached result for this document (0 LLM "
            "calls): same image, same fragments."
        )
    else:
        # Layer 1: fresh classify call.
        try:
            document_type = llm.classify(words)
        except (OllamaError, OllamaParseError) as exc:
            warnings.append(f"Classify step skipped ({exc}); type unknown.")
            document_type = "other"

        # Layer 2: fresh, purpose-agnostic detect call (batched: short ids stay
        # exact, each batch fits comfortably inside the model's output budget).
        # Plan v3: the Layer-1 document type selects a preservation note for the
        # prompt (a statement's ledger dates are not dates of birth).
        try:
            labels = llm.detect_batched(words, document_type=document_type)
            if llm.LAST_DETECT_FAILED_IDS:
                warnings.append(
                    f"{len(llm.LAST_DETECT_FAILED_IDS)} fragment(s) had unparseable "
                    "label output; fail-closed to REDACT."
                )
        except (OllamaError, OllamaParseError) as exc:
            warnings.append(
                f"Detect step failed ({exc}); failing closed: everything redacted."
            )
            labels = []
        _cache_detect(cache_key, document_type, labels)

    # Layer 2b: deterministic label anchors. Pure geometry, preset-independent,
    # free — so they run on every request, including a cache hit, and they are
    # what the cached raw labels get corrected with.
    try:
        labels, retyped = anchor_label_values(words, labels, rows)
        if retyped:
            warnings.append(
                f"{len(retyped)} value(s) the model left as generic text were "
                "re-typed from their own printed label (deterministic rule)."
            )
        labels, unanchored_amounts = anchor_amount_labels(words, labels, rows)
        if unanchored_amounts:
            warnings.append(
                f"{len(unanchored_amounts)} money value(s) without a salary "
                "anchor were demoted to other_amount (deterministic rule)."
            )
        labels, counterparties = anchor_name_labels(words, labels, rows)
        if counterparties:
            warnings.append(
                f"{len(counterparties)} ledger-row name(s) were treated as "
                "counterparties and redacted (deterministic rule)."
            )
        labels, third_parties = anchor_third_party_names(words, labels, rows)
        if third_parties:
            warnings.append(
                f"{len(third_parties)} third-party name(s) (father/mother/spouse/"
                "guardian) the model called `name` were redacted (deterministic "
                "rule)."
            )
        labels, ledger_dates = anchor_date_labels(words, labels, rows)
        if ledger_dates:
            warnings.append(
                f"{len(ledger_dates)} ledger date(s) were kept as "
                "transaction_date, not treated as dates of birth (deterministic "
                "rule)."
            )
    except (OllamaError, OllamaParseError) as exc:  # anchors make no LLM calls
        warnings.append(f"Label anchoring skipped ({exc}).")

    labeled_ids = {entry["id"] for entry in labels}
    unlabeled = sorted({w.id for w in words} - labeled_ids)

    # Layer 3: deterministic policy lookup — keep / partial / redact.
    split = apply_policy(labels, preset)
    unlabeled_set = set(unlabeled)
    partial_masks, unmaskable = build_partial_masks(
        words, labels, split["partial_ids"], preset
    )
    if partial_masks:
        warnings.append(f"{len(partial_masks)} fragment(s) partially masked.")
    if unmaskable:
        warnings.append(
            f"{len(unmaskable)} partial fragment(s) could not be masked under the "
            "preset's spec; fully redacted instead."
        )
    llm_redact_ids = sorted(set(split["redact_ids"]) | unlabeled_set | set(unmaskable))
    keep_ids = [
        i for i in split["keep_ids"]
        if i not in unlabeled_set and i not in partial_masks
    ]

    # Layer 5a: regex net over the kept fragments.
    decision = LLMDecision(
        document_type=document_type,
        redact_ids=llm_redact_ids,
        keep_ids=keep_ids,
        partial_ids=sorted(partial_masks),
        reasoning="plan-v2 layered pipeline",
    )
    corrected, hits, unfilled = apply_validator(words, decision)
    # The context net can downgrade a partial fragment to a full redaction; the
    # printable set must follow the corrected decision, never the old one.
    partial_masks = {
        k: v for k, v in partial_masks.items() if k in set(corrected.partial_ids)
    }
    if unfilled:
        warnings.append(
            f"{len(unfilled)} fragment(s) not classified by the LLM were "
            "fail-closed to REDACT."
        )
    validator_redact = sorted({h.word_id for h in hits} - set(decision.redact_ids))

    # Layer 5b: fresh-context LLM audit — sees ONLY the visible fragments.
    # Guardrail: the deterministic policy stays authoritative for
    # `purpose_critical_types` (the fields the purpose exists to reveal), so the
    # audit can tighten the leftovers but can never re-hide the holder's name or
    # the salary figure it was asked to prove.
    audit_hits: list[ValidationHit] = []
    blocked_flags: list[str] = []
    audit_calls = 0
    # A partial fragment is a black box to a reader; treat it as not visible so
    # the audit input and the reported visible set agree.
    keep_set = set(corrected.keep_ids) - set(partial_masks)
    protected_types = set(preset.get("purpose_critical_types", []))
    protected_ids = {
        entry["id"]
        for entry in labels
        if entry.get("type") in protected_types and entry["id"] in set(corrected.keep_ids)
    }
    for _ in range(max_audit_rounds):
        # Audit input = fully visible fragments ONLY. A partial fragment is a
        # black box to a reader; feeding its raw OCR text here would let the
        # audit "discover" a value that is not in the output image, and would
        # invite the model to re-hide something already bounded by the preset.
        keep_set = set(corrected.keep_ids) - set(partial_masks)
        visible = [w for w in words if w.id in keep_set]
        if not visible:
            break
        audit_calls += 1
        try:
            flagged, reasoning = llm.validate_llm(
                visible, [w.id for w in visible], preset.get("label", purpose_key)
            )
        except (OllamaError, OllamaParseError) as exc:
            warnings.append(f"LLM audit skipped ({exc}).")
            break
        fresh, blocked = audit_guardrail(flagged, [w.id for w in visible], protected_ids)
        blocked_flags.extend(blocked)
        if not fresh:
            break
        audit_rounds += 1
        for fid in fresh:
            audit_hits.append(
                ValidationHit(word_id=fid, pattern="llm_audit", matched_text=reasoning[:80])
            )
        corrected.keep_ids = [i for i in corrected.keep_ids if i not in set(fresh)]
        corrected.redact_ids = sorted(set(corrected.redact_ids) | set(fresh))
    if audit_rounds:
        warnings.append(f"LLM audit re-redacted {len(audit_hits)} fragment(s).")
    if blocked_flags:
        warnings.append(
            f"Policy guardrail overrode {len(blocked_flags)} audit flag(s) on "
            "purpose-critical field(s)."
        )
    hits = hits + audit_hits
    warnings.append(f"LLM audit ran {audit_calls} round(s).")

    if purpose_key == "proof_of_income":
        corrected, repaired = repair_anchored_keeps(words, corrected, purpose_key, rows)
        if repaired:
            warnings.append(
                f"Repaired {len(repaired)} purpose-critical value(s) to stay "
                "visible (label-anchored rule)."
            )

    # Plan v1 compatibility: app.py calls run_pipeline().
    output = render_redaction(
        norm.image, words, corrected.redact_ids, partial_masks=partial_masks
    )
    elapsed = time.perf_counter() - t0

    # Verify every covered word — full redaction *and* partial mask — is really
    # behind a drawn box: the black box is the privacy floor, so a partial
    # fragment that failed to get one would be a raw-value leak.
    covered_ids = set(corrected.redact_ids) | set(partial_masks)
    redacted_words = [w for w in words if w.id in covered_ids]
    boxes = _merge_line_boxes(redacted_words, REDACT_PADDING_PX)
    uncovered = [
        w.id for w in redacted_words
        if not any(
            b[0] <= w.x and b[1] <= w.y and b[2] >= w.x + w.w and b[3] >= w.y + w.h
            for b in boxes
        )
    ]
    if uncovered:
        warnings.append(
            f"{len(uncovered)} redacted word(s) not covered by drawn boxes."
        )

    # The repair can re-keep values an earlier pass had removed, so the reported
    # visible set is recomputed from the FINAL decision — it must always match
    # what the render actually left readable.
    return PipelineResult(
        output_image=output,
        document_type=corrected.document_type,
        llm_redact_ids=corrected.redact_ids,
        validator_redact_ids=validator_redact,
        unfilled_ids=unfilled,
        validator_hits=hits,
        reasoning=corrected.reasoning,
        elapsed_s=elapsed,
        warnings=warnings,
        partial_ids=sorted(partial_masks),
        partial_masks=partial_masks,
        visible_ids=sorted(set(corrected.keep_ids) - set(partial_masks)),
        words=words,
    )


def run_pipeline(
    img: Image.Image, purpose_key: str, preset: dict
) -> PipelineResult:
    """Plan v1 entry point — kept so app.py and existing callers work unchanged.

    Delegates to run_plan_v2, which supersedes the single combined LLM call
    with the layered pipeline (classify → detect → policy → regex net →
    LLM audit → repairs → render).
    """
    return run_plan_v2(img, purpose_key, preset)


if __name__ == "__main__":
    # Metrics probe: per-layer timings of the layered (v2) pipeline.
    import llm as _llm
    from ocr import extract_words, normalize_image
    from PIL import Image

    from redact import load_presets, run_pipeline

    img = Image.open("sample_docs/bank_statement.png").convert("RGB")
    presets = load_presets()
    result = run_pipeline(img, "proof_of_income", presets["proof_of_income"])
    for call in _llm.LAST_LAYER_CALLS:
        layer = call["system_preview"][4:40].split("(")[0].strip()
        print(f"{call['total_s']:6.2f}s  {layer}")
    print(f"total {result.elapsed_s:.1f}s  type={result.document_type}  "
          f"redacted={len(result.llm_redact_ids)}")
    for warning in result.warnings:
        print("  !", warning)
