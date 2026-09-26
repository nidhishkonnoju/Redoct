"""Redaction rendering + regex safety net + pipeline orchestration."""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from PIL import Image, ImageDraw

from config import PRESETS_PATH, REDACT_PADDING_PX
import llm
from llm import LLMDecision, OllamaError, OllamaParseError
from ocr import OcrWord, extract_words, group_lines, normalize_image

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
_DATE_TOKEN_RE = re.compile(r"\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b")
CONTEXT_RULES: dict[str, tuple[tuple[str, ...], re.Pattern[str]]] = {
    "date_of_birth": (("DATE OF BIRTH", "D.O.B", "DOB", "BIRTH DATE"), _DATE_TOKEN_RE),
}


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
    (larger font, lower baseline), so a label anchors both its own line and the
    following line in reading order. The label itself keeps the rule safe: a
    bare date elsewhere in the document is never touched.
    """
    hits: list[ValidationHit] = []
    lines = group_lines(ocr_words)
    for index, line in enumerate(lines):
        text = " ".join(w.text for w in line).upper()
        window = line + (lines[index + 1] if index + 1 < len(lines) else [])
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
    """Fail-closed id accounting + regex override.

    Returns (corrected_decision, validator_hits, unfilled_ids):
    - fragments the LLM never mentioned are forced to redact,
    - fragments the regex net flags while marked keep are forced to redact,
    - every fragment id ends up in exactly one of keep/redact.
    """
    valid = {w.id for w in ocr_words}
    llm_keep = [i for i in decision.keep_ids if i in valid]
    llm_redact = [i for i in decision.redact_ids if i in valid]
    # A model may echo an id in both lists (seen with qwen2.5); REDACT wins.
    llm_keep = [i for i in llm_keep if i not in set(llm_redact)]
    unfilled = sorted(valid - set(llm_keep) - set(llm_redact))

    hits = validate_fragments(ocr_words, set(llm_redact) | set(unfilled))
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
    ocr_words: list[OcrWord], labels: list[dict]
) -> tuple[list[dict], list[str]]:
    """Deterministic refinement between Layer 2 and Layer 3: anchor amounts.

    A small model reliably tags *every* money value on a statement as
    `salary_amount` (rent, card purchases, the closing balance). `salary_amount`
    is a KEEP type under "Proof of Income", so an unanchored mislabel would
    expose the whole spending pattern — exactly what the purpose must not show.

    An amount is accepted as `salary_amount` only when its own visual line
    carries a salary anchor ("SALARY CREDIT", "GROSS SALARY", …). Anything else
    is demoted to `other_amount`, which every preset redacts. Same anchor list
    the deterministic repair uses, so the label layer and the repair layer
    cannot disagree about which lines are salary lines.

    Returns (relabelled labels, demoted ids).
    """
    anchored: set[str] = set()
    for line in group_lines(ocr_words):
        text = " ".join(w.text for w in line).upper()
        if any(anchor in text for anchor in _SALARY_ANCHORS):
            anchored.update(w.id for w in line)
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
    ocr_words: list[OcrWord], labels: list[dict]
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
    rows: set[str] = set()
    for line in group_lines(ocr_words):
        text = " ".join(w.text for w in line).upper()
        money = any(_MONEY_RE.match(w.text.strip()) for w in line)
        dated = any(_DATE_RE.match(w.text.strip()) for w in line)
        hinted = any(hint in text for hint in _TXN_ROW_HINTS)
        if money and (dated or hinted):
            rows.update(w.id for w in line)
    demoted: list[str] = []
    out: list[dict] = []
    for entry in labels:
        if entry.get("type") == "name" and entry.get("id") in rows:
            demoted.append(entry["id"])
            out.append({**entry, "type": "other"})
        else:
            out.append(entry)
    return out, sorted(set(demoted))


def repair_anchored_keeps(
    ocr_words: list[OcrWord], decision: LLMDecision, purpose_key: str
) -> tuple[LLMDecision, list[str]]:
    """Deterministic, auditable repair: re-keep purpose-critical labelled values.

    The LLM reliably over-redacts numeric fragments (every amount looks like an
    account number to it), which hides exactly the values a "Proof of Income"
    request exists to show. For lines explicitly anchored by a salary or
    account-holder label, re-keep their value fragments. Two hard guards:

    - ids whose text matches a PII pattern are NEVER restored — the regex net
      always wins, so no repair can ever undress a real identifier;
    - lines naming a third party (father/spouse/…) are skipped, so a relative's
      name cannot leak back into the output.
    """
    if purpose_key != "proof_of_income":
        return decision, []
    from ocr import group_lines

    protected = pii_word_ids(ocr_words)
    redacted = set(decision.redact_ids)
    repaired: list[str] = []
    for line in group_lines(ocr_words):
        text = " ".join(w.text for w in line).upper()
        if any(bad in text for bad in _NAME_ANCHOR_EXCLUDE):
            continue
        salary_line = any(a in text for a in _SALARY_ANCHORS)
        name_line = any(a in text for a in _NAME_ANCHORS)
        if not (salary_line or name_line):
            continue
        for word in line:
            if word.id not in redacted or word.id in protected:
                continue
            if salary_line and _AMOUNT_RE.match(word.text):
                repaired.append(word.id)
            elif name_line and _NAME_WORD_RE.match(word.text) and len(word.text) > 1:
                repaired.append(word.id)
    if not repaired:
        return decision, []
    repaired = sorted(set(repaired))
    corrected = LLMDecision(
        document_type=decision.document_type,
        redact_ids=[i for i in decision.redact_ids if i not in set(repaired)],
        keep_ids=decision.keep_ids + repaired,
        reasoning=decision.reasoning,
    )
    return corrected, repaired


def render_redaction(
    img: Image.Image,
    ocr_words: list[OcrWord],
    redact_ids: list[str],
    padding: int = REDACT_PADDING_PX,
) -> Image.Image:
    """Return a copy of `img` with solid black boxes over redact_ids."""
    out = img.copy()
    draw = ImageDraw.Draw(out)
    selected = {w.id: w for w in ocr_words if w.id in set(redact_ids)}
    for x0, y0, x1, y1 in _merge_line_boxes(list(selected.values()), padding):
        draw.rectangle((x0, y0, x1, y1), fill=(0, 0, 0))
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


def run_plan_v2(
    img: Image.Image, purpose_key: str, preset: dict, max_audit_rounds: int = 2
) -> PipelineResult:
    """Plan v2 pipeline: OCR -> classify -> detect -> policy -> regex net ->
    LLM audit (<=max_audit_rounds rounds) -> repairs -> render. Any LLM step
    raising OllamaError/OllamaParseError is skipped and the pipeline keeps
    going with fail-closed redaction for that step."""
    from policy import apply_policy

    warnings: list[str] = []
    t0 = time.perf_counter()
    audit_rounds = 0

    norm = normalize_image(img.convert("RGB"))
    words = extract_words(norm.image)
    if not words:
        raise RuntimeError(
            "No text detected in the image. Try a sharper, well-lit photo."
        )

    # Layer 1: fresh classify call.
    try:
        document_type = llm.classify(words)
    except (OllamaError, OllamaParseError) as exc:
        warnings.append(f"Classify step skipped ({exc}); type unknown.")
        document_type = "other"

    # Layer 2: fresh, purpose-agnostic detect call (batched: short ids stay
    # exact, each batch fits comfortably inside the model's JSON budget).
    try:
        labels = llm.detect_batched(words)
        if llm.LAST_DETECT_FAILED_IDS:
            warnings.append(
                f"{len(llm.LAST_DETECT_FAILED_IDS)} fragment(s) had unparseable "
                "label output; fail-closed to REDACT."
            )
        labels, unanchored_amounts = anchor_amount_labels(words, labels)
        if unanchored_amounts:
            warnings.append(
                f"{len(unanchored_amounts)} money value(s) without a salary "
                "anchor were demoted to other_amount (deterministic rule)."
            )
        labels, counterparties = anchor_name_labels(words, labels)
        if counterparties:
            warnings.append(
                f"{len(counterparties)} ledger-row name(s) were treated as "
                "counterparties and redacted (deterministic rule)."
            )
        labeled_ids = {entry["id"] for entry in labels}
        unlabeled = sorted({w.id for w in words} - labeled_ids)
    except (OllamaError, OllamaParseError) as exc:
        warnings.append(f"Detect step failed ({exc}); failing closed: everything redacted.")
        labels, unlabeled = [], [w.id for w in words]

    # Layer 3: deterministic policy lookup.
    split = apply_policy(labels, preset)
    llm_redact_ids = sorted(set(split["redact_ids"]) | set(unlabeled))
    keep_ids = [i for i in split["keep_ids"] if i not in set(unlabeled)]

    # Layer 5a: regex net over the kept fragments.
    decision = LLMDecision(
        document_type=document_type,
        redact_ids=llm_redact_ids,
        keep_ids=keep_ids,
        reasoning="plan-v2 layered pipeline",
    )
    corrected, hits, unfilled = apply_validator(words, decision)
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
    protected_types = set(preset.get("purpose_critical_types", []))
    protected_ids = {
        entry["id"]
        for entry in labels
        if entry.get("type") in protected_types and entry["id"] in set(corrected.keep_ids)
    }
    for _ in range(max_audit_rounds):
        visible = [w for w in words if w.id in set(corrected.keep_ids)]
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
        corrected, repaired = repair_anchored_keeps(words, corrected, purpose_key)
        if repaired:
            warnings.append(
                f"Repaired {len(repaired)} purpose-critical value(s) to stay "
                "visible (label-anchored rule)."
            )

    # Plan v1 compatibility: app.py calls run_pipeline().
    output = render_redaction(norm.image, words, corrected.redact_ids)
    elapsed = time.perf_counter() - t0

    # Verify every redacted word is actually covered by a drawn box.
    redacted_words = [w for w in words if w.id in set(corrected.redact_ids)]
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
