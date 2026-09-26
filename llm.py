"""Ollama-backed LLM layers (Plan v2): Classify, Detect, and Audit.

Each layer makes its own **fresh** `/api/generate` request — no chat session is
reused, so the audit in particular never sees the reasoning of the layers before
it. Parsing is validated per layer; one retry on failure; after a retry the layer
raises `OllamaParseError` so the pipeline can fail closed.
"""
from __future__ import annotations

import json
import re
import time
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from config import (
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_TIMEOUT_S,
)
from ocr import OcrWord, group_lines


DocumentType = Literal[
    "bank_statement", "pan_card", "salary_slip", "marksheet",
    "aadhaar_card", "voter_id", "other",
]


class OllamaError(RuntimeError):
    """Ollama server or model unavailable."""


class OllamaParseError(ValueError):
    """Model output unparseable after retry -> pipeline must fail closed."""


FIELD_TYPES: tuple[str, ...] = (
    "name", "father_name", "address", "phone_number", "email",
    "date_of_birth", "account_number", "ifsc_code", "aadhaar_number",
    "pan_number", "voter_id_number", "salary_amount", "other_amount",
    "employer_name", "designation", "pf_number", "institution_name",
    "qualification", "cgpa_or_marks", "roll_number", "category",
    "transaction_line", "signature_marker", "photo_marker",
    "label_text", "remarks", "other",
)

class LLMDecision(BaseModel):
    """Validated structured decision from the model."""

    document_type: DocumentType = "other"
    redact_ids: list[str] = []
    keep_ids: list[str] = []
    reasoning: str = ""

    model_config = ConfigDict(extra="ignore")

    @field_validator("redact_ids", "keep_ids", mode="before")
    @classmethod
    def _clean_ids(cls, value):
        if value is None:
            return []
        seen: set[str] = set()
        cleaned: list[str] = []
        for item in value if isinstance(value, list) else [value]:
            token = str(item).strip().lower()
            if token and token not in seen:
                seen.add(token)
                cleaned.append(token)
        return cleaned

    @field_validator("reasoning", mode="before")
    @classmethod
    def _clean_reasoning(cls, value):
        return str(value or "").strip()


# --- Short-id compression ----------------------------------------------------
# Full OCR ids like "w0042" cost ~6 output tokens each; qwen spells them out
# ("w00", "42", "0042" variants). Two-digit ids ("b07") are single tokens, so
# prompts use short ids and responses are mapped back to full ids here.
_SHORT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"


def build_id_maps(ocr_words: list["OcrWord"]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (full->short, short->full) id maps, 1:1 and order-stable."""
    forward: dict[str, str] = {}
    backward: dict[str, str] = {}
    for index, word in enumerate(ocr_words):
        short = f"{_SHORT_ALPHABET[index // 26]}{index % 26:02d}"
        forward[word.id] = short
        backward[short] = word.id
    return forward, backward


# Per-layer call log (Plan v2): which layer ran, how long it took.
# Defined before _call_generate so nothing dangles on import order.
LAST_LAYER_CALLS: list[dict] = []


def _call_generate(system: str, user: str) -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": user,
        "format": "json",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_S
        )
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    elapsed = time.perf_counter() - t0
    if resp.status_code != 200:
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
    except ValueError as exc:
        raise OllamaError(f"Unexpected Ollama response: {resp.text[:200]}") from exc
    LAST_LAYER_CALLS.append({
        "system_preview": system[:80], "total_s": round(elapsed, 2),
    })
    return data.get("response", "")


def _classify_texts(texts: list[str]) -> str:
    """Layer 1: document_type only, no ids, no purpose.

    Input is the OCR text as reading-order visual lines (not loose words): a
    small model needs the layout hints ("ACCOUNT NO" next to its value) more
    than it needs every stray token.
    """
    system = (
        "You classify a document from its OCR text. Reply with ONLY the type "
        "name. Types and their tell-tale words: bank_statement (account number, "
        "IFSC, branch, statement period, balance), pan_card (permanent account "
        "number, father's name, income tax), salary_slip (basic, HRA, gross, "
        "net pay, deductions, employee code), marksheet (semester, grade, CGPA, "
        "roll number, university), aadhaar_card (aadhaar, UIDAI, VID), voter_id "
        "(elector, EPIC, constituency), other (none of these)."
    )
    user = (
        "OCR text, reading order, one visual line per row:\n"
        + "\n".join(texts)
        + "\n\nDocument type:"
    )
    raw = _call_generate(system, user)
    cleaned = raw.strip().strip('"').strip("'").lower()
    for candidate in (
        "bank_statement", "pan_card", "salary_slip", "marksheet",
        "aadhaar_card", "voter_id", "other",
    ):
        if candidate in cleaned:
            return candidate
    return "other"


def _parse_labels(raw: str, backward: dict[str, str]) -> list[dict]:
    """Parse Layer-2 labels output into [{"id": full id, "type": field type}]."""
    data = json.loads(raw.strip().strip("`").strip())
    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        items = data["labels"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("expected {labels: [...]} object")
    vocab = set(FIELD_TYPES)
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        short = str(item.get("id", "")).strip().lower()
        ftype = str(item.get("type", "")).strip().lower()
        if ftype not in vocab:
            ftype = "other"
        full = backward.get(short)
        if not full or full in seen:
            continue
        seen.add(full)
        out.append({"id": full, "type": ftype})
    return out


def _parse_flags(raw: str, backward: dict[str, str] | None = None) -> list[str]:
    """Parse Layer-5 audit output into a list of full OCR ids."""
    data = json.loads(raw.strip().strip("`").strip())
    if isinstance(data, dict) and isinstance(data.get("flagged_ids"), list):
        items = data["flagged_ids"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("expected {flagged_ids: [...]} object")
    out: list[str] = []
    for item in items:
        token = str(item).strip().lower()
        full = backward.get(token, token) if backward else token
        if full and full not in out:
            out.append(full)
    return out


DETECT_VOCABULARY = ", ".join(FIELD_TYPES)

DETECT_SYSTEM = (
    "You label OCR text fragments with exactly one field type each. "
    "Valid types: " + DETECT_VOCABULARY + ". "
    "Rules: 'label_text' marks a field LABEL itself ('Name:', captions) — "
    "NOT the value after it. 'salary_amount' is ONLY the salary/income figure "
    "(a value on a line mentioning SALARY, gross pay, net pay or basic); every "
    "other money value — rent, purchases, balances, totals — is 'other_amount'. "
    "Every fragment gets exactly one label. "
    "If genuinely uncertain between a sensitive type and 'other', choose "
    "the sensitive type. Do NOT decide visibility — only identify what "
    "each fragment IS."
)


def detect(ocr_words: "list[OcrWord]") -> list[dict]:
    """Plan v2, Layer 2: purpose-agnostic field labels, one fresh call."""
    forward, backward = build_id_maps(ocr_words)
    fragments = "\n".join(f"{forward[w.id]}: {w.text}" for w in ocr_words)
    user = (
        "Fragments:\n" + fragments + "\n\n"
        "Respond ONLY as JSON, one entry per fragment id, no explanation: "
        '{"labels": [{"id": "a00", "type": "name"}, ...]}'
    )
    for attempt in range(2):
        try:
            out = _parse_labels(_call_generate(DETECT_SYSTEM, user), backward)
        except (ValueError, json.JSONDecodeError, OllamaError) as exc:
            last_error = exc
            continue
        if not out:
            last_error = ValueError("No labeled ids parsed from response")
            continue
        return out
    raise OllamaParseError(f"Detect output unparseable after retry: {last_error}")


# Ids whose Detect batch never produced usable JSON: the pipeline fails them
# closed (redacted) and says so in the UI, instead of losing the whole document.
LAST_DETECT_FAILED_IDS: list[str] = []


def _detect_batch(batch: "list[OcrWord]", depth: int = 0) -> list[dict]:
    """Detect one batch; on unparseable output, split it and retry the halves.

    A 3B model occasionally runs long and returns truncated JSON. Halving
    isolates the offending fragment(s) instead of discarding a whole document;
    batches of <= 5 fragments that still fail are recorded as failed ids.
    """
    try:
        return detect(batch)
    except OllamaParseError:
        if len(batch) <= 5 or depth >= 3:
            LAST_DETECT_FAILED_IDS.extend(w.id for w in batch)
            return []
        mid = len(batch) // 2
        return _detect_batch(batch[:mid], depth + 1) + _detect_batch(batch[mid:], depth + 1)


def detect_batched(ocr_words: "list[OcrWord]", batch_size: int = 25) -> list[dict]:
    """Shard a long fragment list into deterministic id-exact batches.

    Sharding is by position in the (reading-order) fragment list, so the same
    document always produces the same batches. A batch that cannot be parsed is
    split (see `_detect_batch`) rather than failing the whole run.
    """
    LAST_DETECT_FAILED_IDS.clear()
    out: list[dict] = []
    for start in range(0, max(1, len(ocr_words)), batch_size):
        out.extend(_detect_batch(ocr_words[start:start + batch_size]))
    seen: set[str] = set()
    unique: list[dict] = []
    for entry in out:
        if entry["id"] in seen:
            continue
        seen.add(entry["id"])
        unique.append(entry)
    return unique


def classify(ocr_words: "list[OcrWord]") -> str:
    """Plan v2, Layer 1: document_type only, no ids, no purpose info."""
    lines = [
        " ".join(w.text for w in line) for line in group_lines(ocr_words)
    ]
    return _classify_texts(lines[:60])


AUDIT_SYSTEM = (
    "You audit a redaction result. You have NO knowledge of how these "
    "decisions were made. You see only fragments currently kept visible plus "
    "the sharing purpose. Flag an id ONLY if its text is itself sensitive "
    "(a personal identifier, account or card number, address, phone, email, "
    "date of birth, signature, or a family member's name/number). Plain "
    "labels ('Name:', 'Employer:'), generic words, amounts, dates that are "
    "not a date of birth, and employer or institution names are NOT "
    "sensitive — never flag those. When nothing is sensitive, return "
    "flagged_ids as an empty list."
)


def validate_llm(
    ocr_words: "list[OcrWord]",
    keep_ids: list[str],
    purpose_label: str,
) -> tuple[list[str], str]:
    """Plan v2, Layer 5b: fresh-context audit of visible fragments only."""
    by_id = {w.id: w for w in ocr_words}
    fragments = "\n".join(
        f"{w.id}: {w.text}" for w in ocr_words if w.id in set(keep_ids)
    )
    user = (
        f"Sharing purpose: {purpose_label}\n"
        f"Fragments currently kept visible:\n{fragments}\n\n"
        "Respond ONLY as JSON: "
        '{"flagged_ids": ["w0001"], "reasoning": "one sentence"}'
    )
    for attempt in range(2):
        try:
            raw = _call_generate(AUDIT_SYSTEM, user)
            flagged = _parse_flags(raw)
            data = json.loads(raw.strip().strip("`").strip())
            reasoning = str(data.get("reasoning", "")).strip() if isinstance(data, dict) else ""
            valid = {w.id for w in ocr_words}
            return [i for i in flagged if i in valid], reasoning
        except (ValueError, json.JSONDecodeError, KeyError, OllamaError) as exc:
            last_error = exc
    raise OllamaParseError(f"Audit output unparseable after retry: {last_error}")


def check_ollama() -> tuple[bool, str]:
    """Verify the server is up and the configured model is available."""
    try:
        resp = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=5)
        resp.raise_for_status()
        models = [m.get("name", "") for m in resp.json().get("models", [])]
    except (requests.RequestException, ValueError) as exc:
        return False, f"Ollama not reachable at {OLLAMA_BASE_URL}: {exc}"
    base = OLLAMA_MODEL.split(":")[0]
    if any(m == OLLAMA_MODEL or m.split(":")[0] == base for m in models):
        return True, f"Ollama ready ({OLLAMA_MODEL})"
    return False, (
        f"Model '{OLLAMA_MODEL}' not found. Run: ollama pull {OLLAMA_MODEL}"
    )


def _warmup_call() -> bool:
    try:
        _call_generate(system="Reply with JSON only.", user='{"ready": true}')
        return True
    except OllamaError:
        return False


def warmup() -> bool:
    """Load the model with a 1-token call so the first real request is warm."""
    return _warmup_call()


if __name__ == "__main__":
    # Layered smoke test: one real call per layer, plus timings.
    ok, msg = check_ollama()
    print(("OK: " if ok else "FAIL: ") + msg)
    if not ok:
        raise SystemExit(1)
    print("  warmup:", warmup())

    demo = [
        OcrWord("w0001", "Account", 10, 10, 50, 20, 95),
        OcrWord("w0002", "Number:", 70, 10, 60, 20, 95),
        OcrWord("w0003", "5010023456789012", 140, 10, 200, 20, 90),
        OcrWord("w0004", "Name", 10, 40, 40, 20, 95),
        OcrWord("w0005", "ARJUN", 60, 40, 60, 20, 95),
        OcrWord("w0006", "MEHTA", 130, 40, 60, 20, 95),
    ]
    preset = {
        "label": "Proof of Income",
        "keep_types": ["name", "salary_amount", "employer_name", "label_text"],
        "redact_types": ["account_number", "address", "phone_number"],
    }

    t0 = time.perf_counter()
    print("  Layer 1 classify:", classify(demo), f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    labels = detect_batched(demo)
    print(f"  Layer 2 detect: {labels} ({time.perf_counter() - t0:.1f}s)")

    from policy import apply_policy

    split = apply_policy(labels, preset)
    print("  Layer 3 policy: redact", split["redact_ids"], "keep", split["keep_ids"])

    t0 = time.perf_counter()
    visible = [w for w in demo if w.id in set(split["keep_ids"])]
    print(
        "  Layer 5b audit:",
        validate_llm(visible, [w.id for w in visible], preset["label"]),
        f"({time.perf_counter() - t0:.1f}s)",
    )
    for call in LAST_LAYER_CALLS:
        print(f"    {call['total_s']:6.2f}s  {call['system_preview'][:40]}")
