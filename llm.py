"""Ollama-backed LLM layers (Plan v3): Classify, Detect, and Audit.

Each layer makes its own **fresh** `/api/generate` request — no chat session is
reused, so the audit in particular never sees the reasoning of the layers before
it. Parsing is validated per layer; one retry on failure; after a retry the layer
raises `OllamaParseError` so the pipeline can fail closed.

Plan v3 Priority 0: every layer speaks a **line protocol**, not JSON (see
`prompts.py`). A 3B model truncating a nested JSON object mid-string used to
lose a whole batch; a truncated line-based reply still parses, because every
line is self-contained. `format: "json"` (which forced the model into JSON and
into the truncation failure) is gone from the request payload.

All prompt text lives in `prompts.py`; this module owns transport and parsing.
"""
from __future__ import annotations

import json
import re
import time
from typing import Literal

import requests
from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from config import (
    DETECT_BATCH_SIZE,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
    OLLAMA_MODEL,
    OLLAMA_OPTIONS,
    OLLAMA_TIMEOUT_S,
)
from ocr import OcrWord, group_lines
from prompts import (
    AUDIT_SYSTEM,
    CLASSIFY_SYSTEM,
    FIELD_TYPES,
    WARMUP_SYSTEM,
    WARMUP_USER,
    audit_user,
    classify_user,
    detect_system,
    detect_user,
)


DocumentType = Literal[
    "bank_statement", "pan_card", "salary_slip", "marksheet",
    "aadhaar_card", "voter_id", "other",
]


class OllamaError(RuntimeError):
    """Ollama server or model unavailable."""


class OllamaParseError(ValueError):
    """Model output unparseable after retry -> pipeline must fail closed."""


class LLMDecision(BaseModel):
    """Validated structured decision from the model."""

    document_type: DocumentType = "other"
    redact_ids: list[str] = []
    keep_ids: list[str] = []
    # Plan v3 Priority 1: ids covered by a black box with a masked value
    # printed over it — never shown to the audit, never treated as keep.
    partial_ids: list[str] = []
    reasoning: str = ""

    model_config = ConfigDict(extra="ignore")

    @field_validator("redact_ids", "keep_ids", "partial_ids", mode="before")
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


def _call_generate(system: str, user: str, layer: str = "") -> str:
    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": user,
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
        "layer": layer or system[:40],
        "system_preview": system[:80], "total_s": round(elapsed, 2),
    })
    return data.get("response", "")


def _classify_texts(texts: list[str]) -> str:
    """Layer 1: document_type only, no ids, no purpose.

    Input is the OCR text as reading-order visual lines (not loose words): a
    small model needs the layout hints ("ACCOUNT NO" next to its value) more
    than it needs every stray token. Reply is a bare type name, not JSON.
    """
    raw = _call_generate(CLASSIFY_SYSTEM, classify_user("\n".join(texts)), layer="classify")
    cleaned = raw.strip().strip('"').strip("'").strip(".").strip().lower()
    for candidate in (
        "bank_statement", "pan_card", "salary_slip", "marksheet",
        "aadhaar_card", "voter_id", "other",
    ):
        if candidate in cleaned:
            return candidate
    return "other"


# --- Line-protocol parsing (Plan v3 Priority 0) -------------------------------
# One line per entry, `id: value`. Tolerant about the noise a 3B model adds
# (bullets, numbering, markdown emphasis, `=` instead of `:`) and strict about
# the value: an id whose value is not a known type fails closed to `other`.
_LABEL_LINE_RE = re.compile(
    r"^[\s>*#`\-–—•]*(?:\d+[.)]\s*)?[`*_]*([A-Za-z][A-Za-z0-9]{1,3})[`*_]*\s*(?:[:=]|->|→)\s*(.+?)[\s.,;`*_]*$"
)
_AUDIT_LINE_RE = re.compile(r"^[\s>*#`\-–—•]*([A-Z]+)\s*[:=]\s*(.*)$")
_NO_FLAG_WORDS = ("none", "n/a", "na", "no", "nothing", "nil", "-", "[]", "{}", "no ids")
_MATCHABLE_TYPES = sorted(FIELD_TYPES, key=len, reverse=True)
# Shortest token allowed to be rescued by a unique-prefix match: below this,
# "na" would mean `name`, and one typo would silently move a fragment between
# keep and redact.
_MIN_PREFIX_MATCH = 4



def _match_type(token: str) -> str | None:
    """Map a model's type token to a vocabulary entry, else None.

    Exact match first; otherwise the longest vocabulary entry contained in the
    token, which rescues chatty answers ("type is name", "name (person)").
    Longest-first matters: `other_amount` must beat `other`.

    Last resort is a *unique prefix* match, which rescues the one line a reply
    truncated mid-value ("a07: account_num"). It is deliberately narrow: the
    token must be at least `_MIN_PREFIX_MATCH` characters AND exactly one
    vocabulary entry may start with it. Anything ambiguous — "date" (could be
    `date_of_birth` or `transaction_date`), "o", "trans" — stays None, and the
    caller fails closed to `other` (a redact type in every preset).
    """
    token = token.strip().strip("`*_\"'.").lower()
    if not token:
        return None
    if token in _MATCHABLE_TYPES:
        return token
    for candidate in _MATCHABLE_TYPES:
        if candidate in token:
            return candidate
    if len(token) >= _MIN_PREFIX_MATCH:
        prefixed = [c for c in _MATCHABLE_TYPES if c.startswith(token)]
        if len(prefixed) == 1:
            return prefixed[0]
    return None


def _parse_label_lines(raw: str, backward: dict[str, str]) -> list[dict]:
    """Parse `id: type` lines into [{\"id\": full id, \"type\": field type}]."""
    out: list[dict] = []
    seen: set[str] = set()
    unknown = 0
    for line in raw.splitlines():
        match = _LABEL_LINE_RE.match(line)
        if not match:
            continue
        short, type_token = match.group(1).lower(), match.group(2)
        full = backward.get(short)
        if not full or full in seen:
            continue
        ftype = _match_type(type_token)
        if ftype is None:
            unknown += 1
            ftype = "other"
        seen.add(full)
        out.append({"id": full, "type": ftype})
    if out and unknown == len(out):
        # Every value was unrecognized: the model echoed the fragment text or
        # answered in prose. Treat as a parse failure so the batch splitter can
        # retry with fewer fragments instead of failing the whole document.
        raise ValueError(f"no recognizable field types in {unknown} labeled line(s)")
    return out


def _parse_labels_json(raw: str, backward: dict[str, str]) -> list[dict]:
    """Legacy JSON fallback: a model that ignores the line protocol still parses."""
    data = json.loads(raw.strip().strip("`").strip())
    if isinstance(data, dict) and isinstance(data.get("labels"), list):
        items = data["labels"]
    elif isinstance(data, list):
        items = data
    else:
        raise ValueError("expected {labels: [...]} object")
    out: list[dict] = []
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            continue
        full = backward.get(str(item.get("id", "")).strip().lower())
        if not full or full in seen:
            continue
        seen.add(full)
        out.append({"id": full, "type": _match_type(str(item.get("type", ""))) or "other"})
    return out


def _parse_labels(raw: str, backward: dict[str, str]) -> list[dict]:
    """Parse Layer-2 output: line protocol first, JSON only as a fallback."""
    stripped = raw.strip().strip("`").strip()
    if stripped.startswith(("{", "[")):
        try:
            return _parse_labels_json(stripped, backward)
        except (ValueError, json.JSONDecodeError):
            pass
    return _parse_label_lines(raw, backward)


def _parse_audit(raw: str) -> tuple[list[str], str]:
    """Parse Layer-5b output: `FLAGGED: ...` / `REASON: ...` lines."""
    flagged: list[str] = []
    reasoning = ""
    saw_flags = False
    for line in raw.splitlines():
        match = _AUDIT_LINE_RE.match(line)
        if not match:
            continue
        key, value = match.group(1), match.group(2).strip().strip("`*_\"'")
        if key == "FLAGGED":
            saw_flags = True
            if value.lower() in _NO_FLAG_WORDS:
                continue
            for token in re.split(r"[,\s]+", value):
                token = token.strip().strip("[]\"'`.", ).lower()
                if token and token not in _NO_FLAG_WORDS:
                    flagged.append(token)
        elif key == "REASON" and not reasoning:
            reasoning = value
    if not saw_flags:
        stripped = raw.strip().strip("`").strip()
        if stripped.startswith(("{", "[")):
            data = json.loads(stripped)
            if isinstance(data, dict) and isinstance(data.get("flagged_ids"), list):
                return [str(i).strip().lower() for i in data["flagged_ids"]], str(
                    data.get("reasoning", "")
                ).strip()
        raise ValueError("no FLAGGED line in audit output")
    return flagged, reasoning


def detect(ocr_words: "list[OcrWord]", document_type: str = "other") -> list[dict]:
    """Plan v3, Layer 2: purpose-agnostic field labels, one fresh call.

    `document_type` (from Layer 1) only selects a preservation note for the
    system prompt — it never changes what is *sensitive*, only what the model
    should not mistake for a date of birth. There is no JSON in the request and
    none in the reply: one `id: type` line per fragment.
    """
    forward, backward = build_id_maps(ocr_words)
    fragments = "\n".join(f"{forward[w.id]}: {w.text}" for w in ocr_words)
    system = detect_system(document_type)
    user = detect_user(fragments)
    for attempt in range(2):
        try:
            raw = _call_generate(system, user, layer="detect")
            out = _parse_labels(raw, backward)
        except (ValueError, json.JSONDecodeError, OllamaError) as exc:
            last_error = exc
            continue
        if not out:
            last_error = ValueError("No labeled ids parsed from response")
            continue
        return out
    raise OllamaParseError(f"Detect output unparseable after retry: {last_error}")


# Ids whose Detect batch never produced usable labels: the pipeline fails them
# closed (redacted) and says so in the UI, instead of losing the whole document.
LAST_DETECT_FAILED_IDS: list[str] = []


def _detect_batch(
    batch: "list[OcrWord]", depth: int = 0, document_type: str = "other"
) -> list[dict]:
    """Detect one batch; on unparseable output, split it and retry the halves.

    Kept as the *secondary* defence after Plan v3's line protocol: a truncated
    reply now usually still parses, but a model can always go off-format, and
    halving isolates the offending fragment(s) instead of discarding a whole
    document. Batches of <= 5 fragments that still fail are recorded as failed
    ids.
    """
    try:
        return detect(batch, document_type=document_type)
    except OllamaParseError:
        if len(batch) <= 5 or depth >= 3:
            LAST_DETECT_FAILED_IDS.extend(w.id for w in batch)
            return []
        mid = len(batch) // 2
        return _detect_batch(batch[:mid], depth + 1, document_type) + _detect_batch(
            batch[mid:], depth + 1, document_type
        )


def detect_batched(
    ocr_words: "list[OcrWord]",
    batch_size: int = DETECT_BATCH_SIZE,
    document_type: str = "other",
) -> list[dict]:
    """Shard a long fragment list into deterministic id-exact batches.

    Sharding is by position in the (reading-order) fragment list, so the same
    document always produces the same batches. A batch that cannot be parsed is
    split (see `_detect_batch`) rather than failing the whole run.
    """
    LAST_DETECT_FAILED_IDS.clear()
    out: list[dict] = []
    for start in range(0, max(1, len(ocr_words)), batch_size):
        out.extend(
            _detect_batch(ocr_words[start:start + batch_size], document_type=document_type)
        )
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


def validate_llm(
    ocr_words: "list[OcrWord]",
    keep_ids: list[str],
    purpose_label: str,
) -> tuple[list[str], str]:
    """Plan v2, Layer 5b: fresh-context audit of visible fragments only.

    Plan v3: the reply is `FLAGGED: <ids>` / `REASON: <sentence>`, not JSON.
    Only ids that exist in this document are returned, so a hallucinated id can
    never be "re-redacted" into the result.
    """
    fragments = "\n".join(
        f"{w.id}: {w.text}" for w in ocr_words if w.id in set(keep_ids)
    )
    user = audit_user(purpose_label, fragments)
    for attempt in range(2):
        try:
            raw = _call_generate(AUDIT_SYSTEM, user, layer="audit")
            flagged, reasoning = _parse_audit(raw)
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
        _call_generate(system=WARMUP_SYSTEM, user=WARMUP_USER, layer="warmup")
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
    doc_type = classify(demo)
    print("  Layer 1 classify:", doc_type, f"({time.perf_counter() - t0:.1f}s)")

    t0 = time.perf_counter()
    labels = detect_batched(demo, document_type=doc_type)
    print(f"  Layer 2 detect: {labels} ({time.perf_counter() - t0:.1f}s)")

    from policy import apply_policy

    split = apply_policy(labels, preset)
    print(
        "  Layer 3 policy: redact", split["redact_ids"],
        "keep", split["keep_ids"],
        "partial", split["partial_ids"],
    )

    t0 = time.perf_counter()
    visible = [w for w in demo if w.id in set(split["keep_ids"])]
    print(
        "  Layer 5b audit:",
        validate_llm(visible, [w.id for w in visible], preset["label"]),
        f"({time.perf_counter() - t0:.1f}s)",
    )
    for call in LAST_LAYER_CALLS:
        print(f"    {call['total_s']:6.2f}s  {call['system_preview'][:40]}")
