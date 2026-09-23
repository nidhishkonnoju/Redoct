"""Ollama-backed Classify+Detect step: one structured-JSON call per document.

Uses the exact prompt contract from the PRD (section 5), called directly via
the Ollama REST API (no framework wrapper). Parsing is pydantic-validated;
one retry on failure; unparseable after retry -> OllamaParseError so the
pipeline can fail closed.
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

# Metrics of the most recent classify_and_detect calls (debug/demo aid).
LAST_CALL_METRICS: list[dict] = []


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


_STOP_WORDS = frozenset({
    "a", "an", "the", "of", "to", "is", "in", "on", "at", "by", "for",
    "with", "from", "and", "or", "but", "per", "this", "these", "those",
    "period", "statement", "said", "will", "was", "are", "were",
})

_ONLY_SYMBOLS_RE = re.compile(r"^[^A-Za-z0-9]+$")


def filter_fragments(words: list["OcrWord"]) -> list["OcrWord"]:
    """Drop uninformative fragments before sending to the LLM.

    Stop words, punctuation-only tokens, and lone symbols carry no
    redaction-relevant signal. Fewer fragments => the LLM is more likely
    to list *every* ID in its response. Dropped words are handled by the
    fail-closed path in apply_validator (they simply end up redacted).
    """
    kept = []
    for w in words:
        text = w.text.strip()
        if not text:
            continue
        if _ONLY_SYMBOLS_RE.match(text):
            continue
        if text.lower() in _STOP_WORDS:
            continue
        kept.append(w)
    return kept


def build_id_maps(ocr_words: list["OcrWord"]) -> tuple[dict[str, str], dict[str, str]]:
    """Return (full->short, short->full) id maps, 1:1 and order-stable."""
    forward: dict[str, str] = {}
    backward: dict[str, str] = {}
    for index, word in enumerate(ocr_words):
        short = f"{_SHORT_ALPHABET[index // 26]}{index % 26:02d}"
        forward[word.id] = short
        backward[short] = word.id
    return forward, backward


SYSTEM_PROMPT = """You are a document redaction assistant. You will receive a list of text
fragments extracted from a document via OCR, each with a unique short ID
(like "a07"). Your job:

1. Classify the document type: one of ["bank_statement", "pan_card", "salary_slip",
   "marksheet", "aadhaar_card", "voter_id", "other"], using these decisive
   markers in order:
   - an IFSC code, a "Balance" column, or "Statement of Account" => bank_statement
     (transaction rows reading "SALARY CREDIT" do NOT make it a salary_slip)
   - "Permanent Account Number" or "Income Tax Department" => pan_card
   - "UIDAI" or "Aadhaar", or a 12-digit number in 4-4-4 groups => aadhaar_card
   - an earnings vs deductions breakdown with gross/net pay => salary_slip
   - subject names with marks or grades => marksheet
   - none of the above => other
2. For the given SHARING PURPOSE, decide which fragment IDs must be KEPT
   (visible). Everything you do NOT keep is hidden automatically, so name
   only the fragments to keep.
3. Match each OCR fragment to the closest field category by its content and
   context (e.g. a 10-digit number near the word "Account" is account_number;
   a 12-digit number in groups of 4 is likely aadhaar_number). Labels like
   "Name:" or "Amount" on their own are usually safe to KEEP — redact the
   sensitive values, not the labels.
4. Be stingy: keep only what this purpose genuinely requires. Omit anything
   uncertain — an omitted fragment is hidden, so a wrong omission is harmless
   while a wrong keep leaks data.

Respond ONLY with valid JSON in this exact schema:
{"document_type": "...", "keep_ids": ["a03","a04"], "reasoning": "<=15 words"}

keep_ids must list EVERY fragment you want to remain visible. Do not list the
ids you want hidden, and do not invent ids that were not in the prompt.

The document is the OCR text only; the id lists contain only ids."""


def build_prompt(ocr_words: list[OcrWord], preset: dict) -> tuple[str, str, dict[str, str]]:
    """Return (system, user, short->full id map).

    Fragments are rendered one per line as "short_id: text" — fragment format
    benchmarked best on decision quality; short ids cut output tokens ~4x.
    """
    forward, backward = build_id_maps(ocr_words)
    fragments = "\n".join(f"{forward[w.id]}: {w.text}" for w in ocr_words)
    user = (
        f"Sharing purpose: {preset['label']}\n"
        f"Fields to KEEP for this purpose: {preset['keep']}\n"
        f"Fields to REDACT for this purpose: {preset['redact']}\n\n"
        f"OCR fragments:\n{fragments}\n\n"
        "Classify the document and decide redact_ids/keep_ids now."
    )
    return SYSTEM_PROMPT, user, backward


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


def _call_ollama(system: str, user: str) -> tuple[str, dict]:
    payload = {
        "model": OLLAMA_MODEL,
        "system": system,
        "prompt": user,
        "format": "json",
        "stream": False,
        "keep_alive": OLLAMA_KEEP_ALIVE,
        "options": OLLAMA_OPTIONS,
    }
    try:
        resp = requests.post(
            f"{OLLAMA_BASE_URL}/api/generate", json=payload, timeout=OLLAMA_TIMEOUT_S
        )
    except requests.RequestException as exc:
        raise OllamaError(f"Ollama request failed: {exc}") from exc
    if resp.status_code != 200:
        raise OllamaError(f"Ollama HTTP {resp.status_code}: {resp.text[:200]}")
    try:
        data = resp.json()
        return data["response"], data
    except (ValueError, KeyError) as exc:
        raise OllamaError(f"Unexpected Ollama response: {resp.text[:200]}") from exc


_FENCE_RE = re.compile(r"^```[a-zA-Z]*\s*|\s*```$")


def _parse_decision(raw: str) -> LLMDecision:
    """Parse model output; tolerate optional markdown fences."""
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc
    if not isinstance(data, dict):
        raise ValueError("expected a JSON object")
    try:
        return LLMDecision.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"schema mismatch: {exc}") from exc


def _warmup_call() -> bool:
    try:
        _call_ollama(system="Reply with JSON only.", user='{"ready": true}')
        return True
    except OllamaError:
        return False


def warmup() -> bool:
    """Load the model with a 1-token call so the first real request is warm."""
    return _warmup_call()


def classify_and_detect(
    ocr_words: list[OcrWord], purpose_key: str, preset: dict
) -> LLMDecision:
    """One structured call (+1 retry) -> validated decision with full ids."""
    filtered = filter_fragments(ocr_words)
    system, user, short_to_full = build_prompt(filtered, preset)
    last_error: Exception | None = None
    decision: LLMDecision | None = None
    for attempt in range(2):
        prompt = user
        if attempt == 1:
            prompt = (
                f"{user}\n\nYour previous reply was not valid for this schema "
                f"({last_error}). Reply with ONLY the JSON object."
            )
        t0 = time.perf_counter()
        raw, meta = _call_ollama(system, prompt)
        LAST_CALL_METRICS.append({
            "purpose": purpose_key,
            "attempt": attempt + 1,
            "model": meta.get("model", OLLAMA_MODEL),
            "total_s": round(time.perf_counter() - t0, 2),
            "load_s": round(meta.get("load_duration", 0) / 1e9, 2),
            "prompt_eval_s": round(meta.get("prompt_eval_duration", 0) / 1e9, 2),
            "eval_s": round(meta.get("eval_duration", 0) / 1e9, 2),
            "prompt_tokens": meta.get("prompt_eval_count", 0),
            "output_tokens": meta.get("eval_count", 0),
            "raw_response": raw,
        })
        try:
            decision = _parse_decision(raw)
            break
        except ValueError as exc:
            last_error = exc
    if decision is None:
        raise OllamaParseError(f"LLM output unparseable after retry: {last_error}")

    # Map short ids back to full OCR ids; keep only known ids.
    valid = {w.id for w in ocr_words}
    unmapped_keep = [i for i in decision.keep_ids if i not in short_to_full]
    unmapped_redact = [i for i in decision.redact_ids if i not in short_to_full]
    decision.redact_ids = [
        short_to_full[i] for i in decision.redact_ids if i in short_to_full
    ]
    decision.keep_ids = [
        short_to_full[i] for i in decision.keep_ids if i in short_to_full
    ]
    decision.redact_ids = [i for i in decision.redact_ids if i in valid]
    decision.keep_ids = [i for i in decision.keep_ids if i in valid]
    # Ids the model invented (not in the prompt) — visible in metrics for debug.
    LAST_CALL_METRICS[-1]["unmapped_keep"] = unmapped_keep
    LAST_CALL_METRICS[-1]["unmapped_redact"] = unmapped_redact
    return decision


if __name__ == "__main__":
    import time

    from ocr import OcrWord

    ok, msg = check_ollama()
    print(("OK: " if ok else "FAIL: ") + msg)
    if not ok:
        raise SystemExit(1)

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
        "keep": ["name", "salary_amount", "employer_name"],
        "redact": ["account_number", "address", "phone_number"],
    }
    t0 = time.perf_counter()
    result = classify_and_detect(demo, "proof_of_income", preset)
    print(f"in {time.perf_counter() - t0:.1f}s ->")
    print("  metrics:", LAST_CALL_METRICS[-1])
    print("  type:", result.document_type)
    print("  redact:", result.redact_ids)
    print("  keep:", result.keep_ids)
    print("  reasoning:", result.reasoning)
