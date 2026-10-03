"""All model-facing text for the three LLM layers (Plan v3).

Why this module exists
----------------------
Plan v3 Priority 0 replaced the JSON output format with a **line protocol**: a
3B model on a 2048-token output budget was truncating mid-string, and the
pipeline had to recover from `Unterminated string ... column 2744`. A line
protocol degrades gracefully — a reply that is cut short still yields every
line it managed to print — and it costs fewer output tokens per fragment than
nested JSON.

Everything here is a plain string or a pure function over strings: no I/O, no
Ollama, no imports. The field vocabulary and the birth-date label list are
defined here *once* so the prompt cannot drift from the parser's vocabulary or
from the regex net's anchors (see `DOB_ANCHORS`). That also makes this file a
direct port target for the Android build — copy the constants into a Kotlin
`object Prompts` and the ported pipeline behaves identically (README,
"Porting to Android").
"""
from __future__ import annotations

# --- Vocabulary shared with the parser and the regex net ---------------------

# Field vocabulary: the types Detect may emit. `llm._parse_labels` accepts only
# these, so a model that invents a type cannot smuggle it past the parser.
FIELD_TYPES: tuple[str, ...] = (
    "name", "father_name", "address", "phone_number", "email",
    "date_of_birth", "transaction_date", "account_number", "ifsc_code",
    "aadhaar_number", "pan_number", "voter_id_number", "salary_amount",
    "other_amount", "employer_name", "designation", "pf_number",
    "institution_name", "qualification", "cgpa_or_marks", "roll_number",
    "category", "transaction_line", "signature_marker", "photo_marker",
    "label_text", "remarks", "other",
)

DETECT_VOCABULARY = ", ".join(FIELD_TYPES)

# Labels that make a date a date of birth. `redact.py` uses this same tuple for
# its `date_of_birth` context rule and for `anchor_date_labels()`, so the
# prompt's guidance and the deterministic net cannot disagree. Devanagari
# entries cover a document photographed with a Hindi birth-date label.
DOB_ANCHORS: tuple[str, ...] = (
    "DATE OF BIRTH",
    "D.O.B",
    "DOB",
    "BIRTH DATE",
    "BIRTH",
    "जन्म",
    "जन्म की तारीख",
)
_DOB_LABEL_HINT = ", ".join("'" + anchor + "'" for anchor in DOB_ANCHORS)

NO_FORMAT_NOISE = (
    "Plain text only: no JSON, no braces, no quotes, no bullets, no headings, "
    "no commentary."
)


# --- Layer 1: Classify --------------------------------------------------------

CLASSIFY_SYSTEM = (
    "You classify a document from its OCR text. Reply with the type name only: "
    "one lowercase word, no punctuation, no explanation. "
    "Types and their tell-tale words: bank_statement (account number, IFSC, "
    "branch, statement period, balance), pan_card (permanent account number, "
    "father's name, income tax), salary_slip (basic, HRA, gross, net pay, "
    "deductions, employee code), marksheet (semester, grade, CGPA, roll "
    "number, university), aadhaar_card (aadhaar, UIDAI, VID), voter_id "
    "(elector, EPIC, constituency), other (none of these). " + NO_FORMAT_NOISE
)


def classify_user(lines: str) -> str:
    """Layer-1 user prompt from reading-order visual lines."""
    return (
        "OCR text, reading order, one visual line per row:\n"
        + lines
        + "\n\nDocument type:"
    )


# --- Layer 2: Detect ----------------------------------------------------------

DETECT_BASE_SYSTEM = (
    "You label OCR text fragments with exactly one field type each. "
    "Valid types: " + DETECT_VOCABULARY + ". "
    "Rules: 'label_text' marks a field LABEL itself ('Name:', captions) — "
    "NOT the value after it. 'salary_amount' is ONLY the salary/income figure "
    "(the value on a row mentioning SALARY, gross pay, net pay or basic); "
    "every other money value — rent, purchases, balances, totals — is "
    "'other_amount'. 'transaction_date' is any date printed in a statement, "
    "table or ledger row (transaction dates, statement periods, pay periods, "
    "document dates). 'date_of_birth' is ONLY a date written beside a "
    "birth-date label (" + _DOB_LABEL_HINT + "); a bare date with no such "
    "label is 'transaction_date'. "
    "Give every fragment exactly one label. If you are genuinely unsure "
    "between a sensitive type and 'other', choose the sensitive type. Do NOT "
    "decide visibility — only identify what each fragment IS."
)

# Plan v3: document-type-aware preservation notes. The base rules are
# purpose-agnostic; these notes only stop the model from inventing a date of
# birth on a document whose dates are all ledger dates (the failure the
# `transaction_date` type exists to fix).
DETECT_PRESERVE_NOTES: dict[str, str] = {
    "bank_statement": (
        "This is a bank statement: a date inside the transaction table is the "
        "transaction date, never a date of birth, and the account holder's own "
        "name is not a third party. Label a date in a transaction row "
        "'transaction_date' and the account holder's name 'name'."
    ),
    "salary_slip": (
        "This is a salary slip: its dates are pay periods, joining dates or "
        "payment dates, so label them 'transaction_date'. Only a date beside a "
        "birth-date label is 'date_of_birth'. The employee's own name is "
        "'name'."
    ),
    "marksheet": (
        "This is a marksheet: its dates are examination, result or admission "
        "dates, so label them 'transaction_date'. Only a date beside a "
        "birth-date label is 'date_of_birth'. The student's own name is 'name'; "
        "a parent's name is 'father_name'."
    ),
}


def detect_system(document_type: str = "other") -> str:
    """Layer-2 system prompt with the document-type preserve note appended."""
    note = DETECT_PRESERVE_NOTES.get(document_type)
    if not note:
        return DETECT_BASE_SYSTEM
    return DETECT_BASE_SYSTEM + " " + note


DETECT_OUTPUT_RULES = (
    "Answer with ONE LINE PER FRAGMENT and nothing else, in exactly this form:\n"
    "<id>: <type>\n"
    "A correct answer looks like:\n"
    "a00: name\n"
    "a01: label_text\n"
    "a02: account_number\n"
    "One id per line: the id first, then ':', then one type from the valid "
    "list. Never repeat the fragment's text, never write JSON, braces, quotes, "
    "numbering, bullets, headings or commentary, and never explain an answer. "
    "Every id you leave out is treated as sensitive and hidden, so label every "
    "id you can, and if you run short of room, stop after a completed line."
)


def detect_user(fragment_lines: str) -> str:
    """Layer-2 user prompt: the fragments plus the line-protocol instructions."""
    return (
        "Fragments to label, one per line as '<id>: <text>':\n"
        + fragment_lines
        + "\n\n"
        + DETECT_OUTPUT_RULES
    )



# --- Layer 5b: Audit ----------------------------------------------------------

AUDIT_SYSTEM = (
    "You audit a redaction result. You have NO knowledge of how these "
    "decisions were made. You see only fragments currently kept visible plus "
    "the sharing purpose. Flag an id ONLY if its text is itself sensitive "
    "(a personal identifier, account or card number, address, phone, email, "
    "date of birth, signature, or a family member's name/number). Plain "
    "labels ('Name:', 'Employer:'), generic words, amounts, dates that are "
    "not a date of birth, and employer or institution names are NOT "
    "sensitive — never flag those. When nothing is sensitive, write "
    "'FLAGGED: none'. " + NO_FORMAT_NOISE
)

AUDIT_OUTPUT_RULES = (
    "Answer with at most two lines and nothing else:\n"
    "FLAGGED: <id>, <id>\n"
    "REASON: <one short sentence>\n"
    "Write exactly 'FLAGGED: none' when nothing is sensitive. Copy fragment "
    "ids exactly as given, never invent an id, and never write JSON."
)


def audit_user(purpose_label: str, fragment_lines: str) -> str:
    """Layer-5b user prompt: the sharing purpose plus the visible fragments."""
    return (
        "Sharing purpose: " + purpose_label + "\n"
        "Fragments currently kept visible, one per line as '<id>: <text>':\n"
        + fragment_lines
        + "\n\n"
        + AUDIT_OUTPUT_RULES
    )


# --- Warmup -------------------------------------------------------------------

WARMUP_SYSTEM = "You reply with one short line of plain text."
WARMUP_USER = "Reply with exactly: ready"

