# Purpose-Based Document Redaction (Round 1 Prototype)

Local, private, purpose-based document redaction: pick **why** you're sharing
a document (Proof of Income, ID Verification Only, …) and the app decides
what stays visible and blacks out the rest.

**Nothing leaves this machine** — Tesseract OCR + a local LLM served by
Ollama. No cloud, no API keys, no telemetry.

## Pipeline (Plan v2 — layered, Redacto-baseline architecture)

```
Image -> [1] OCR                pytesseract, word-level boxes + stable ids
       -> [2] Classify (LLM)    FRESH call: document_type only, no purpose info
       -> [3] Detect (LLM)      FRESH call: label every fragment with a FIELD TYPE
                                (name, account_number, father_name, ...). The
                                Detect call never decides visibility.
       -> [4] Anchors (no LLM)  deterministic label refinement: a `salary_amount`
                                needs a salary row to stand on, and a name on a
                                ledger row is a counterparty, not the subject
       -> [5] Policy (no LLM)   deterministic lookup in presets.json:
                                keep_types / redact_types -> redact_ids
       -> [6] Regex net         PII patterns override any "keep"
       -> [7] LLM audit         FRESH conversation, sees only visible fragments
       -> [8] Repairs + render  label-anchored repair, black boxes, PNG download
```

Why the split: a single LLM call was doing *classify + detect + decide
sensitivity* at once, so "is this a name?" and "should this be visible?" got
tangled. Now the LLM only ever answers *what a fragment is*; **visibility is a
dictionary lookup** (`presets.json`), so it is consistent, explainable and
testable with zero model calls.

Safety nets:
- **Fail-closed accounting** — any word the Detect layer never labelled is
  REDACTED. A type outside `keep_types` is REDACTED. Redact always beats keep.
- **Deterministic label anchors** — the model labels, the anchors correct the
  drift a 3B model shows on ledgers. A `salary_amount` counts only on a salary
  row (`SALARY CREDIT`, `GROSS`, `NET PAY`, `BASIC`, …), so rent, card purchases
  and the closing balance are demoted to `other_amount` and redacted instead of
  leaking the spending pattern. A `name` on a ledger row — a money value plus a
  date or a transfer keyword (`PURCHASE`, `NEFT/`, `UPI/`) — is a counterparty
  and is demoted too, while a name on its own row (a PAN card prints the label
  and the value on separate rows) keeps its purpose-critical label. Only the
  row's own content decides, never the presence of a label word.
- **Regex validator** — Aadhaar / PAN / phone / IFSC / long-digit / email
  patterns override any "keep" decision; catches are flagged in the UI.
- **Context-anchored rules** — e.g. a bare date is not treated as a date of
  birth; a date only becomes DOB-sensitive under a DOB label (the rule looks at
  the label's line *and* the next OCR line, for label/value layouts).
- **LLM audit + policy guardrail** — a second, independent LLM pass reviews
  only the fragments still visible. It may tighten the leftovers, but the
  deterministic policy keeps final say on `purpose_critical_types` (the fields
  the stated purpose exists to reveal): the audit can never re-hide the
  account-holder's name or the salary figure it was asked to prove. Overrides
  are counted in the UI warnings.
- **Label-anchored repair** — salary values and the account-holder name on
  labelled lines are re-kept deterministically, so the one value the purpose
  exists to reveal is never lost to blanket over-redaction. PII pattern
  matches are never un-redacted, and third-party name lines are skipped.
- **Partial-failure containment** — Detect runs in 25-fragment batches; a batch
  whose JSON comes back truncated is split and retried, so one bad fragment
  costs a few ids (fail-closed to REDACT, and said so in the warnings) instead
  of the whole document. A dead Ollama server still fails closed everywhere.

## Prerequisites

```bash
winget install --id UB-Mannheim.TesseractOCR -e   # Tesseract 5.x
# Ollama: https://ollama.com/download  (store on D: on this machine)
ollama pull llama3.2:3b

pip install -r requirements.txt
```

Model choice (`config.py`): `llama3.2:3b` is the benchmarked default — 100% GPU
on a 4 GB card. The layered pipeline runs up to 4 LLM calls per document
(measurements on the three sample docs, warm, `num_ctx=4096`):

| Layer | Sample timings | Notes |
|---|---|---|
| Classify | 2.6–4.8 s | output is one word |
| Detect | 3–13 s per call (batched, ≤25 fragments/call) | 1–4 calls: the dominant cost |
| LLM audit | 3–5 s per round (≤2 rounds) | stops early when nothing is flagged |
| OCR + anchors + policy + render | < 1 s | no model involved |

End-to-end: **~22 s (ID card) to ~60 s (dense bank statement)**. Override the
model without code changes:

```bash
set REDACT_MODEL=qwen2.5:3b     # or gemma2:2b / qwen2.5:1.5b (faster, weaker)
```

Requests run at `temperature=0` with a fixed `seed`, so the same document yields
the same decision on every run — important when demoing live.

## Run

```bash
cd redact-app
streamlit run app.py
```

Use **sample docs** (`sample_docs/*.png` — all fake data) via the upload pane,
or capture from the webcam.

## 60–90 second demo script

1. Show the sidebar: purpose presets + "local only" status. (~10 s)
2. Upload `sample_docs/bank_statement.png`, pick **Proof of Income**, hit
   Redact. Point out: classification reads `bank_statement`, the account-holder
   name and the `SALARY CREDIT` amount stay visible, while the account number,
   IFSC, balance and every other transaction go black. Two warnings worth
   reading aloud: *"N money value(s) without a salary anchor were demoted"* and
   *"N ledger-row name(s) were treated as counterparties and redacted"* — the
   model had called the rent, the card purchases and the closing balance
   `salary_amount`, and the shops `name`; the deterministic anchors corrected
   both before any policy lookup. Then *"Policy guardrail overrode N audit
   flag(s) on purpose-critical field(s)"*: the independent audit tried to hide
   the name too, and the deterministic policy refused. (~60 s)
3. Upload `sample_docs/pan_card.png`, pick **ID Verification Only**. Name and
   photo placeholder stay; PAN number, DOB and father's name go black. Worth
   pointing out here: the name *value* is on its own OCR row (`ARJUN MEHTA`),
   one row below the `Name` label — the row-content anchors leave it alone, and
   the audit round can only flag the leftover label words. Whenever something
   the regex net knows (`date_of_birth`, `long_digit`, …) survives every earlier
   layer, the "🛡️ Auto-caught by the regex safety net: `pattern` → value" banner
   names it: that is the deterministic floor under the model. (~22 s)
4. Upload `sample_docs/salary_slip.png`, pick **Proof of Income**. Read the two
   deterministic warnings: *"N money value(s) without a salary anchor were
   demoted"* (the HRA/deduction rows are not income figures) and *"Repaired N
   purpose-critical value(s) to stay visible (label-anchored rule)"* — the LLM
   had redacted the gross/net figures, and the repair put them back while PII
   stayed hidden. Download the PNG. (~50 s)

## Known limitations (by design for Round 1)

- Photo/signature regions are not detected (OCR-only pipeline) — next step.
- Four fixed presets; custom purposes are a `presets.json` edit away.
- Single image per run; no multi-page PDF.
- Latency ~22–60 s per document on a GTX 1650 laptop (warm; first call loads
  the model) for up to 6 LLM calls. Layer timings are printed by
  `python redact.py`; `llm.py` also records per-layer durations.
- A ledger row whose layout hides the date and the amount in *different* OCR
  boxes (merged columns) can keep a counterparty name visible; the audit layer
  is the backstop there, and the row is still redacted whenever the amount
  shares the row.

- Native Android/NPU port (LiteRT-LM + Gemma) is the on-site build.

## Tests

```bash
python -m unittest discover -s tests -v   # 55 unit tests, no services needed:
                                          # policy lookup (every type x preset),
                                          # label anchors (salary amounts,
                                          # counterparty names), detect batching
                                          # + partial-failure containment,
                                          # father-name + label-vs-value
                                          # regressions, audit guardrail,
                                          # regex net, context rules,
                                          # fail-closed accounting, repairs
python ocr.py        # OCR smoke: word table + count on a sample doc
python llm.py        # Ollama health + one real call per layer + timing metrics
python redact.py     # full layered pipeline probe + per-layer call metrics
```

`sample_docs/generate_samples.py` regenerates the synthetic mock documents
(PIL-drawn, clean and OCR-friendly — all data is fake).
