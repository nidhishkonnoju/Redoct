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
                                needs a salary row to stand on, a name on a
                                ledger row is a counterparty, the value under a
                                printed label is that field, and a relative's
                                name is never the subject's
       -> [5] Policy (no LLM)   deterministic lookup in presets.json:
                                keep_types / partial_types / redact_types
       -> [6] Regex net         PII patterns override any "keep"
       -> [7] LLM audit         FRESH conversation, sees only fully visible
                                fragments — never a partial mask, never the raw
                                value behind one
       -> [8] Repairs + render  label-anchored repair, black boxes with masked
                                reveals, PNG download
```

Layers 1–2 are purpose-agnostic by design, so they are cached per normalized
image (`redact.DETECT_CACHE`): re-running the *same* document under a different
purpose redoes only the free policy lookup and the render. Measured on
`voter_id_card.png`: 17.0 s cold, 7.9 s on the second purpose (Classify and
Detect skipped entirely — the audit still runs).

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
  row's own content decides, never the presence of a label word. Two Plan v3
  rules close the gaps a 3B model leaves on a layout it has not seen: the
  **value-label anchor** re-types a fragment the model left as generic
  `label_text` when the document prints its own field label directly above
  it (an Address block, a PAN / Aadhaar / EPIC number, a Category row)
  — which is what makes `id_verification`'s last-4 reveal actually
  happen; and the **third-party anchor** demotes a `name` the model put on
  a relative's value under "Father's / Mother's / Spouse / Guardian /
  Nominee Name" to `father_name`, because `name` is a keep *and*
  purpose-critical type that would otherwise print the relative and shield
  it from the audit.
- **Regex validator** — Aadhaar / PAN / phone / IFSC / long-digit / email
  patterns override any "keep" decision; catches are flagged in the UI.
- **Context-anchored rules** — e.g. a bare date is not treated as a date of
  birth; a date only becomes DOB-sensitive under a DOB label (the rule looks at
  the label's line *and* the next OCR line, for label/value layouts).
- **Partial masking (Plan v3)** — `id_verification` declares `partial_types`
  (Aadhaar / PAN / voter id, last 4). A `partial` fragment is still covered
  by an opaque black box — the privacy floor is never skipped — and the
  masked string is painted *over* it. That string is pure template
  substitution from the preset (`policy.mask_value()`), never LLM output and
  never the raw fragment, and it fails closed to a bare redaction whenever
  the reveal cannot be honoured: a template carrying literal characters, a
  value too short to hide 4 characters, a box too small to print at the
  minimum font. The audit call is never shown a partial fragment at all —
  showing it *anything* about an already-handled fragment only invites the
  wasted round Redacto documented.
- **Deliberate divergence from the reference architecture (Guiding Principle A)**
  — Redacto's `regex_fallback` module exists but is deliberately not
  wired into their pipeline (they were asked for LLM-only redaction). Ours
  *is* wired in, on purpose: a 3B model on a laptop GPU is not a 4B model on
  a dedicated NPU, the net costs zero latency, and it has already caught a
  real leak live. A considered design choice, not an oversight.
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
- **Partial-failure containment** — Detect runs in 40-fragment batches
  (Plan v3 Priority 0 moved the layer to a line protocol, `a00: name` per
  fragment, and re-measured the batch size: far fewer tokens per fragment
  than the old JSON array, so a batch can hold more, and a cut-off
  response still parses line by line); a batch
  whose output comes back truncated is split and retried, so one bad fragment
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
on a 4 GB card. The layered pipeline makes at most 6 LLM calls per document (Classify +
Detect + up to 2 audit rounds + the detect batch splits); the numbers below
come from `python e2e_acceptance.py --timings`, warm, `num_ctx=4096`:

| Layer | Sample timings | Notes |
|---|---|---|
| Classify | 2.6–4.8 s | output is one word |
| Detect | 3–13 s per call (batched, ≤40 fragments/call) | 1–4 calls: the dominant cost |
| LLM audit | 3–5 s per round (≤2 rounds) | stops early when nothing is flagged |
| OCR + anchors + policy + render | < 1 s | no model involved |

Re-measured after Plan v3 (six real document/purpose pairs, warm, from
`python e2e_acceptance.py --timings`):

| Case | Total | Classify | Detect | Audit | OCR + render |
|---|---|---|---|---|---|
| `pan_card` + ID Verification | 14.5 s | 2.7 s | 7.8 s | 3.6 s | 0.4 s |
| `voter_id_card` + ID Verification | 17.0 s | 2.9 s | 9.8 s | 3.9 s | 0.3 s |
| `bank_statement` + Proof of Income | 28.8 s | 3.7 s | 21.4 s | 2.8 s | 0.8 s |
| `salary_slip` + Proof of Income | 26.6 s | 3.4 s | 15.1 s | 7.6 s | 0.5 s |
| `marksheet` + Education Proof | 48.0 s | 3.4 s | 38.2 s | 5.8 s | 0.6 s |
| `voter_id_card` + Proof of Address (2nd purpose, cached) | 7.9 s | cached | cached | 7.5 s | 0.4 s |

Detect dominates (55–80% of LLM time) and Classify is a single ~3 s call,
so merging Classify into Detect's first batch would save ~1 call on a *cold*
run at the cost of one combined prompt and a worse failure mode (a
mis-classified document type would take the labels down with it). Measured
first, kept separate — and the Detect cache already removes Classify from
the hot re-run path.

End-to-end: **~14 s (ID card) to ~48 s (dense marksheet) cold; ~8 s for a
second purpose on the same document**. Override the
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
5. If time allows: upload `sample_docs/voter_id_card.png` with **ID
   Verification Only** — a layout the model has never seen. The EPIC number
   comes out as `XXXXXXX4567`: a black box with only the preset's declared
   last 4 painted over it (the *"1 fragment(s) partially masked"* warning
   names the count), the name stays, and the father's name, DOB and address
   go black. Then `sample_docs/marksheet.png` with **Education Proof**:
   institution, programme and `CGPA: 8.72` stay visible while the roll
   number, category, father's name, DOB and the registrar's phone go black.
   (~20 s each, and switching purpose on the same document is fast because
   Classify + Detect are cached.)

## Known limitations (by design for Round 1)

- Photo/signature regions are not detected (OCR-only pipeline) — next step.
- Four fixed presets; custom purposes are a `presets.json` edit away.
- Single image per run; no multi-page PDF.
- Latency ~22–60 s per document on a GTX 1650 laptop (warm; first call loads
  the model) for up to 6 LLM calls cold; a second purpose on the same
  document skips Classify + Detect (cached) and lands near the audit cost
  alone. Layer timings are printed by `python redact.py` and, per case, by
  `python e2e_acceptance.py --timings`; `llm.py` records per-layer durations.
- Partial masking reveals exactly the window the preset declares (`last4`). A
  spec that would reveal more than it hides, a value too short to hide 4
  characters, or a box too small to print at the minimum font all fail closed
  to a bare black box — never a clipped or raw value.
- The deterministic anchors only fire on the labels they list (salary rows,
  ledger rows, third-party names, and the Address / PAN / Aadhaar / EPIC /
  Category values). Everything else still relies on the model's label, the
  regex net and the audit — the warnings say when a rule had to step in.
- A ledger row whose layout hides the date and the amount in *different* OCR
  boxes (merged columns) can keep a counterparty name visible; the audit layer
  is the backstop there, and the row is still redacted whenever the amount
  shares the row.

- Native Android/NPU port (LiteRT-LM + Gemma) is the on-site build — see
  *Porting to Android* below.

## Verification: PII visible = none

`python e2e_acceptance.py` runs the real pipeline over every synthetic
document and checks three things per case, against the rendered image rather
than the pipeline's own word:

1. **PII covered** — every fragment the regex / context net recognises as
   PII is behind a drawn box. The check is label-free: it re-derives the PII
   set from the OCR fragments, so a mislabelled fragment cannot fool it.
2. **Privacy floor** — every covered fragment's box is opaque black in the
   output pixels, and every partial box carries exactly one mask, recomputed
   from the preset's own spec (it must reveal the last 4 and never a hidden
   character).
3. **Purpose delivered** (reported, not gated) — the values the purpose
   exists to reveal are still readable.

| Document | Purpose | Result | Purpose values readable | Masked |
|---|---|---|---|---|
| `bank_statement` | Proof of Income | PASS | ARJUN, 85,000.00 | none |
| `pan_card` | ID Verification | PASS | ARJUN | `XXXXXX234F` |
| `salary_slip` | Proof of Income | PASS | ARJUN, 85,000 | none |
| `voter_id_card` | ID Verification | PASS | ARJUN | `XXXXXXX4567` |
| `voter_id_card` | Proof of Address | PASS | Flat, Bengaluru | none |
| `marksheet` | Education Proof | PASS | BENGALURU, 8.72 | none |

6/6 cases -> `PII visible=none`, with the names and figures those purposes
exist to prove still legible, and the two identifiers revealed exactly as
their preset declares.

## Porting to Android (Plan v3 Priority 6 — scoping, not built yet)

| Component | Porting effort | Notes |
|---|---|---|
| `policy.py` (`apply_policy`, `mask_value`) | **Direct port** | Pure logic,
  no I/O: the keep / partial / redact decision and the mask template move as
  they are |
| `presets.json` | **No change** | Bundle as an Android asset |
| `reconstruct_rows()` + anchors (`redact.py`) | **Direct port** | Pure
  geometry maths; only the anchor label lists need reviewing for the on-site
  documents |
| Line protocol + batching (Priority 0) | **Direct port** | Parsing `id: type`
  lines is simpler in Kotlin than JSON bracket matching, and degrades the same
  way on a truncated response |
| `ocr.py` (Tesseract) | **Replace** | Google ML Kit Text Recognition,
  matching Redacto's own choice |
| `llm.py` (Ollama HTTP) | **Replace** | LiteRT-LM's Android SDK. The prompt
  *text* transfers close to as-is because it lives in `prompts.py`, not
  inlined in call logic |
| Render (`redact.py`) | **Replace** | PIL `ImageDraw` -> Android
  Canvas/Bitmap; the mask rules (box -> fit font -> print, else stay blank)
  are a dozen lines of arithmetic |
| `app.py` (Streamlit) | **Full rewrite** | Jetpack Compose, per the PRD |

## Tests

```bash
python -m unittest discover -s tests     # 144 unit tests, no services needed:
                                         # policy lookup (every type x preset,
                                         # keep / partial / redact precedence),
                                         # mask_value fail-closed rules, mask
                                         # rendering + audit-input guard, value
                                         # label + third-party name anchors,
                                         # salary-amount and counterparty
                                         # anchors, detect batching + detect
                                         # cache + partial-failure containment,
                                         # father-name + label-vs-value
                                         # regressions, audit guardrail, regex
                                         # net, context rules, row
                                         # reconstruction, line protocol,
                                         # fail-closed accounting, repairs
python ocr.py            # OCR smoke: word table + count on a sample doc
python llm.py            # Ollama health + one call per layer + timing metrics
python redact.py         # full layered pipeline probe + per-layer call metrics
python e2e_acceptance.py # REAL end-to-end acceptance (Tesseract + Ollama, no
                         # stubs) over the synthetic documents: asserts PII
                         # visible=none, that every covered fragment sits
                         # behind an opaque box in the output pixels, and that
                         # every partial mask is preset-derived and bounded.
                         # --timings adds the latency table, --list the cases
```

`sample_docs/generate_samples.py` regenerates the synthetic mock documents
(PIL-drawn, clean and OCR-friendly, all data fake): `bank_statement`,
`pan_card`, `salary_slip`, `voter_id_card`, `marksheet`.
