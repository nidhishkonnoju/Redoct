# Purpose-Based Document Redaction (Round 1 Prototype)

Local, private, purpose-based document redaction: pick **why** you're sharing
a document (Proof of Income, ID Verification Only, …) and the app decides
what stays visible and blacks out the rest.

**Nothing leaves this machine** — Tesseract OCR + a local LLM served by
Ollama. No cloud, no API keys, no telemetry.

## Pipeline

```
Image -> [1] OCR (pytesseract, word-level boxes)
       -> [2] Classify+Detect (one structured-JSON call to a local LLM)
       -> [3] Render (solid black boxes over redacted words)
       -> [4] Validate (regex safety net force-redacts anything the LLM missed)
       -> Redacted image (before/after view + PNG download)
```

Safety nets (in `redact.py`):
- **Fail-closed accounting** — any word the LLM fails to classify is REDACTed.
- **Regex validator** — Aadhaar / PAN / phone / IFSC / long-digit / email
  patterns override any "keep" decision; catches are flagged in the UI.
- **Context-anchored rules** — e.g. a date is sensitive only under a
  "Date of Birth" label, so statement dates survive while DOB does not.
- **Label-anchored repair** — salary values and the account-holder name on
  labelled lines are re-kept deterministically, so the one value the purpose
  exists to reveal is never lost to blanket over-redaction. PII pattern
  matches are never un-redacted, and third-party name lines are skipped.

## Prerequisites

```bash
winget install --id UB-Mannheim.TesseractOCR -e   # Tesseract 5.x
# Ollama: https://ollama.com/download  (store on D: on this machine)
ollama pull llama3.2:3b

pip install -r requirements.txt
```

Model choice (`config.py`): `llama3.2:3b` is the benchmarked default — 100% GPU
on a 4 GB card, terse output ⇒ 10–27 s warm per document, correct/stable
decisions on all three sample docs. Override without code changes:

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
   IFSC, balance and other transactions go black. (~35 s)
3. Upload `sample_docs/pan_card.png`, pick **ID Verification Only**. Name and
   photo placeholder stay; PAN number, DOB and father's name go black. **This is
   the validator moment**: the LLM kept `14/08/1999` (a bare date doesn't look
   sensitive), and the "🛡️ Auto-caught by the regex safety net:
   `date_of_birth`" banner names the pattern it caught. (~30 s)
4. Upload `sample_docs/salary_slip.png`, pick **Proof of Income**. Note the
   "label-anchored rule" warning: the LLM had redacted *every* number, and the
   deterministic repair put the salary values back while PII stayed hidden.
   Download the PNG. (~25 s)

## Known limitations (by design for Round 1)

- Photo/signature regions are not detected (OCR-only pipeline) — next step.
- Four fixed presets; custom purposes are a `presets.json` edit away.
- Single image per run; no multi-page PDF.
- Latency ~10–26 s per document on a GTX 1650 laptop (warm; first call loads
  the model), within the Round 1 demo budget.

- Native Android/NPU port (LiteRT-LM + Gemma) is the on-site build.

## Tests

```bash
python -m unittest discover -s tests -v   # 28 unit tests, no services needed:
                                          # regex net, context rules, fail-closed
                                          # accounting, label-anchored repair
python ocr.py        # OCR smoke: word table + count on a sample doc
python llm.py        # Ollama health + one real classify call + timing metrics
python redact.py     # full pipeline probe + LLM call metrics breakdown
```

`sample_docs/generate_samples.py` regenerates the synthetic mock documents
(PIL-drawn, clean and OCR-friendly — all data is fake).
