"""Shared configuration for the redaction prototype.

All knobs live here so on-site changes (model swap, OCR tuning, latency
budget) are edits to one file, not code hunts.
"""
from __future__ import annotations

import os
from pathlib import Path

# --- Ollama -----------------------------------------------------------------
# llama3.2:3b: benchmarked best on this hardware — terse output (175-250
# tokens => 10-26 s warm, 100% GPU on the 4 GB card) and correct decisions
# with all sensitive fields redacted on all three sample docs.
# Alternatives via env (no code change):
#   REDACT_MODEL=qwen2.5:3b    (same quality, ~3x slower: verbose output)
#   REDACT_MODEL=qwen2.5:1.5b  (fast but unstable keep/redact split)
OLLAMA_BASE_URL: str = os.environ.get("REDACT_OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL: str = os.environ.get("REDACT_MODEL", "llama3.2:3b")
OLLAMA_TIMEOUT_S: float = 180.0  # covers cold-load spikes on a 4 GB GPU
OLLAMA_KEEP_ALIVE: str = "30m"  # keep the model resident between demo runs
# 4096 ctx: prompts here are ~100 short fragments; smaller KV cache lets the
# whole model + cache fit in 4 GB VRAM (full GPU) instead of splitting to CPU.
OLLAMA_OPTIONS: dict = {
    "temperature": 0,
    "seed": 0,  # fixed seed => same document gives the same decision every run
    "num_ctx": 4096,  # fits model + KV cache in 4 GB VRAM (stays on GPU)
    "num_predict": 1024,  # detect batches are small; 1024 bounds each generate
}

# --- Tesseract ---------------------------------------------------------------
# Explicit override wins; otherwise ocr.configure_tesseract() auto-detects.
TESSERACT_CMD: str | None = os.environ.get("REDACT_TESSERACT") or None
OCR_TESSERACT_CONFIG: str = "--psm 3"
OCR_MIN_CONF: int = 30  # tesseract confidence floor (0-100)

# --- Image normalization (single coordinate space for OCR + rendering) ------
OCR_MIN_SIDE_PX: int = 1000   # upscale if the shorter edge is below this
OCR_MAX_SIDE_PX: int = 3500   # downscale if the longer edge exceeds this

# --- Redaction rendering -----------------------------------------------------
REDACT_PADDING_PX: int = 5

# --- Paths -------------------------------------------------------------------
PROJECT_ROOT: Path = Path(__file__).resolve().parent
PRESETS_PATH: Path = PROJECT_ROOT / "presets.json"
SAMPLE_DOCS_DIR: Path = PROJECT_ROOT / "sample_docs"
