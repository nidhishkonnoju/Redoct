"""Streamlit UI — purpose-based document redaction, fully local.

Run:  streamlit run app.py
"""
from __future__ import annotations

import io

import streamlit as st
from PIL import Image

import llm
from config import OLLAMA_MODEL
from redact import load_presets, run_pipeline

st.set_page_config(page_title="Purpose-Based Document Redaction", layout="wide")


@st.cache_data(show_spinner=False)
def cached_presets() -> dict:
    return load_presets()


def resolve_image(camera_file, upload_file) -> Image.Image | None:
    """Camera capture wins when both inputs are present."""
    source = camera_file or upload_file
    if source is None:
        return None
    return Image.open(source).convert("RGB")


def render_sidebar(presets: dict) -> tuple[str, dict]:
    st.sidebar.title("Sharing Purpose")
    labels = {p["label"]: key for key, p in presets.items()}
    label = st.sidebar.radio(
        "Why are you sharing this document?", list(labels.keys())
    )
    key = labels[label]

    st.sidebar.divider()
    ok, message = llm.check_ollama()
    if ok:
        st.sidebar.success(message)
    else:
        st.sidebar.error(message)

    st.sidebar.divider()
    st.sidebar.caption(
        "Everything runs on this machine: Tesseract OCR + a local LLM via "
        f"Ollama ({OLLAMA_MODEL}). No image or text ever leaves the device."
    )
    return key, presets[key]


def render_results(before: Image.Image, result, preset_label: str) -> None:
    left, right = st.columns(2)
    with left:
        st.subheader("Original")
        st.image(before, use_container_width=True)
    with right:
        st.subheader(f"Redacted — {preset_label}")
        st.image(result.output_image, use_container_width=True)

    buf = io.BytesIO()
    result.output_image.save(buf, format="PNG")
    st.download_button(
        "Download redacted image (PNG)",
        data=buf.getvalue(),
        file_name="redacted.png",
        mime="image/png",
    )

    st.caption(
        f"Classified as **{result.document_type}** in {result.elapsed_s:.1f}s — "
        "layered pipeline: classify → detect → label anchors → policy lookup → "
        "regex net → fresh-context LLM audit."
    )

    if result.validator_hits:
        caught = ", ".join(
            f"`{h.pattern}` → {h.matched_text}" for h in result.validator_hits
        )
        st.warning(f"🛡️ Auto-caught by the regex safety net: {caught}")
    for warning in result.warnings:
        st.warning(warning)


def main() -> None:
    st.title(" Purpose-Based Document Redaction")
    st.markdown(
        "Pick **why** you're sharing a document — the app decides what must "
        "stay visible and blacks out everything else, locally."
    )

    try:
        presets = cached_presets()
    except RuntimeError as exc:
        st.error(str(exc))
        return

    purpose_key, preset = render_sidebar(presets)

    left, right = st.columns(2)
    with left:
        camera_file = st.camera_input("Capture a document")
    with right:
        upload_file = st.file_uploader(
            "…or upload an image", type=["png", "jpg", "jpeg", "webp"]
        )

    image = resolve_image(camera_file, upload_file)
    if image is None:
        st.info("Capture or upload a document to begin.")
        return

    if not st.button("Redact", type="primary"):
        return

    ok, message = llm.check_ollama()
    if not ok:
        st.error(message)
        return

    try:
        with st.spinner("Running local OCR + LLM pipeline…"):
            result = run_pipeline(image, purpose_key, preset)
    except RuntimeError as exc:
        st.error(str(exc))
        return

    render_results(image, result, preset["label"])


if __name__ == "__main__":
    main()
