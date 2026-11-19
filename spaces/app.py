"""
spaces/app.py
=============
Gradio demo for HuggingFace Spaces — AccessiCaption Accessibility Image Captioning.

Loads the fine-tuned LLaVA model from HuggingFace Hub and provides an
interactive interface for generating WCAG-compliant alt-text captions.
"""

from __future__ import annotations

import logging
import os
import sys

# Allow relative import of the inference module when running from the
# spaces/ directory (adjusts the Python path to include the repo root).
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import gradio as gr

from inference.captioner import Captioner, CaptionOptions, CaptionResult

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HF_MODEL_ID: str = os.getenv(
    "HF_MODEL_ID", "YOUR_HF_USERNAME/accessicaption-llava-qlora"
)
DEVICE: str = os.getenv("DEVICE", "auto")
LOAD_IN_4BIT: bool = os.getenv("LOAD_IN_4BIT", "true").lower() == "true"

# ---------------------------------------------------------------------------
# Model singleton
# ---------------------------------------------------------------------------

_captioner: Captioner | None = None


def get_captioner() -> Captioner:
    """Return (or lazily initialise) the global Captioner instance."""
    global _captioner
    if _captioner is None:
        logger.info("Loading model: %s …", HF_MODEL_ID)
        _captioner = Captioner(
            model_path=HF_MODEL_ID,
            device=DEVICE,
            load_in_4bit=LOAD_IN_4BIT,
        )
        _captioner.warmup()
        logger.info("Model ready.")
    return _captioner


# ---------------------------------------------------------------------------
# Inference function
# ---------------------------------------------------------------------------


def generate_caption(
    image,
    confidence_threshold: float,
    max_new_tokens: int,
    temperature: float,
) -> tuple[str, float, str]:
    """
    Gradio callback — run inference and return (caption, confidence, warning).

    Parameters
    ----------
    image:
        PIL image from the Gradio Image component.
    confidence_threshold:
        Slider value from the UI.
    max_new_tokens:
        Slider value from the UI.
    temperature:
        Slider value from the UI.

    Returns
    -------
    tuple[str, float, str]
        ``(caption_text, confidence_score, warning_message)``
    """
    if image is None:
        return "", 0.0, "⚠️ Please upload an image."

    captioner = get_captioner()
    options = CaptionOptions(
        confidence_threshold=confidence_threshold,
        max_new_tokens=int(max_new_tokens),
        temperature=temperature,
    )

    try:
        result: CaptionResult = captioner.caption(image, options)
    except Exception as exc:  # noqa: BLE001
        logger.error("Inference error: %s", exc, exc_info=True)
        return "", 0.0, f"❌ Error during inference: {exc}"

    warning = ""
    if result.flagged:
        warning = (
            f"⚠️ Low confidence ({result.confidence:.0%}). "
            "The caption may be inaccurate — please review before use."
        )

    return result.caption, result.confidence, warning


# ---------------------------------------------------------------------------
# Example images (diverse, publicly accessible)
# ---------------------------------------------------------------------------

EXAMPLES = [
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        0.75, 80, 0.2,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/1/14/Gatto_europeo4.jpg/320px-Gatto_europeo4.jpg",
        0.75, 80, 0.2,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/a/a7/Camponotus_flavomarginatus_ant.jpg/320px-Camponotus_flavomarginatus_ant.jpg",
        0.75, 80, 0.2,
    ],
    [
        "https://upload.wikimedia.org/wikipedia/commons/thumb/6/6d/Good_Food_Display_-_NCI_Visuals_Online.jpg/320px-Good_Food_Display_-_NCI_Visuals_Online.jpg",
        0.75, 80, 0.2,
    ],
]

# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

_DESCRIPTION = """
# AccessiCaption — Accessibility Image Captioning

**AccessiCaption** generates WCAG 2.1–compliant alt-text descriptions for
images, designed to assist visually impaired users who rely on screen readers.

### How it works
1. Upload an image (or choose one of the examples below).
2. Adjust the generation parameters if needed.
3. Click **Generate Caption** — the model produces a concise, human-readable
   description of the image content.

### WCAG compliance
Generated captions follow WCAG 2.1 Success Criterion 1.1.1 (Non-text Content):
- Descriptions are **concise** (1–2 sentences).
- They **do not** begin with "Image of" or "Photo of".
- They focus on the **main subject and context**.
- They include any **important visible text**.

> **Note:** Confidence scores below 75 % are flagged for human review.
"""

with gr.Blocks(
    title="AccessiCaption — Accessibility Image Captioning",
    theme=gr.themes.Soft(primary_hue="blue"),
) as demo:
    gr.Markdown(_DESCRIPTION)

    with gr.Row():
        # --- Left column: inputs ---
        with gr.Column(scale=1):
            image_input = gr.Image(
                type="pil",
                label="Upload Image",
                elem_id="image-upload",
            )

            with gr.Accordion("Advanced Options", open=False):
                confidence_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.75,
                    step=0.05,
                    label="Confidence Threshold",
                    info="Results below this threshold are flagged for review.",
                )
                tokens_slider = gr.Slider(
                    minimum=20,
                    maximum=200,
                    value=80,
                    step=10,
                    label="Max New Tokens",
                    info="Maximum length of the generated caption.",
                )
                temperature_slider = gr.Slider(
                    minimum=0.0,
                    maximum=1.0,
                    value=0.2,
                    step=0.05,
                    label="Temperature",
                    info="Higher values → more creative but less deterministic output.",
                )

            generate_btn = gr.Button("Generate Caption", variant="primary")

        # --- Right column: outputs ---
        with gr.Column(scale=1):
            caption_output = gr.Textbox(
                label="Alt-Text Caption",
                placeholder="Your WCAG-compliant alt-text will appear here …",
                lines=4,
                show_copy_button=True,
            )

            confidence_output = gr.Slider(
                minimum=0.0,
                maximum=1.0,
                value=0.0,
                label="Confidence Score",
                interactive=False,
                info="Model confidence in the generated caption (0 = low, 1 = high).",
            )

            warning_output = gr.Markdown(value="", visible=True)

    # Wire up the button
    generate_btn.click(
        fn=generate_caption,
        inputs=[
            image_input,
            confidence_slider,
            tokens_slider,
            temperature_slider,
        ],
        outputs=[caption_output, confidence_output, warning_output],
    )

    # Also trigger on image upload for convenience
    image_input.change(
        fn=generate_caption,
        inputs=[
            image_input,
            confidence_slider,
            tokens_slider,
            temperature_slider,
        ],
        outputs=[caption_output, confidence_output, warning_output],
    )

    gr.Examples(
        examples=EXAMPLES,
        inputs=[image_input, confidence_slider, tokens_slider, temperature_slider],
        outputs=[caption_output, confidence_output, warning_output],
        fn=generate_caption,
        cache_examples=False,
        label="Example Images",
    )

    gr.Markdown(
        """
---
Built with ❤️ for accessibility · Powered by [LLaVA](https://llava-vl.github.io/) · 
[WCAG 2.1 Guidelines](https://www.w3.org/TR/WCAG21/)
        """
    )

# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
        share=False,
    )
