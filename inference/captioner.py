"""
inference/captioner.py
======================
Wraps the fine-tuned LLaVA model for single-image and batch inference,
producing WCAG-compliant alt-text captions with confidence estimation.
"""

from __future__ import annotations

import io
import logging
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Union

import torch
import PIL.Image
from PIL import Image, ImageOps

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt templates per detail level
# ---------------------------------------------------------------------------

PROMPTS = {
    # Layer 1: Concise alt-text with strict spatial grounding and anti-hallucination rules
    "baseline": (
        "USER: <image>\n"
        "You are an accessibility visual assistant. Describe this image in one concise, strictly factual alt-text sentence for a visually impaired user.\n"
        "Crucial rules:\n"
        "- Describe only what is unambiguously visible. Do not guess or assume objects that are not clearly identifiable.\n"
        "- Verify physical support surfaces: accurately state what is below/beneath versus on top of structures.\n"
        "- Mention all distinct people and primary subjects present.\n"
        "- Do not start with 'Image of' or 'Photo of'.\n"
        "ASSISTANT:"
    ),
    # Layer 2: Detailed scene description covering visible people, objects, and surroundings
    "detailed": (
        "USER: <image>\n"
        "You are an accessibility visual assistant. Provide a comprehensive, multi-sentence scene description for a visually impaired user. "
        "Write at least three complete sentences:\n"
        "1. Describe the primary subjects and any key objects resting near them.\n"
        "2. Describe all visible people, their positions (foreground, midground, background), and what they are doing.\n"
        "3. Describe the surrounding environment, physical setting (e.g. water, ground, architectural structures), and lighting.\n"
        "Do not start with 'Image of' or 'Photo of'.\n"
        "ASSISTANT:"
    ),
    # Layer 3: Comprehensive progressive disclosure (Alt-Text, Scene, Spatial Layout)
    "layered": (
        "USER: <image>\n"
        "Analyze this image for accessibility and provide a 3-layer breakdown:\n"
        "Layer 1 (Alt-Text): 1 factual sentence with verified spatial relationships.\n"
        "Layer 2 (Detailed Scene): 2-3 sentences describing all confirmed people, items, and environmental context.\n"
        "Layer 3 (Spatial Layout): Specific breakdown of Foreground, Midground, and Background.\n"
        "Describe strictly what is visually present without guessing.\n"
        "ASSISTANT:"
    ),
}

LLAVA_PROMPT = PROMPTS["baseline"]

# Target size for LLaVA vision encoder
TARGET_SIZE = 336


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class CaptionOptions:
    """Runtime options for a captioning call."""

    confidence_threshold: float = 0.75
    """Minimum confidence below which the result is flagged."""

    max_new_tokens: int = 100
    """Maximum number of new tokens to generate."""

    temperature: float = 0.2
    """Sampling temperature (lower → more deterministic)."""

    detail_level: str = "baseline"
    """Detail tier: 'baseline' (1-sentence alt-text), 'detailed' (2-3 sentences), or 'layered' (3 tiers)."""


@dataclass
class CaptionResult:
    """Output of a single captioning call."""

    caption: str
    """WCAG-normalized alt-text string."""

    confidence: float
    """Confidence score in [0, 1] derived from logit entropy."""

    flagged: bool
    """True when confidence < CaptionOptions.confidence_threshold."""

    latency_ms: float
    """Wall-clock inference time in milliseconds."""

    detail_level: str = "baseline"
    """Detail level used to generate this caption."""

    layers: dict[str, str] = field(default_factory=dict)
    """Dictionary of individual layers when detail_level='layered'."""


# ---------------------------------------------------------------------------
# WCAG normalization (inline fallback if data.wcag_normalize is unavailable)
# ---------------------------------------------------------------------------

def _wcag_normalize(text: str, max_chars: int | None = 150) -> str:
    """
    Apply lightweight WCAG 2.1 alt-text normalization.
    """
    import re

    text = text.strip()

    # Remove known forbidden prefixes (case-insensitive)
    forbidden_prefixes = (
        r"^(image of|photo of|picture of|this is an? |this image shows?|"
        r"this photo shows?|a photo of|a picture of|an image of)\s*"
    )
    text = re.sub(forbidden_prefixes, "", text, flags=re.IGNORECASE)

    # Capitalise first character
    if text:
        text = text[0].upper() + text[1:]

    # Collapse multiple whitespace
    text = re.sub(r"\s+", " ", text)

    # Ensure terminal punctuation
    if text and text[-1] not in {".", "!", "?"}:
        text += "."

    # Truncate safely at word boundary if max_chars is specified
    if max_chars is not None and len(text) > max_chars:
        text = text[: max_chars - 3].rsplit(" ", 1)[0] + "…"

    return text


try:
    from data.wcag_normalize import normalize_caption as wcag_normalize  # type: ignore
except ImportError:
    wcag_normalize = _wcag_normalize


# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------


def _pad_to_square(image: Image.Image) -> Image.Image:
    """Pad *image* to square with black borders, then resize to TARGET_SIZE."""
    image = ImageOps.pad(image, (TARGET_SIZE, TARGET_SIZE), color=(0, 0, 0))
    return image


def load_image(source: Union[str, bytes, Path, Image.Image]) -> Image.Image:
    """
    Load a PIL image from various source types.

    Parameters
    ----------
    source:
        - ``PIL.Image.Image`` — returned as-is (converted to RGB).
        - ``bytes``           — decoded from raw bytes.
        - ``str`` / ``Path`` — interpreted as a URL if it starts with
          ``http``; otherwise treated as a filesystem path.

    Returns
    -------
    PIL.Image.Image
        RGB image resized/padded to 336×336.
    """
    if isinstance(source, Image.Image):
        img = source.convert("RGB")
    elif isinstance(source, bytes):
        img = Image.open(io.BytesIO(source)).convert("RGB")
    elif isinstance(source, (str, Path)):
        source_str = str(source)
        if source_str.startswith("http://") or source_str.startswith("https://"):
            with urllib.request.urlopen(source_str) as resp:  # noqa: S310
                img = Image.open(io.BytesIO(resp.read())).convert("RGB")
        else:
            img = Image.open(source_str).convert("RGB")
    else:
        raise TypeError(f"Unsupported image source type: {type(source)}")

    return _pad_to_square(img)


# ---------------------------------------------------------------------------
# Captioner
# ---------------------------------------------------------------------------


class Captioner:
    """
    Thin wrapper around a fine-tuned LLaVA model for accessibility captioning.

    Parameters
    ----------
    model_path:
        Local directory path **or** HuggingFace Hub model ID.
    device:
        ``"auto"`` lets ``accelerate`` decide; pass ``"cpu"``, ``"cuda"``,
        or ``"cuda:0"`` etc. to override.
    load_in_4bit:
        When *True*, load the model in 4-bit via ``bitsandbytes`` to reduce
        GPU memory usage (requires ``bitsandbytes >= 0.41``).
    """

    MODEL_VERSION: str = "accessicaption-v1"

    def __init__(
        self,
        model_path: str,
        device: str = "auto",
        load_in_4bit: bool = True,
    ) -> None:
        self.model_path = model_path
        self.device_arg = device
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._processor = None
        self._device: torch.device | None = None

        self._load_model()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_model(self) -> None:
        """Lazy-load the LLaVA model and processor."""
        try:
            from transformers import (  # type: ignore
                LlavaForConditionalGeneration,
                AutoProcessor,
                BitsAndBytesConfig,
            )
        except ImportError as exc:
            raise ImportError(
                "transformers>=4.37 and bitsandbytes are required for Captioner."
            ) from exc

        logger.info("Loading LLaVA model from %s …", self.model_path)

        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
            )

        self._model = LlavaForConditionalGeneration.from_pretrained(
            self.model_path,
            quantization_config=quantization_config,
            device_map=self.device_arg,
            torch_dtype=torch.float16,
        )
        self._model.eval()

        self._processor = AutoProcessor.from_pretrained(self.model_path)

        # Resolve the actual device (first parameter device)
        self._device = next(self._model.parameters()).device
        logger.info("Model loaded on device: %s", self._device)

    def _build_inputs(self, image: Image.Image, detail_level: str = "baseline") -> dict:
        """Tokenise *prompt* + *image* into model inputs."""
        prompt = PROMPTS.get(detail_level, PROMPTS["baseline"])
        inputs = self._processor(
            text=prompt,
            images=image,
            return_tensors="pt",
        )
        return {k: v.to(self._device) for k, v in inputs.items()}

    def _estimate_confidence(self, logits: torch.Tensor) -> float:
        """
        Estimate generation confidence from logit entropy.
        """
        from inference.confidence import estimate_confidence_logit_variance  # type: ignore

        logits_list = [logits[i] for i in range(logits.shape[0])]
        return estimate_confidence_logit_variance(logits_list)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def caption(
        self,
        image: Union[str, bytes, Path, Image.Image],
        options: CaptionOptions | None = None,
    ) -> CaptionResult:
        """
        Generate a WCAG-compliant alt-text caption for a single image.

        Parameters
        ----------
        image:
            Anything accepted by :func:`load_image`.
        options:
            Runtime generation options; defaults to :class:`CaptionOptions`.

        Returns
        -------
        CaptionResult
        """
        if options is None:
            options = CaptionOptions()

        pil_image = load_image(image)
        t0 = time.perf_counter()

        level = options.detail_level if options.detail_level in PROMPTS else "baseline"
        inputs = self._build_inputs(pil_image, detail_level=level)

        # Adapt token budget & truncation based on detail level
        if level == "baseline":
            max_tokens = min(options.max_new_tokens, 100)
            min_tokens = 10
            max_chars = 180
        elif level == "detailed":
            max_tokens = max(options.max_new_tokens, 180)
            min_tokens = 45  # forces comprehensive multi-sentence generation
            max_chars = 800
        else:  # layered
            max_tokens = max(options.max_new_tokens, 260)
            min_tokens = 70  # forces multi-layer progressive output
            max_chars = None

        with torch.inference_mode():
            output = self._model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                min_new_tokens=min_tokens,
                temperature=options.temperature,
                do_sample=options.temperature > 0,
                return_dict_in_generate=True,
                output_scores=True,
            )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Decode only newly generated tokens
        input_len = inputs["input_ids"].shape[-1]
        generated_ids = output.sequences[0, input_len:]
        raw_caption = self._processor.decode(generated_ids, skip_special_tokens=True)
        caption = wcag_normalize(raw_caption.strip(), max_chars=max_chars)

        # Confidence from stacked scores
        if output.scores:
            stacked = torch.stack(output.scores, dim=0)  # (gen_len, vocab)
            confidence = self._estimate_confidence(stacked)
        else:
            confidence = 0.0

        layers_dict = {}
        if level == "layered":
            layers_dict["full_text"] = caption

        return CaptionResult(
            caption=caption,
            confidence=confidence,
            flagged=confidence < options.confidence_threshold,
            latency_ms=latency_ms,
            detail_level=level,
            layers=layers_dict,
        )

    def caption_multilevel(
        self,
        image: Union[str, bytes, Path, Image.Image],
    ) -> dict[str, str]:
        """
        Convenience method to return progressive disclosure layers:
        - Layer 1 (Baseline Alt-Text): Concise, strictly grounded alt-text.
        - Layer 2 (Detailed Description): Full scene breakdown including people & context.
        """
        res_baseline = self.caption(image, CaptionOptions(detail_level="baseline"))
        res_detailed = self.caption(image, CaptionOptions(detail_level="detailed"))
        return {
            "layer_1_baseline_alt_text": res_baseline.caption,
            "layer_2_detailed_scene": res_detailed.caption,
        }

    def caption_batch(
        self,
        images: list[Union[str, bytes, Path, Image.Image]],
        options: CaptionOptions | None = None,
    ) -> list[CaptionResult]:
        """
        Generate captions for a list of images.

        Images are processed individually (the LLaVA architecture makes
        true batching non-trivial with variable-length vision tokens).
        A future optimisation may pad and batch when all images share the
        same resolution.

        Parameters
        ----------
        images:
            List of image sources accepted by :func:`load_image`.
        options:
            Shared generation options for all images.

        Returns
        -------
        list[CaptionResult]
            Results in the same order as *images*.
        """
        if options is None:
            options = CaptionOptions()

        results: list[CaptionResult] = []
        for img_source in images:
            try:
                result = self.caption(img_source, options)
            except Exception as exc:  # noqa: BLE001
                logger.error("Error captioning image: %s", exc, exc_info=True)
                result = CaptionResult(
                    caption="",
                    confidence=0.0,
                    flagged=True,
                    latency_ms=0.0,
                )
            results.append(result)
        return results

    def warmup(self) -> None:
        """
        Run one dummy forward pass to initialise CUDA kernels.

        Call this once after loading the model so that the first real
        request does not incur JIT-compilation overhead.
        """
        logger.info("Running model warmup …")
        dummy = Image.fromarray(
            __import__("numpy").zeros((TARGET_SIZE, TARGET_SIZE, 3), dtype="uint8")
        )
        _ = self.caption(dummy, CaptionOptions(max_new_tokens=10))
        logger.info("Warmup complete.")
