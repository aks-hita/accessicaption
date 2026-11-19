"""
api/content_filter.py
=====================
Image and caption content-safety filtering for the AccessiCaption API.

Image screening uses the ``Falconsai/nsfw_image_detection`` HuggingFace
pipeline (lazy-loaded).  Caption toxicity checking uses the ``detoxify``
library (also lazy-loaded).

Both dependencies are optional at import time; if unavailable, the filter
logs a warning and defaults to allowing all content through.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import PIL.Image

from api.models import ContentFilterLevel, ContentFilterResult

if TYPE_CHECKING:
    pass  # kept for forward-reference hygiene

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NSFW label sets per filter level
# ---------------------------------------------------------------------------

# Labels emitted by Falconsai/nsfw_image_detection
# (adjust if the model's label schema changes)
_NSFW_LABEL = "nsfw"
_SUGGESTIVE_LABELS: frozenset[str] = frozenset({"suggestive", "sexy", "revealing"})


class ContentFilter:
    """
    Two-stage content safety filter.

    Stage 1 — **Image**: uses the ``Falconsai/nsfw_image_detection`` image
    classification pipeline to detect inappropriate visual content.

    Stage 2 — **Caption**: uses ``detoxify`` to detect toxic language in the
    generated alt-text string.

    Both models are loaded lazily on the first call so that the API can start
    even when the optional dependencies are not installed.
    """

    def __init__(self) -> None:
        self._image_pipeline = None  # lazily loaded
        self._detoxify_model = None  # lazily loaded
        self._image_pipeline_available: bool | None = None
        self._detoxify_available: bool | None = None

    # ------------------------------------------------------------------
    # Lazy loaders
    # ------------------------------------------------------------------

    def _get_image_pipeline(self):
        """Return the NSFW image detection pipeline, loading it on first call."""
        if self._image_pipeline_available is False:
            return None

        if self._image_pipeline is None:
            try:
                from transformers import pipeline  # type: ignore

                logger.info(
                    "Loading Falconsai/nsfw_image_detection content filter …"
                )
                self._image_pipeline = pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                )
                self._image_pipeline_available = True
                logger.info("NSFW image filter loaded.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "NSFW image detection model unavailable (%s). "
                    "Content filter will default to safe for all images.",
                    exc,
                )
                self._image_pipeline_available = False

        return self._image_pipeline

    def _get_detoxify(self):
        """Return the detoxify model, loading it on first call."""
        if self._detoxify_available is False:
            return None

        if self._detoxify_model is None:
            try:
                from detoxify import Detoxify  # type: ignore

                logger.info("Loading detoxify caption toxicity model …")
                self._detoxify_model = Detoxify("original")
                self._detoxify_available = True
                logger.info("Detoxify model loaded.")
            except Exception as exc:  # noqa: BLE001
                logger.warning(
                    "detoxify not available (%s). Caption toxicity checking disabled.",
                    exc,
                )
                self._detoxify_available = False

        return self._detoxify_model

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def check_image(
        self,
        image: PIL.Image.Image,
        level: ContentFilterLevel,
    ) -> ContentFilterResult:
        """
        Run the NSFW image classifier and return a filter result.

        Parameters
        ----------
        image:
            PIL image to screen.
        level:
            Filter sensitivity:

            - ``strict``   — block NSFW **and** suggestive content.
            - ``moderate`` — block only explicit NSFW content.
            - ``off``      — always allow; return ``triggered=False``.

        Returns
        -------
        ContentFilterResult
        """
        if level == ContentFilterLevel.off:
            return ContentFilterResult(triggered=False, category=None)

        pipeline_fn = self._get_image_pipeline()
        if pipeline_fn is None:
            # Model not available — default to safe
            logger.debug("Image pipeline unavailable; defaulting to safe.")
            return ContentFilterResult(triggered=False, category=None)

        try:
            results: list[dict] = pipeline_fn(image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running NSFW pipeline: %s", exc, exc_info=True)
            return ContentFilterResult(triggered=False, category=None)

        # Sort by score descending; take the top label
        results = sorted(results, key=lambda r: r.get("score", 0.0), reverse=True)
        if not results:
            return ContentFilterResult(triggered=False, category=None)

        top_label: str = results[0].get("label", "").lower()

        if _NSFW_LABEL in top_label:
            return ContentFilterResult(triggered=True, category="nsfw")

        if level == ContentFilterLevel.strict:
            for label_info in results:
                lbl = label_info.get("label", "").lower()
                if any(s in lbl for s in _SUGGESTIVE_LABELS):
                    return ContentFilterResult(triggered=True, category="suggestive")

        return ContentFilterResult(triggered=False, category=None)

    def check_caption(self, caption: str) -> bool:
        """
        Check a generated caption for toxic language.

        Uses ``detoxify`` to score the caption across several toxicity
        dimensions (toxicity, severe_toxicity, obscene, threat, insult,
        identity_attack).  Returns ``False`` (unsafe) if any score exceeds
        a 0.5 threshold.

        Parameters
        ----------
        caption:
            The alt-text string produced by the captioning model.

        Returns
        -------
        bool
            ``True`` if the caption is safe; ``False`` if toxic content was
            detected.
        """
        detoxify = self._get_detoxify()
        if detoxify is None:
            # Library not available — default to safe
            logger.debug("detoxify unavailable; assuming caption is safe.")
            return True

        if not caption.strip():
            return True  # empty captions are not toxic

        try:
            scores: dict[str, float] = detoxify.predict(caption)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running detoxify: %s", exc, exc_info=True)
            return True  # fail open

        _TOXICITY_THRESHOLD = 0.5
        for dimension, score in scores.items():
            if score > _TOXICITY_THRESHOLD:
                logger.warning(
                    "Caption flagged by detoxify — dimension=%s score=%.3f: %r",
                    dimension, score, caption,
                )
                return False

        return True
