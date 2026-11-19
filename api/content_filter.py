"""
api/content_filter.py
=====================
Image and caption content-safety filtering for AccessiCaption.

Stage 1 — Image: Optional Hugging Face pipeline (Falconsai/nsfw_image_detection).
Stage 2 — Caption: Fast stdlib pattern screening (ponytail: replaces 500MB RoBERTa model).
"""

from __future__ import annotations

import logging
import re
from typing import TYPE_CHECKING

import PIL.Image

from api.models import ContentFilterLevel, ContentFilterResult

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

_NSFW_LABEL = "nsfw"
_SUGGESTIVE_LABELS: frozenset[str] = frozenset({"suggestive", "sexy", "revealing"})

# ponytail: lightweight standard library profanity/hate regex replaces 500MB detoxify model
_TOXIC_PATTERNS = re.compile(
    r"\b(nigger|faggot|kike|chink|cunt|kill\s+yourself|rape)\b",
    re.IGNORECASE,
)


class ContentFilter:
    """Two-stage content safety filter (lazy-loaded)."""

    def __init__(self) -> None:
        self._image_pipeline = None
        self._image_pipeline_available: bool | None = None

    def _get_image_pipeline(self):
        """Return the NSFW image detection pipeline, loading it on first call."""
        if self._image_pipeline_available is False:
            return None

        if self._image_pipeline is None:
            try:
                from transformers import pipeline  # type: ignore

                logger.info("Loading Falconsai/nsfw_image_detection content filter …")
                self._image_pipeline = pipeline(
                    "image-classification",
                    model="Falconsai/nsfw_image_detection",
                )
                self._image_pipeline_available = True
            except Exception as exc:  # noqa: BLE001
                logger.debug("NSFW image detection pipeline not available (%s); defaulting to safe.", exc)
                self._image_pipeline_available = False

        return self._image_pipeline

    def check_image(
        self,
        image: PIL.Image.Image,
        level: ContentFilterLevel,
    ) -> ContentFilterResult:
        """Screen image for NSFW content."""
        if level == ContentFilterLevel.off:
            return ContentFilterResult(triggered=False, category=None)

        pipeline_fn = self._get_image_pipeline()
        if pipeline_fn is None:
            return ContentFilterResult(triggered=False, category=None)

        try:
            results: list[dict] = pipeline_fn(image)
        except Exception as exc:  # noqa: BLE001
            logger.error("Error running NSFW pipeline: %s", exc)
            return ContentFilterResult(triggered=False, category=None)

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
        """Screen generated caption text. Returns True if safe, False if toxic."""
        if not caption.strip():
            return True

        if _TOXIC_PATTERNS.search(caption):
            logger.warning("Caption blocked by safety pattern filter: %r", caption)
            return False

        return True
