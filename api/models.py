"""
api/models.py
=============
Pydantic v2 request / response models for the AccessiCaption REST API.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Enumerations
# ---------------------------------------------------------------------------


class ContentFilterLevel(str, Enum):
    """Controls how aggressively the content filter rejects images."""

    strict = "strict"
    """Block explicit NSFW content **and** suggestive / borderline content."""

    moderate = "moderate"
    """Block only clearly explicit NSFW content (default)."""

    off = "off"
    """Disable the content filter entirely (use with caution)."""


# ---------------------------------------------------------------------------
# Sub-models
# ---------------------------------------------------------------------------


class CaptionRequestOptions(BaseModel):
    """Optional per-request tuning parameters."""

    confidence_threshold: float = Field(
        0.75,
        ge=0.0,
        le=1.0,
        description=(
            "Minimum confidence score in [0, 1]. Results below this value "
            "are marked as flagged."
        ),
    )
    content_filter_level: ContentFilterLevel = Field(
        ContentFilterLevel.moderate,
        description="Sensitivity level for the content safety filter.",
    )
    max_new_tokens: int = Field(
        80,
        ge=10,
        le=256,
        description="Maximum number of tokens to generate.",
    )
    temperature: float = Field(
        0.2,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. 0 = greedy decoding.",
    )
    detail_level: str = Field(
        "baseline",
        description="Detail tier: 'baseline' (1-sentence alt-text), 'detailed' (2-3 sentences), or 'layered'.",
    )


class ContentFilterResult(BaseModel):
    """Result of the image (and caption) content safety check."""

    triggered: bool = Field(
        ...,
        description="True if the content filter blocked this request.",
    )
    category: str | None = Field(
        None,
        description="Category label for the detected content (e.g. 'nsfw', 'suggestive').",
    )


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------


class CaptionRequest(BaseModel):
    """
    Single-image captioning request.

    The image may be supplied as:

    - A **base64-encoded string** (data-URI or raw base64).
    - A **publicly accessible URL** (``http://`` or ``https://``).
    """

    image: str = Field(
        ...,
        description="Base64-encoded image data OR a publicly accessible image URL.",
    )
    options: CaptionRequestOptions = Field(
        default_factory=CaptionRequestOptions,
        description="Optional generation and filter settings.",
    )

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "image": "https://example.com/photo.jpg",
                "options": {"confidence_threshold": 0.8},
            }
        ]
    }}


class BatchCaptionRequest(BaseModel):
    """Batch captioning request (up to 50 images per call)."""

    images: list[str] = Field(
        ...,
        min_length=1,
        max_length=50,
        description=(
            "List of base64-encoded images or public URLs. "
            "Maximum 50 images per request."
        ),
    )
    options: CaptionRequestOptions = Field(
        default_factory=CaptionRequestOptions,
        description="Shared generation and filter settings applied to all images.",
    )


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------


class CaptionResponse(BaseModel):
    """Response for a single captioned image."""

    caption: str = Field(..., description="WCAG-compliant alt-text caption.")
    confidence: float = Field(..., description="Model confidence in [0, 1].")
    flagged: bool = Field(
        ...,
        description="True when confidence < options.confidence_threshold.",
    )
    content_filter: ContentFilterResult = Field(
        ..., description="Content safety filter outcome."
    )
    latency_ms: float = Field(
        ..., description="End-to-end inference latency in milliseconds."
    )
    request_id: str = Field(..., description="UUID4 identifier for this request.")
    model_version: str = Field(..., description="Model version string.")
    detail_level: str = Field("baseline", description="Detail tier used: 'baseline', 'detailed', or 'layered'.")
    layers: dict[str, str] = Field(default_factory=dict, description="Structured layers if detail_level='layered'.")


class BatchCaptionResponse(BaseModel):
    """Response for a batch captioning request."""

    results: list[CaptionResponse] = Field(
        ..., description="Ordered list of results, one per input image."
    )
    total_latency_ms: float = Field(
        ..., description="Total wall-clock time for the entire batch in milliseconds."
    )
    request_id: str = Field(
        ..., description="UUID4 identifier shared across the batch."
    )


class ErrorResponse(BaseModel):
    """Standard error envelope returned on 4xx / 5xx responses."""

    error: str = Field(..., description="Human-readable error message.")
    code: str = Field(
        ...,
        description=(
            "Machine-readable error code. "
            "Examples: 'bad_image', 'too_large', 'unsupported_format', "
            "'rate_limited', 'model_error', 'content_filtered'."
        ),
    )
    request_id: str = Field(..., description="UUID4 identifier for the failed request.")

    model_config = {"json_schema_extra": {
        "examples": [
            {
                "error": "Image exceeds maximum allowed size of 20 MB.",
                "code": "too_large",
                "request_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
            }
        ]
    }}
