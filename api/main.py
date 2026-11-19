"""
api/main.py
===========
FastAPI application for the AccessiCaption REST API.

Endpoints
---------
POST /v1/caption          — single-image captioning (JSON or multipart)
POST /v1/caption/batch    — batch captioning (JSON, up to 50 images)
GET  /v1/health           — model status, version, uptime

Middleware / cross-cutting concerns
------------------------------------
- Bearer-token API-key authentication
- Per-key rate limiting via slowapi (60 req / min)
- UUID4 request-ID injection
- Structured request logging
- 20 MB max image size enforcement
- Allowed MIME type enforcement
"""

from __future__ import annotations

import base64
import hashlib
import io
import logging
import os
import sys
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated

import PIL.Image
from fastapi import (
    Depends,
    FastAPI,
    Header,
    HTTPException,
    Request,
    Response,
    UploadFile,
    File,
    Form,
    status,
)
from fastapi.exception_handlers import http_exception_handler
from fastapi.responses import JSONResponse

# Slowapi rate-limiting
try:
    from slowapi import Limiter, _rate_limit_exceeded_handler  # type: ignore
    from slowapi.util import get_remote_address  # type: ignore
    from slowapi.errors import RateLimitExceeded  # type: ignore
    _SLOWAPI_AVAILABLE = True
except ImportError:
    _SLOWAPI_AVAILABLE = False

from api.models import (
    BatchCaptionRequest,
    BatchCaptionResponse,
    CaptionRequest,
    CaptionRequestOptions,
    CaptionResponse,
    ContentFilterLevel,
    ContentFilterResult,
    ErrorResponse,
)
from api.content_filter import ContentFilter
from inference.captioner import Captioner, CaptionOptions, load_image

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    stream=sys.stdout,
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
)
logger = logging.getLogger("accessicaption.api")

# ---------------------------------------------------------------------------
# Configuration from environment
# ---------------------------------------------------------------------------

MODEL_PATH: str = os.getenv(
    "MODEL_PATH", "./outputs/accessicaption-v1/merged"
)
_raw_api_keys: str = os.getenv("ACCESSICAPTION_API_KEYS", "")
VALID_API_KEYS: frozenset[str] = frozenset(
    k.strip() for k in _raw_api_keys.split(",") if k.strip()
)

MAX_IMAGE_BYTES: int = 20 * 1024 * 1024  # 20 MB
ALLOWED_MIME_TYPES: frozenset[str] = frozenset(
    {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp", "image/tiff"}
)
RATE_LIMIT: str = "60/minute"

# ---------------------------------------------------------------------------
# Application singletons (populated at startup)
# ---------------------------------------------------------------------------

_captioner: Captioner | None = None
_content_filter: ContentFilter | None = None
_startup_time: float = time.time()

# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


def _get_api_key_identity(request: Request) -> str:
    """Use a hash of the API key (not the raw key) as the rate-limit identity."""
    auth = request.headers.get("Authorization", "")
    key = auth.removeprefix("Bearer ").strip()
    return hashlib.sha256(key.encode()).hexdigest()[:16]


if _SLOWAPI_AVAILABLE:
    limiter = Limiter(key_func=_get_api_key_identity)
else:
    limiter = None  # type: ignore


# ---------------------------------------------------------------------------
# Lifespan (startup / shutdown)
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model singletons on startup; release on shutdown."""
    global _captioner, _content_filter

    logger.info("Starting AccessiCaption API …")
    logger.info("Model path: %s", MODEL_PATH)

    _content_filter = ContentFilter()

    try:
        _captioner = Captioner(model_path=MODEL_PATH)
        _captioner.warmup()
        logger.info("Captioner ready.")
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to load Captioner: %s", exc, exc_info=True)
        # Allow the API to start in degraded mode (health check will report down)

    yield

    logger.info("Shutting down AccessiCaption API.")


# ---------------------------------------------------------------------------
# FastAPI app
# ---------------------------------------------------------------------------

app = FastAPI(
    title="AccessiCaption API",
    description=(
        "WCAG-compliant image alt-text generation for accessibility.\n\n"
        "All endpoints require a valid API key passed as "
        "`Authorization: Bearer <key>`."
    ),
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/v1/docs",
    redoc_url="/v1/redoc",
)

if _SLOWAPI_AVAILABLE and limiter is not None:
    app.state.limiter = limiter
    app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ---------------------------------------------------------------------------
# Middleware helpers
# ---------------------------------------------------------------------------


def _inject_request_id(request: Request) -> str:
    """Generate and attach a UUID4 request ID to the request state."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    return request_id


@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    """Inject X-Request-ID into every request and response."""
    request_id = str(uuid.uuid4())
    request.state.request_id = request_id
    response: Response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ---------------------------------------------------------------------------
# Auth dependency
# ---------------------------------------------------------------------------


async def verify_api_key(
    authorization: Annotated[str | None, Header()] = None,
    request: Request = None,
) -> str:
    """
    Validate the ``Authorization: Bearer <key>`` header.

    Returns the raw key on success; raises HTTP 401 on failure.

    If ``ACCESSICAPTION_API_KEYS`` is empty the check is skipped (dev mode).
    """
    request_id = getattr(request.state, "request_id", "unknown") if request else "unknown"

    if not VALID_API_KEYS:
        logger.warning("No API keys configured — running in unauthenticated dev mode.")
        return "dev"

    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error="Missing Authorization header.",
                code="unauthorized",
                request_id=request_id,
            ).model_dump(),
        )

    scheme, _, key = authorization.partition(" ")
    if scheme.lower() != "bearer" or not key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error="Authorization header must use Bearer scheme.",
                code="unauthorized",
                request_id=request_id,
            ).model_dump(),
        )

    if key not in VALID_API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=ErrorResponse(
                error="Invalid API key.",
                code="unauthorized",
                request_id=request_id,
            ).model_dump(),
        )

    return key


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _require_captioner(request: Request) -> Captioner:
    """Return the global Captioner or raise 503 if not loaded."""
    if _captioner is None:
        request_id = getattr(request.state, "request_id", "unknown")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=ErrorResponse(
                error="Model is not loaded. The service is starting up or encountered an error.",
                code="model_unavailable",
                request_id=request_id,
            ).model_dump(),
        )
    return _captioner


def _decode_image_source(
    source: str, request_id: str
) -> PIL.Image.Image:
    """
    Decode *source* into a PIL image.

    *source* may be:
    - A data URI (``data:image/...;base64,...``).
    - A raw base64-encoded string.
    - A publicly accessible HTTP/HTTPS URL.
    """
    source = source.strip()

    if source.startswith("data:"):
        # data URI
        try:
            header, encoded = source.split(",", 1)
            img_bytes = base64.b64decode(encoded)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="Invalid data URI encoding.",
                    code="bad_image",
                    request_id=request_id,
                ).model_dump(),
            )
    elif source.startswith("http://") or source.startswith("https://"):
        # Remote URL — load_image handles this
        try:
            return load_image(source)
        except Exception as exc:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error=f"Could not fetch image from URL: {exc}",
                    code="bad_image",
                    request_id=request_id,
                ).model_dump(),
            )
    else:
        # Raw base64
        try:
            img_bytes = base64.b64decode(source)
        except Exception:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=ErrorResponse(
                    error="Image field must be a base64-encoded string or a URL.",
                    code="bad_image",
                    request_id=request_id,
                ).model_dump(),
            )

    _validate_image_bytes(img_bytes, request_id)
    return load_image(img_bytes)


def _validate_image_bytes(data: bytes, request_id: str) -> None:
    """Raise HTTPException for oversized or invalid image bytes."""
    if len(data) > MAX_IMAGE_BYTES:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=ErrorResponse(
                error=f"Image exceeds maximum allowed size of {MAX_IMAGE_BYTES // (1024 * 1024)} MB.",
                code="too_large",
                request_id=request_id,
            ).model_dump(),
        )

    # Probe MIME type via Pillow
    try:
        probe = PIL.Image.open(io.BytesIO(data))
        fmt = (probe.format or "").lower()
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error="Cannot parse image data.",
                code="bad_image",
                request_id=request_id,
            ).model_dump(),
        )

    mime = f"image/{fmt}"
    if mime not in ALLOWED_MIME_TYPES:
        raise HTTPException(
            status_code=status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
            detail=ErrorResponse(
                error=f"Unsupported image format '{fmt}'. Allowed: jpeg, png, webp, gif, bmp, tiff.",
                code="unsupported_format",
                request_id=request_id,
            ).model_dump(),
        )


def _build_caption_response(
    pil_image: PIL.Image.Image,
    options: CaptionRequestOptions,
    request_id: str,
    captioner: Captioner,
) -> CaptionResponse:
    """
    Run the content filter, then caption, then log and return the response.
    """
    cf_result: ContentFilterResult = _content_filter.check_image(
        pil_image, options.content_filter_level
    )

    if cf_result.triggered:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=ErrorResponse(
                error=f"Image blocked by content filter (category: {cf_result.category}).",
                code="content_filtered",
                request_id=request_id,
            ).model_dump(),
        )

    cap_opts = CaptionOptions(
        confidence_threshold=options.confidence_threshold,
        max_new_tokens=options.max_new_tokens,
        temperature=options.temperature,
    )

    try:
        result = captioner.caption(pil_image, cap_opts)
    except Exception as exc:
        logger.error("Model inference error: %s", exc, exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=ErrorResponse(
                error="An error occurred during model inference.",
                code="model_error",
                request_id=request_id,
            ).model_dump(),
        )

    # Caption toxicity check
    caption_safe = _content_filter.check_caption(result.caption)
    if not caption_safe:
        result = result.__class__(
            caption="[Caption withheld by content filter]",
            confidence=result.confidence,
            flagged=True,
            latency_ms=result.latency_ms,
        )

    logger.info(
        "request_id=%s latency_ms=%.1f confidence=%.3f flagged=%s cf_triggered=%s",
        request_id,
        result.latency_ms,
        result.confidence,
        result.flagged,
        cf_result.triggered,
    )

    return CaptionResponse(
        caption=result.caption,
        confidence=result.confidence,
        flagged=result.flagged,
        content_filter=cf_result,
        latency_ms=result.latency_ms,
        request_id=request_id,
        model_version=Captioner.MODEL_VERSION,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.post(
    "/v1/caption",
    response_model=CaptionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Caption a single image",
    tags=["Captioning"],
)
async def caption_image_json(
    body: CaptionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Generate a WCAG-compliant alt-text caption for a single image.

    The image may be supplied as a **base64-encoded string** (raw or data-URI)
    or a **publicly accessible URL**.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    captioner = _require_captioner(request)

    pil_image = _decode_image_source(body.image, request_id)
    return _build_caption_response(pil_image, body.options, request_id, captioner)


@app.post(
    "/v1/caption/upload",
    response_model=CaptionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Caption a single image (file upload)",
    tags=["Captioning"],
)
async def caption_image_upload(
    request: Request,
    file: UploadFile = File(..., description="Image file to caption."),
    confidence_threshold: float = Form(0.75),
    content_filter_level: ContentFilterLevel = Form(ContentFilterLevel.moderate),
    max_new_tokens: int = Form(80),
    temperature: float = Form(0.2),
    api_key: str = Depends(verify_api_key),
):
    """
    Caption a single image uploaded as ``multipart/form-data``.

    Use this endpoint when you have the image file on disk and prefer not to
    base64-encode it.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    captioner = _require_captioner(request)

    img_bytes = await file.read()
    _validate_image_bytes(img_bytes, request_id)
    pil_image = load_image(img_bytes)

    options = CaptionRequestOptions(
        confidence_threshold=confidence_threshold,
        content_filter_level=content_filter_level,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
    )
    return _build_caption_response(pil_image, options, request_id, captioner)


@app.post(
    "/v1/caption/batch",
    response_model=BatchCaptionResponse,
    responses={
        400: {"model": ErrorResponse},
        401: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        415: {"model": ErrorResponse},
        429: {"description": "Rate limit exceeded"},
        500: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Caption multiple images in a single request",
    tags=["Captioning"],
)
async def caption_batch(
    body: BatchCaptionRequest,
    request: Request,
    api_key: str = Depends(verify_api_key),
):
    """
    Caption up to **50 images** in a single API call.

    Each image may be a base64-encoded string or a URL.  Results are returned
    in the same order as the input list.
    """
    request_id: str = getattr(request.state, "request_id", str(uuid.uuid4()))
    captioner = _require_captioner(request)

    t_batch_start = time.perf_counter()
    results: list[CaptionResponse] = []

    for idx, img_source in enumerate(body.images):
        item_request_id = f"{request_id}#{idx}"
        try:
            pil_image = _decode_image_source(img_source, item_request_id)
            resp = _build_caption_response(
                pil_image, body.options, item_request_id, captioner
            )
        except HTTPException as exc:
            # Return a stub result instead of aborting the whole batch
            detail = exc.detail if isinstance(exc.detail, dict) else {}
            resp = CaptionResponse(
                caption="",
                confidence=0.0,
                flagged=True,
                content_filter=ContentFilterResult(triggered=False, category=None),
                latency_ms=0.0,
                request_id=item_request_id,
                model_version=Captioner.MODEL_VERSION,
            )
            logger.warning(
                "Batch item %d failed: %s", idx, detail.get("error", exc.detail)
            )
        results.append(resp)

    total_latency_ms = (time.perf_counter() - t_batch_start) * 1000.0

    logger.info(
        "Batch request_id=%s items=%d total_latency_ms=%.1f",
        request_id, len(results), total_latency_ms,
    )

    return BatchCaptionResponse(
        results=results,
        total_latency_ms=total_latency_ms,
        request_id=request_id,
    )


@app.get(
    "/v1/health",
    summary="Health check",
    tags=["Operations"],
)
async def health_check():
    """
    Return service health status, model readiness, and uptime.

    Does **not** require an API key so that load balancers and monitoring
    systems can poll it freely.
    """
    uptime_seconds = time.time() - _startup_time
    model_ready = _captioner is not None

    return {
        "status": "ok" if model_ready else "degraded",
        "model_ready": model_ready,
        "model_version": Captioner.MODEL_VERSION if model_ready else None,
        "model_path": MODEL_PATH,
        "uptime_seconds": round(uptime_seconds, 1),
        "api_version": app.version,
    }
