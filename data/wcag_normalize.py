"""
wcag_normalize.py
-----------------
WCAG 2.1/2.2 compliant alt-text normalization for AccessiCaption.

Provides:
  - normalize_caption(text: str) -> str
  - batch_normalize(captions: list[str]) -> list[str]
"""

from __future__ import annotations

import re
import textwrap

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MAX_CAPTION_CHARS: int = 150

# Prefixes to strip (case-insensitive).  Ordered longest-first so the most
# specific pattern wins when prefixes share a common start.
_STRIP_PREFIXES: list[str] = [
    r"this image shows",
    r"this is an image of",
    r"this is a picture of",
    r"this is a photo of",
    r"this is a",
    r"an image of",
    r"a picture of",
    r"a photo of",
    r"image of",
    r"picture of",
    r"photo of",
]

# Compiled single regex that matches any of the above at the very start of the
# (stripped) string, optionally followed by a space or comma+space.
_PREFIX_RE: re.Pattern[str] = re.compile(
    r"^(?:" + "|".join(_STRIP_PREFIXES) + r")[,\s]*",
    re.IGNORECASE,
)

# Passive constructions to replace with active equivalents.
# Each entry is (pattern, replacement); replacements may use back-references.
_PASSIVE_REPLACEMENTS: list[tuple[re.Pattern[str], str]] = [
    # "X is shown" → "X appears" / "shows X"
    (re.compile(r"\bis shown\b", re.IGNORECASE), "appears"),
    # "X is seen" → "X is visible"
    (re.compile(r"\bis seen\b", re.IGNORECASE), "is visible"),
    # "can be seen" → "is visible"
    (re.compile(r"\bcan be seen\b", re.IGNORECASE), "is visible"),
    # "are shown" → "appear"
    (re.compile(r"\bare shown\b", re.IGNORECASE), "appear"),
    # "are seen" → "are visible"
    (re.compile(r"\bare seen\b", re.IGNORECASE), "are visible"),
]

# Words that must NOT appear as the first word (case-insensitive).
_FORBIDDEN_FIRST_WORDS: frozenset[str] = frozenset({"image", "picture", "photo"})

# Sentence-boundary pattern used for safe truncation.
_SENTENCE_END_RE: re.Pattern[str] = re.compile(r"(?<=[.!?])\s+")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def normalize_caption(text: str, max_chars: int | None = MAX_CAPTION_CHARS) -> str:
    """Apply WCAG 2.1/2.2 alt-text normalization rules to a single caption.

    Rules applied in order:
      1. Strip excessive whitespace.
      2. Remove common leading boilerplate prefixes.
      3. Replace passive "is shown" / "is seen" constructions.
      4. Capitalize the first letter.
      5. Ensure the caption ends with a period.
      6. Truncate at the last sentence boundary before ``MAX_CAPTION_CHARS``
         characters if necessary.
      7. Refuse to start with "image", "picture", or "photo".

    Parameters
    ----------
    text:
        Raw caption string.

    Returns
    -------
    str
        Normalized caption.
    """
    if not text or not text.strip():
        return ""

    # 1. Collapse whitespace
    result: str = " ".join(text.split())

    # 2. Strip boilerplate prefixes (may need multiple passes for stacked ones,
    #    e.g. "This is a photo of an image of …")
    for _ in range(5):
        stripped = _PREFIX_RE.sub("", result).strip()
        if stripped == result:
            break
        result = stripped

    # 3. Replace passive constructions
    for pattern, replacement in _PASSIVE_REPLACEMENTS:
        result = pattern.sub(replacement, result)

    # Collapse whitespace again after substitutions
    result = " ".join(result.split()).strip()

    if not result:
        return ""

    # 4. Capitalize first letter (preserve the rest)
    result = result[0].upper() + result[1:]

    # 5. Ensure trailing period (not if ends with !, ?)
    if result and result[-1] not in {".", "!", "?"}:
        result += "."

    # 6. Truncate at sentence boundary if too long and max_chars is specified
    if max_chars is not None and len(result) > max_chars:
        result = _truncate_at_sentence(result, max_chars)

    # 7. Reject forbidden first words — prefix with a neutral placeholder
    first_word = result.split()[0].rstrip(".,!?").lower() if result else ""
    if first_word in _FORBIDDEN_FIRST_WORDS:
        result = "Alt-text: " + result

    return result


def batch_normalize(captions: list[str]) -> list[str]:
    """Apply :func:`normalize_caption` to every caption in the list.

    Parameters
    ----------
    captions:
        List of raw caption strings.

    Returns
    -------
    list[str]
        List of normalized captions in the same order.
    """
    return [normalize_caption(c) for c in captions]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _truncate_at_sentence(text: str, max_chars: int) -> str:
    """Truncate *text* to at most *max_chars* characters at a sentence boundary.

    If no sentence boundary exists before *max_chars*, falls back to a hard
    truncation at the last space to avoid cutting mid-word, then appends "…".

    Parameters
    ----------
    text:
        Source text (already ends with punctuation).
    max_chars:
        Hard character limit.

    Returns
    -------
    str
        Truncated string that ends with punctuation.
    """
    candidate = text[:max_chars]

    # Try to find a sentence boundary inside the candidate
    sentences = _SENTENCE_END_RE.split(candidate)
    if len(sentences) > 1:
        # Drop the last (potentially incomplete) sentence and rejoin
        result = " ".join(sentences[:-1]).rstrip()
        if result and result[-1] not in {".", "!", "?"}:
            result += "."
        return result

    # No sentence boundary — truncate at last space
    last_space = candidate.rfind(" ")
    if last_space > 0:
        truncated = candidate[:last_space].rstrip(".,!?…") + "…"
        return truncated

    # Absolute fallback: hard cut
    return candidate.rstrip(".,!?…") + "…"


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _TEST_CASES: list[tuple[str, str]] = [
        # (input, description)
        (
            "image of a dog sitting on a sofa",
            "Strips 'image of' prefix",
        ),
        (
            "Photo of two people shaking hands in an office",
            "Strips 'Photo of' prefix and capitalizes",
        ),
        (
            "This is a picture of a sunset over the mountains",
            "Strips 'This is a picture of'",
        ),
        (
            "This image shows a crowded marketplace with colorful stalls.",
            "Strips 'This image shows'",
        ),
        (
            "a cat is shown sleeping on a window sill",
            "Replaces 'is shown' with 'appears'",
        ),
        (
            "three birds can be seen perched on a wire",
            "Replaces 'can be seen' with 'is visible'",
        ),
        (
            "   excessive   whitespace    everywhere   ",
            "Collapses whitespace",
        ),
        (
            "a very long caption " + "word " * 40 + "ending here",
            "Truncates long caption at sentence boundary",
        ),
        (
            "short caption without period",
            "Adds trailing period",
        ),
        (
            "An image of a photo of a picture of a cat",
            "Handles stacked prefixes",
        ),
    ]

    print(f"{'─' * 72}")
    print(f"{'WCAG Normalize — test suite':^72}")
    print(f"{'─' * 72}")

    all_passed = True
    for idx, (raw, description) in enumerate(_TEST_CASES, start=1):
        normalized = normalize_caption(raw)
        first_word = normalized.split()[0].rstrip(".,!?").lower() if normalized else ""
        ends_with_punct = normalized[-1] in {".", "!", "?", "…"} if normalized else False
        too_long = len(normalized) > MAX_CAPTION_CHARS
        starts_forbidden = first_word in _FORBIDDEN_FIRST_WORDS

        issues: list[str] = []
        if not ends_with_punct:
            issues.append("missing trailing punctuation")
        if too_long:
            issues.append(f"exceeds {MAX_CAPTION_CHARS} chars ({len(normalized)})")
        if starts_forbidden:
            issues.append(f"starts with forbidden word '{first_word}'")

        status = "✗ FAIL" if issues else "✓ PASS"
        if issues:
            all_passed = False

        print(f"\n[{idx:02d}] {status} — {description}")
        print(f"     IN : {raw[:80]!r}{'…' if len(raw) > 80 else ''}")
        print(f"     OUT: {normalized[:100]!r}{'…' if len(normalized) > 100 else ''}")
        if issues:
            print(f"     !! : {', '.join(issues)}")

    print(f"\n{'─' * 72}")
    print(f"Result: {'ALL PASSED ✓' if all_passed else 'SOME TESTS FAILED ✗'}")
    print(f"{'─' * 72}\n")
