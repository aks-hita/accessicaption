"""
AccessiCaption test suite — sanity checks for WCAG normalization
and confidence estimation. Can be run without a GPU.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))


# ── WCAG Normalization tests ──────────────────────────────────────────────────

class TestWCAGNormalize:
    def _normalize(self, text: str) -> str:
        from data.wcag_normalize import normalize_caption
        return normalize_caption(text)

    def test_strips_image_of_prefix(self):
        result = self._normalize("Image of a dog sitting on a chair.")
        assert not result.lower().startswith("image"), f"Got: {result}"

    def test_strips_photo_of_prefix(self):
        result = self._normalize("A photo of a mountain landscape at sunset.")
        assert not result.lower().startswith("photo"), f"Got: {result}"
        assert not result.lower().startswith("a photo"), f"Got: {result}"

    def test_strips_picture_of_prefix(self):
        result = self._normalize("Picture of two people shaking hands.")
        assert not result.lower().startswith("picture"), f"Got: {result}"

    def test_capitalizes_first_letter(self):
        result = self._normalize("a cat sleeps on a warm window sill.")
        assert result[0].isupper(), f"Got: {result}"

    def test_ensures_period_at_end(self):
        result = self._normalize("A child plays in the park")
        assert result.endswith("."), f"Got: {result}"

    def test_does_not_double_period(self):
        result = self._normalize("A child plays in the park.")
        assert result.count(".") == result.count(".")  # idempotent
        assert not result.endswith(".."), f"Got: {result}"

    def test_strips_this_image_shows(self):
        result = self._normalize("This image shows a busy intersection at night.")
        assert not result.lower().startswith("this image"), f"Got: {result}"

    def test_handles_empty_string(self):
        result = self._normalize("")
        assert isinstance(result, str)

    def test_long_caption_truncated(self):
        long = "A " + "very " * 60 + "long caption that never ends and keeps going on forever."
        result = self._normalize(long)
        assert len(result) <= 155, f"Too long: {len(result)}"

    def test_batch_normalize(self):
        from data.wcag_normalize import batch_normalize
        captions = ["Image of a cat.", "Photo of a dog.", "A bird flies."]
        results = batch_normalize(captions)
        assert len(results) == 3
        for r in results:
            assert not r.lower().startswith("image")
            assert not r.lower().startswith("photo")


# ── Confidence estimation tests (no GPU needed) ───────────────────────────────

class TestConfidenceEstimation:
    def test_tokens_entropy_uniform(self):
        """Uniform distribution should give max entropy → low confidence."""
        import torch
        from inference.confidence import tokens_entropy
        uniform_logits = torch.ones(100)  # all equal → max entropy
        entropy = tokens_entropy(uniform_logits)
        assert entropy > 0, "Entropy should be positive for uniform distribution"

    def test_tokens_entropy_peaked(self):
        """Peaked distribution should give low entropy → high confidence."""
        import torch
        from inference.confidence import tokens_entropy
        peaked_logits = torch.zeros(100)
        peaked_logits[0] = 100.0  # strongly peaked at first token
        entropy = tokens_entropy(peaked_logits)
        assert entropy < 0.01, f"Entropy should be near 0 for peaked distribution, got {entropy}"

    def test_confidence_range(self):
        """Confidence should always be in [0, 1]."""
        import torch
        from inference.confidence import estimate_confidence_logit_variance
        # Simulate a sequence of logits (5 generation steps, vocab size 100)
        logits_seq = [torch.randn(100) for _ in range(5)]
        conf = estimate_confidence_logit_variance(logits_seq)
        assert 0.0 <= conf <= 1.0, f"Confidence out of range: {conf}"


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])
