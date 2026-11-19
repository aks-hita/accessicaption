"""
inference/confidence.py
=======================
Standalone confidence-estimation utilities for LLaVA generation outputs.

Two estimation strategies are provided:

1. **Logit-variance** (fast, no extra forward passes):
   Uses the per-step generation logits to compute token-level Shannon
   entropy, then converts the mean entropy to a [0, 1] confidence score.

2. **MC-Dropout** (slower, more accurate):
   Runs *n* stochastic forward passes with dropout enabled and measures
   the consistency of the generated captions via sentence-embedding
   cosine similarity.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from transformers import (  # type: ignore
        LlavaForConditionalGeneration,
        AutoProcessor,
    )
    import PIL.Image

logger = logging.getLogger(__name__)

# Top-k logits to retain when computing entropy (reduces noise from long tail)
TOP_K = 50


# ---------------------------------------------------------------------------
# Low-level helper
# ---------------------------------------------------------------------------


def tokens_entropy(logits: torch.Tensor, top_k: int = TOP_K) -> float:
    """
    Compute Shannon entropy of the top-k probability mass for a single step.

    Parameters
    ----------
    logits:
        1-D tensor of raw logits for one generation step (shape: ``[vocab]``).
    top_k:
        Number of highest-scoring tokens to retain before softmax.  Focusing
        on the top-k prevents the long tail of near-zero logits from
        dominating the entropy calculation.

    Returns
    -------
    float
        Shannon entropy in nats (not normalised to [0, 1]).
    """
    if logits.dim() != 1:
        raise ValueError(f"Expected 1-D logits tensor, got shape {logits.shape}")

    # Retain only top-k entries
    k = min(top_k, logits.size(-1))
    top_logits, _ = torch.topk(logits, k)

    probs = F.softmax(top_logits.float(), dim=-1)
    # Numerical stability: clamp to avoid log(0)
    probs = probs.clamp(min=1e-9)
    entropy = -(probs * probs.log()).sum().item()
    return entropy


# ---------------------------------------------------------------------------
# Strategy 1 — logit variance / entropy
# ---------------------------------------------------------------------------


def estimate_confidence_logit_variance(
    logits_sequence: list[torch.Tensor],
    top_k: int = TOP_K,
) -> float:
    """
    Estimate generation confidence from per-step logit entropy.

    For each generated token the entropy of the top-k softmax distribution
    is computed.  The mean entropy across the sequence is then normalised by
    ``log(top_k)`` (maximum possible entropy under a uniform distribution
    over *top_k* outcomes) to obtain a value in ``[0, 1]``, which is finally
    inverted so that *high confidence = low entropy*.

    Parameters
    ----------
    logits_sequence:
        List of 1-D (or 2-D with batch dim 1) tensors, one per generation
        step, each of shape ``[vocab_size]`` or ``[1, vocab_size]``.
    top_k:
        Number of top logits retained per step (default: 50).

    Returns
    -------
    float
        Confidence in ``[0, 1]``. Returns ``0.0`` for an empty sequence.
    """
    if not logits_sequence:
        return 0.0

    entropies: list[float] = []
    for step_logits in logits_sequence:
        # Squeeze batch dim if present
        if step_logits.dim() == 2:
            step_logits = step_logits.squeeze(0)
        entropies.append(tokens_entropy(step_logits, top_k=top_k))

    mean_entropy = sum(entropies) / len(entropies)

    # Maximum entropy under a uniform distribution over top_k outcomes
    max_entropy = math.log(min(top_k, logits_sequence[0].numel()))
    if max_entropy == 0:
        return 1.0  # single-token vocab — trivially certain

    normalised_entropy = mean_entropy / max_entropy
    # Clamp to [0, 1] in case of floating-point overshoot
    confidence = 1.0 - min(max(normalised_entropy, 0.0), 1.0)
    return confidence


# ---------------------------------------------------------------------------
# Strategy 2 — MC-Dropout
# ---------------------------------------------------------------------------


def _enable_dropout(model: "LlavaForConditionalGeneration") -> None:
    """Set all Dropout layers in *model* to training mode."""
    for module in model.modules():
        if isinstance(module, torch.nn.Dropout):
            module.train()


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Return cosine similarity between two equal-length vectors."""
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x ** 2 for x in a))
    norm_b = math.sqrt(sum(y ** 2 for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _embed_text(text: str) -> list[float]:
    """
    Produce a simple bag-of-characters embedding for similarity comparison.

    This is intentionally lightweight so that the MC-Dropout estimator has
    no extra heavy dependencies.  For production use, replace with a
    sentence-transformer embedding.
    """
    # Character frequency vector (ASCII printable: 32–126)
    vec = [0.0] * 95
    for ch in text.lower():
        idx = ord(ch) - 32
        if 0 <= idx < 95:
            vec[idx] += 1.0
    # L2-normalise
    norm = math.sqrt(sum(v ** 2 for v in vec)) or 1.0
    return [v / norm for v in vec]


def estimate_confidence_mc_dropout(
    model: "LlavaForConditionalGeneration",
    processor: "AutoProcessor",
    image: "PIL.Image.Image",
    prompt: str,
    n_passes: int = 5,
    max_new_tokens: int = 80,
    device: torch.device | str | None = None,
) -> float:
    """
    Estimate caption confidence via Monte Carlo Dropout.

    Runs *n_passes* stochastic forward passes with dropout **enabled** and
    measures how consistently the model produces the same caption.  The
    confidence is defined as:

    .. code-block:: text

        confidence = 1 - std(pairwise_cosine_similarities)

    where ``pairwise_cosine_similarities`` is the list of cosine similarities
    between every pair of generated captions (using a lightweight character
    n-gram embedding).

    Parameters
    ----------
    model:
        A LLaVA model already loaded (may be in eval mode; dropout is
        temporarily re-enabled inside this function).
    processor:
        The corresponding ``AutoProcessor``.
    image:
        Pre-processed PIL image.
    prompt:
        Full formatted prompt string (including ``<image>`` token).
    n_passes:
        Number of stochastic forward passes (default: 5).
    max_new_tokens:
        Token budget per pass.
    device:
        Target device.  If ``None``, inferred from model parameters.

    Returns
    -------
    float
        Confidence in ``[0, 1]``.  Returns ``1.0`` if only one pass is
        requested (no variance to measure).
    """
    if n_passes <= 1:
        return 1.0

    if device is None:
        device = next(model.parameters()).device

    # Temporarily enable dropout for stochastic sampling
    model.eval()
    _enable_dropout(model)

    captions: list[str] = []
    inputs = processor(text=prompt, images=image, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    input_len = inputs["input_ids"].shape[-1]

    try:
        for _ in range(n_passes):
            with torch.no_grad():
                output = model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=1.0,  # let dropout provide stochasticity
                )
            gen_ids = output[0, input_len:]
            caption = processor.decode(gen_ids, skip_special_tokens=True).strip()
            captions.append(caption)
    finally:
        # Always restore eval mode
        model.eval()

    if len(captions) < 2:
        return 1.0

    # Pairwise cosine similarities
    embeddings = [_embed_text(c) for c in captions]
    sims: list[float] = []
    for i in range(len(embeddings)):
        for j in range(i + 1, len(embeddings)):
            sims.append(_cosine_similarity(embeddings[i], embeddings[j]))

    if not sims:
        return 1.0

    mean_sim = sum(sims) / len(sims)
    var_sim = sum((s - mean_sim) ** 2 for s in sims) / len(sims)
    std_sim = math.sqrt(var_sim)

    # High std → low agreement → low confidence
    confidence = max(0.0, min(1.0, 1.0 - std_sim))
    logger.debug(
        "MC-Dropout confidence: %.3f (mean_sim=%.3f, std_sim=%.3f, n=%d)",
        confidence, mean_sim, std_sim, n_passes,
    )
    return confidence
