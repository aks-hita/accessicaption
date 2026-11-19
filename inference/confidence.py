"""
inference/confidence.py
=======================
Lightweight confidence estimation derived directly from generation logit entropy.
No extra forward passes or heavy external models required.
"""

from __future__ import annotations

import math
import torch
import torch.nn.functional as F

TOP_K = 50


def tokens_entropy(logits: torch.Tensor, top_k: int = TOP_K) -> float:
    """Compute Shannon entropy of the top-k probability mass for a generation step."""
    if logits.dim() != 1:
        raise ValueError(f"Expected 1-D logits tensor, got shape {logits.shape}")

    k = min(top_k, logits.size(-1))
    top_logits, _ = torch.topk(logits, k)
    probs = F.softmax(top_logits.float(), dim=-1).clamp(min=1e-9)
    return -(probs * probs.log()).sum().item()


def estimate_confidence_logit_variance(
    logits_sequence: list[torch.Tensor],
    top_k: int = TOP_K,
) -> float:
    """
    Estimate confidence in [0, 1] from per-step logit entropy.
    Low entropy across generated tokens indicates high model confidence.
    """
    if not logits_sequence:
        return 0.0

    entropies = [
        tokens_entropy(step.squeeze(0) if step.dim() == 2 else step, top_k=top_k)
        for step in logits_sequence
    ]
    mean_entropy = sum(entropies) / len(entropies)
    max_entropy = math.log(min(top_k, logits_sequence[0].numel()))

    if max_entropy == 0:
        return 1.0

    normalised = mean_entropy / max_entropy
    return 1.0 - min(max(normalised, 0.0), 1.0)
