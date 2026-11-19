"""
dataset_stats.py
----------------
Reads processed train.json / eval.json and prints detailed statistics about
the AccessiCaption dataset.

Usage
-----
    python dataset_stats.py --data_dir ./data/processed
"""

from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Keywords whose presence flags a caption as text/OCR-related
_OCR_KEYWORDS: frozenset[str] = frozenset(
    {
        "text",
        "sign",
        "word",
        "letter",
        "number",
        "written",
        "label",
        "caption",
        "read",
        "ocr",
        "font",
        "print",
        "logo",
        "brand",
        "menu",
        "street",
        "banner",
    }
)

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_split(path: Path) -> list[Record]:
    """Load a JSON records file.

    Parameters
    ----------
    path:
        Path to a JSON file containing a list of record dicts.

    Returns
    -------
    list[Record]
        Parsed list of record dicts.

    Raises
    ------
    SystemExit
        If the file does not exist or cannot be parsed.
    """
    if not path.exists():
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        sys.exit(1)
    with path.open("r", encoding="utf-8") as fh:
        try:
            return json.load(fh)
        except json.JSONDecodeError as exc:
            print(f"[ERROR] Could not parse {path}: {exc}", file=sys.stderr)
            sys.exit(1)


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------


def _word_lengths(records: list[Record]) -> list[int]:
    """Return a list of word-counts for each record's caption."""
    return [len(r["caption"].split()) for r in records if r.get("caption")]


def _percentile(sorted_data: list[float], pct: float) -> float:
    """Compute *pct*-th percentile (0–100) of an already-sorted list.

    Uses the nearest-rank method.

    Parameters
    ----------
    sorted_data:
        Sorted (ascending) list of numeric values.
    pct:
        Percentile to compute (e.g. 25, 50, 75).

    Returns
    -------
    float
        Percentile value.
    """
    if not sorted_data:
        return 0.0
    idx = max(0, math.ceil(pct / 100 * len(sorted_data)) - 1)
    return float(sorted_data[idx])


def caption_length_stats(records: list[Record]) -> dict[str, float]:
    """Compute caption-length statistics (in words).

    Parameters
    ----------
    records:
        List of record dicts.

    Returns
    -------
    dict[str, float]
        Keys: ``min``, ``max``, ``mean``, ``median``, ``p25``, ``p75``.
    """
    lengths = sorted(_word_lengths(records))
    if not lengths:
        return {"min": 0, "max": 0, "mean": 0, "median": 0, "p25": 0, "p75": 0}

    n = len(lengths)
    total = sum(lengths)
    return {
        "min": float(lengths[0]),
        "max": float(lengths[-1]),
        "mean": total / n,
        "median": _percentile(lengths, 50),
        "p25": _percentile(lengths, 25),
        "p75": _percentile(lengths, 75),
    }


def ocr_keyword_pct(records: list[Record]) -> float:
    """Return the percentage of captions containing at least one OCR keyword.

    Parameters
    ----------
    records:
        List of record dicts.

    Returns
    -------
    float
        Percentage (0–100).
    """
    if not records:
        return 0.0
    hits = sum(
        1
        for r in records
        if any(kw in r.get("caption", "").lower().split() for kw in _OCR_KEYWORDS)
    )
    return hits / len(records) * 100


def source_counts(records: list[Record]) -> dict[str, int]:
    """Count records per ``source`` field.

    Parameters
    ----------
    records:
        List of record dicts.

    Returns
    -------
    dict[str, int]
        Mapping of source name → record count.
    """
    counts: dict[str, int] = {}
    for r in records:
        src = r.get("source", "unknown")
        counts[src] = counts.get(src, 0) + 1
    return dict(sorted(counts.items()))


def sample_captions(records: list[Record], n: int = 5, seed: int = 0) -> dict[str, list[str]]:
    """Sample *n* captions per source.

    Parameters
    ----------
    records:
        List of record dicts.
    n:
        Number of captions to sample per source.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict[str, list[str]]
        Mapping of source → list of sampled captions.
    """
    rng = random.Random(seed)
    by_source: dict[str, list[str]] = {}
    for r in records:
        src = r.get("source", "unknown")
        by_source.setdefault(src, []).append(r.get("caption", ""))

    return {
        src: rng.sample(caps, min(n, len(caps)))
        for src, caps in sorted(by_source.items())
    }


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

_HR = "─" * 70


def print_split_report(split_name: str, records: list[Record]) -> None:
    """Print a full statistics report for one dataset split.

    Parameters
    ----------
    split_name:
        Human-readable split name (e.g. ``"TRAIN"``).
    records:
        Records belonging to this split.
    """
    print(f"\n{'═' * 70}")
    print(f"  {split_name}  ({len(records):,} records)")
    print(f"{'═' * 70}")

    # --- Per-source breakdown ---
    print("\n📊  Records per source:")
    counts = source_counts(records)
    for src, cnt in counts.items():
        bar_len = int(cnt / max(counts.values()) * 30)
        bar = "█" * bar_len
        pct = cnt / len(records) * 100 if records else 0
        print(f"    {src:12s}  {cnt:>8,}  ({pct:5.1f}%)  {bar}")

    # --- Caption length distribution ---
    print("\n📏  Caption length distribution (words):")
    stats = caption_length_stats(records)
    print(f"    {'min':>8}: {stats['min']:.0f}")
    print(f"    {'p25':>8}: {stats['p25']:.1f}")
    print(f"    {'median':>8}: {stats['median']:.1f}")
    print(f"    {'mean':>8}: {stats['mean']:.1f}")
    print(f"    {'p75':>8}: {stats['p75']:.1f}")
    print(f"    {'max':>8}: {stats['max']:.0f}")

    # --- OCR / text keyword presence ---
    pct_ocr = ocr_keyword_pct(records)
    print(f"\n🔤  Captions with text/OCR keywords: {pct_ocr:.1f}%")

    # --- Sample captions ---
    print("\n💬  Sample captions (5 per source):")
    samples = sample_captions(records, n=5)
    for src, caps in samples.items():
        print(f"\n    [{src}]")
        for i, cap in enumerate(caps, start=1):
            # Wrap long lines for readability
            wrapped = cap if len(cap) <= 80 else cap[:77] + "…"
            print(f"      {i}. {wrapped}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Print statistics for the processed AccessiCaption dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data/processed",
        help="Directory containing train.json and eval.json.",
    )
    parser.add_argument(
        "--sample_seed",
        type=int,
        default=0,
        help="Random seed for caption sampling.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the statistics script.

    Parameters
    ----------
    argv:
        CLI arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    data_dir = Path(args.data_dir)
    train_path = data_dir / "train.json"
    eval_path = data_dir / "eval.json"

    print(f"\nAccessiCaption — Dataset Statistics")
    print(f"Data directory: {data_dir.resolve()}")

    train_records = load_split(train_path)
    eval_records = load_split(eval_path)

    total = len(train_records) + len(eval_records)
    print(f"\nTotal records across all splits: {total:,}")

    print_split_report("TRAIN", train_records)
    print_split_report("EVAL", eval_records)

    print(f"\n{_HR}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
