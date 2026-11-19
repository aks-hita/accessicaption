"""
prepare_dataset.py
------------------
Downloads and preprocesses COCO Captions 2017 and VizWiz Captions datasets,
merges them into a unified JSON format with WCAG-normalized captions, applies
VizWiz oversampling, and saves train/eval splits.

Usage
-----
    python prepare_dataset.py --output_dir ./data/processed --cache_dir ./data/cache

Dependencies
------------
    datasets, tqdm, Pillow (PIL), requests
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Local module — must be in the same directory or on PYTHONPATH
from wcag_normalize import normalize_caption

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("prepare_dataset")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VIZWIZ_OVERSAMPLE_FACTOR: int = 2  # Duplicate VizWiz train records this many times
TRAIN_RATIO: float = 0.90
EVAL_RATIO: float = 0.10
MAX_RETRIES: int = 3
RETRY_BACKOFF_BASE: float = 2.0  # seconds; exponential back-off

# HuggingFace dataset identifiers
COCO_DATASET_ID: str = "HuggingFaceM4/COCO"
VIZWIZ_TRAIN_DATASET_ID: str = "Multimodal-Fatima/VizWiz_train"
VIZWIZ_VAL_DATASET_ID: str = "Multimodal-Fatima/VizWiz_val"


# ---------------------------------------------------------------------------
# Record type alias
# ---------------------------------------------------------------------------

Record = dict[str, Any]


# ---------------------------------------------------------------------------
# Download helpers
# ---------------------------------------------------------------------------


def _load_with_retry(
    dataset_id: str,
    split: str,
    cache_dir: str,
    *,
    trust_remote_code: bool = True,
    extra_kwargs: dict[str, Any] | None = None,
) -> Any:
    """Load a HuggingFace dataset split with exponential-back-off retries.

    Parameters
    ----------
    dataset_id:
        HuggingFace dataset identifier (e.g. ``"HuggingFaceM4/COCO"``).
    split:
        Dataset split name (e.g. ``"train"`` or ``"validation"``).
    cache_dir:
        Local directory to cache downloaded data.
    trust_remote_code:
        Passed through to ``datasets.load_dataset``.
    extra_kwargs:
        Additional keyword arguments forwarded to ``datasets.load_dataset``.

    Returns
    -------
    datasets.Dataset
        The loaded dataset split.

    Raises
    ------
    RuntimeError
        If all retries are exhausted.
    """
    from datasets import load_dataset  # type: ignore[import]

    kwargs: dict[str, Any] = {
        "cache_dir": cache_dir,
        "trust_remote_code": trust_remote_code,
        **(extra_kwargs or {}),
    }

    last_exc: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            log.info(
                "Loading '%s' split='%s' (attempt %d/%d)…",
                dataset_id,
                split,
                attempt,
                MAX_RETRIES,
            )
            ds = load_dataset(dataset_id, split=split, **kwargs)
            log.info("  → %d examples loaded.", len(ds))
            return ds
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            wait = RETRY_BACKOFF_BASE**attempt
            log.warning(
                "  Failed (attempt %d/%d): %s. Retrying in %.0fs…",
                attempt,
                MAX_RETRIES,
                exc,
                wait,
            )
            time.sleep(wait)

    raise RuntimeError(
        f"Could not load '{dataset_id}' split='{split}' after {MAX_RETRIES} attempts."
    ) from last_exc


# ---------------------------------------------------------------------------
# COCO Captions
# ---------------------------------------------------------------------------


def _extract_coco_caption(captions_field: Any) -> str:
    """Extract the first usable caption from a COCO captions field.

    COCO stores multiple captions per image; we take the first non-empty one.

    Parameters
    ----------
    captions_field:
        Value stored in the ``captions`` column, which may be a list of dicts,
        a list of strings, a dict, or a plain string depending on dataset
        version.

    Returns
    -------
    str
        A single raw caption string.
    """
    if isinstance(captions_field, str):
        return captions_field.strip()
    if isinstance(captions_field, dict):
        # {"raw": "...", ...}
        for key in ("raw", "text", "caption"):
            if key in captions_field and captions_field[key]:
                return str(captions_field[key]).strip()
        return str(next(iter(captions_field.values()), "")).strip()
    if isinstance(captions_field, list):
        for item in captions_field:
            if isinstance(item, str) and item.strip():
                return item.strip()
            if isinstance(item, dict):
                for key in ("raw", "text", "caption"):
                    if key in item and item[key]:
                        return str(item[key]).strip()
    return ""


def load_coco(cache_dir: str) -> list[Record]:
    """Download and convert COCO Captions 2017 (train + val) to unified records.

    Parameters
    ----------
    cache_dir:
        HuggingFace cache directory.

    Returns
    -------
    list[Record]
        Unified records with keys: ``id``, ``image_path``, ``caption``,
        ``source``, ``split``.
    """
    records: list[Record] = []

    split_map = {"train": "train", "validation": "validation"}
    unified_split_map = {"train": "train", "validation": "eval"}

    for hf_split, unified_split in split_map.items():
        try:
            ds = _load_with_retry(
                COCO_DATASET_ID,
                hf_split,
                cache_dir,
                trust_remote_code=True,
            )
        except RuntimeError as exc:
            log.error("Skipping COCO %s: %s", hf_split, exc)
            continue

        for example in tqdm(ds, desc=f"COCO {hf_split}", unit="img"):
            # Image: may be a PIL.Image or a path string
            image_obj = example.get("image") or example.get("img")
            image_path: str = ""
            if hasattr(image_obj, "filename") and image_obj.filename:
                image_path = str(image_obj.filename)
            elif isinstance(image_obj, str):
                image_path = image_obj

            # Captions field varies by dataset version
            raw_caption = _extract_coco_caption(
                example.get("captions") or example.get("caption") or ""
            )
            if not raw_caption:
                continue

            image_id = example.get("image_id") or example.get("id") or len(records)

            records.append(
                {
                    "id": f"coco_{image_id}",
                    "image_path": image_path,
                    "caption": normalize_caption(raw_caption),
                    "source": "coco",
                    "split": unified_split_map[hf_split],
                }
            )

    log.info("COCO: %d records collected.", len(records))
    return records


# ---------------------------------------------------------------------------
# VizWiz Captions
# ---------------------------------------------------------------------------


def _extract_vizwiz_caption(example: dict[str, Any]) -> str:
    """Extract the best caption from a VizWiz example.

    VizWiz may have multiple annotator answers; select the most common
    non-'unanswerable' one, or fall back to the first.

    Parameters
    ----------
    example:
        A single VizWiz dataset example.

    Returns
    -------
    str
        A single raw caption string.
    """
    # Try dedicated caption field first
    for key in ("caption", "text", "answer"):
        val = example.get(key)
        if isinstance(val, str) and val.strip() and val.lower() != "unanswerable":
            return val.strip()

    # Fallback: list of answers
    answers = example.get("answers") or []
    if isinstance(answers, list):
        texts = [
            a.get("answer", "") if isinstance(a, dict) else str(a)
            for a in answers
            if (a.get("answer", "") if isinstance(a, dict) else str(a)).lower()
            not in {"unanswerable", ""}
        ]
        if texts:
            # Return most common
            from collections import Counter

            return Counter(texts).most_common(1)[0][0]

    return ""


def load_vizwiz(cache_dir: str) -> list[Record]:
    """Download and convert VizWiz Captions (train + val) to unified records.

    Parameters
    ----------
    cache_dir:
        HuggingFace cache directory.

    Returns
    -------
    list[Record]
        Unified records with keys: ``id``, ``image_path``, ``caption``,
        ``source``, ``split``.
    """
    records: list[Record] = []

    dataset_map = {
        VIZWIZ_TRAIN_DATASET_ID: "train",
        VIZWIZ_VAL_DATASET_ID: "eval",
    }

    for dataset_id, unified_split in dataset_map.items():
        try:
            ds = _load_with_retry(
                dataset_id,
                "train",  # Both VizWiz HF datasets expose a single 'train' split
                cache_dir,
                trust_remote_code=True,
            )
        except RuntimeError as exc:
            log.error("Skipping VizWiz (%s): %s", dataset_id, exc)
            continue

        for idx, example in enumerate(tqdm(ds, desc=f"VizWiz {unified_split}", unit="img")):
            image_obj = example.get("image") or example.get("img")
            image_path: str = ""
            if hasattr(image_obj, "filename") and image_obj.filename:
                image_path = str(image_obj.filename)
            elif isinstance(image_obj, str):
                image_path = image_obj

            raw_caption = _extract_vizwiz_caption(example)
            if not raw_caption:
                continue

            image_id = example.get("image_id") or example.get("id") or f"{idx}"

            records.append(
                {
                    "id": f"vizwiz_{image_id}",
                    "image_path": image_path,
                    "caption": normalize_caption(raw_caption),
                    "source": "vizwiz",
                    "split": unified_split,
                }
            )

    log.info("VizWiz: %d records collected.", len(records))
    return records


# ---------------------------------------------------------------------------
# Oversampling
# ---------------------------------------------------------------------------


def oversample_vizwiz(records: list[Record], factor: int = VIZWIZ_OVERSAMPLE_FACTOR) -> list[Record]:
    """Duplicate VizWiz *train* records by *factor* (default 2×).

    Only training records are oversampled; eval records are left unchanged to
    avoid data leakage.

    Parameters
    ----------
    records:
        Combined dataset records (may include COCO and VizWiz, any split).
    factor:
        Duplication multiplier for VizWiz train records.

    Returns
    -------
    list[Record]
        Records with VizWiz train entries duplicated.
    """
    oversampled: list[Record] = []
    for rec in records:
        oversampled.append(rec)
        if rec["source"] == "vizwiz" and rec["split"] == "train":
            for copy_idx in range(1, factor):
                dup = dict(rec)
                dup["id"] = f"{rec['id']}_os{copy_idx}"
                oversampled.append(dup)
    log.info(
        "After %dx VizWiz oversampling: %d → %d records.",
        factor,
        len(records),
        len(oversampled),
    )
    return oversampled


# ---------------------------------------------------------------------------
# Split logic
# ---------------------------------------------------------------------------


def split_dataset(
    records: list[Record],
    train_ratio: float = TRAIN_RATIO,
    seed: int = 42,
) -> tuple[list[Record], list[Record]]:
    """Re-partition all records into train / eval splits.

    Records that already carry a ``split`` of ``"eval"`` are preferentially
    assigned to the eval set.  The remainder are shuffled and split by
    *train_ratio*.

    Parameters
    ----------
    records:
        All merged, oversampled records.
    train_ratio:
        Fraction (0–1) of records to use for training.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    tuple[list[Record], list[Record]]
        ``(train_records, eval_records)``
    """
    rng = random.Random(seed)

    native_eval = [r for r in records if r["split"] == "eval"]
    train_pool = [r for r in records if r["split"] == "train"]

    rng.shuffle(train_pool)
    split_idx = int(len(train_pool) * train_ratio)
    train_records = train_pool[:split_idx]
    held_out = train_pool[split_idx:]

    eval_records = native_eval + held_out

    # Update the split field to reflect actual assignment
    for r in train_records:
        r["split"] = "train"
    for r in eval_records:
        r["split"] = "eval"

    log.info("Final split — train: %d, eval: %d", len(train_records), len(eval_records))
    return train_records, eval_records


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------


def print_statistics(train: list[Record], eval_: list[Record]) -> None:
    """Print dataset composition statistics to stdout.

    Parameters
    ----------
    train:
        Training split records.
    eval_:
        Evaluation split records.
    """
    for split_name, split_records in [("train", train), ("eval", eval_)]:
        from collections import Counter

        source_counts = Counter(r["source"] for r in split_records)
        caption_lengths = [len(r["caption"].split()) for r in split_records]
        avg_len = sum(caption_lengths) / len(caption_lengths) if caption_lengths else 0

        print(f"\n{'─' * 60}")
        print(f"  Split: {split_name.upper()} — {len(split_records):,} records")
        print(f"{'─' * 60}")
        for src, count in sorted(source_counts.items()):
            print(f"  {src:12s}: {count:>8,}  ({count / len(split_records) * 100:.1f}%)")
        print(f"  {'avg caption':12s}: {avg_len:.1f} words")
    print()


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------


def save_json(records: list[Record], path: Path) -> None:
    """Write *records* to a JSON file at *path*.

    Parameters
    ----------
    records:
        List of unified record dicts.
    path:
        Destination file path; parent directories are created as needed.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        json.dump(records, fh, ensure_ascii=False, indent=2)
    log.info("Saved %d records → %s", len(records), path)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    """Build and return the CLI argument parser."""
    parser = argparse.ArgumentParser(
        description="Prepare AccessiCaption dataset from COCO and VizWiz.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./data/processed",
        help="Directory where train.json and eval.json will be saved.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="./data/cache",
        help="HuggingFace datasets cache directory.",
    )
    parser.add_argument(
        "--oversample_factor",
        type=int,
        default=VIZWIZ_OVERSAMPLE_FACTOR,
        help="VizWiz train oversampling multiplier.",
    )
    parser.add_argument(
        "--train_ratio",
        type=float,
        default=TRAIN_RATIO,
        help="Fraction of training-pool records to assign to train split.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--skip_coco",
        action="store_true",
        help="Skip COCO download (useful for debugging with VizWiz only).",
    )
    parser.add_argument(
        "--skip_vizwiz",
        action="store_true",
        help="Skip VizWiz download (useful for debugging with COCO only).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Entry point for the data preparation pipeline.

    Parameters
    ----------
    argv:
        Command-line arguments (defaults to ``sys.argv[1:]``).

    Returns
    -------
    int
        Exit code (0 = success).
    """
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    output_dir = Path(args.output_dir)
    cache_dir = str(Path(args.cache_dir).resolve())

    all_records: list[Record] = []

    # 1. Download COCO
    if not args.skip_coco:
        log.info("=== Downloading COCO Captions 2017 ===")
        coco_records = load_coco(cache_dir)
        all_records.extend(coco_records)
    else:
        log.info("Skipping COCO (--skip_coco).")

    # 2. Download VizWiz
    if not args.skip_vizwiz:
        log.info("=== Downloading VizWiz Captions ===")
        vizwiz_records = load_vizwiz(cache_dir)
        all_records.extend(vizwiz_records)
    else:
        log.info("Skipping VizWiz (--skip_vizwiz).")

    if not all_records:
        log.error("No records loaded. Exiting.")
        return 1

    log.info("Total records before oversampling: %d", len(all_records))

    # 3. Oversample VizWiz train
    all_records = oversample_vizwiz(all_records, factor=args.oversample_factor)

    # 4. Split
    train_records, eval_records = split_dataset(
        all_records, train_ratio=args.train_ratio, seed=args.seed
    )

    # 5. Save
    save_json(train_records, output_dir / "train.json")
    save_json(eval_records, output_dir / "eval.json")

    # 6. Statistics
    print_statistics(train_records, eval_records)

    log.info("Dataset preparation complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
