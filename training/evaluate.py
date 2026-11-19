"""
evaluate.py — Evaluation script for AccessiCaption fine-tuned models.

Runs inference over the eval split and computes BLEU-1/2/4, METEOR,
ROUGE-L, and optionally CIDEr metrics.

Usage:
    python evaluate.py \\
        --model_path ./outputs/accessicaption-v1/merged \\
        --eval_data ../data/processed/eval.json \\
        --batch_size 8

    # Using base model + LoRA adapter instead of merged checkpoint:
    python evaluate.py \\
        --model_path llava-hf/llava-1.5-7b-hf \\
        --adapter_path ./outputs/accessicaption-v1/final_adapter \\
        --eval_data ../data/processed/eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt template (must match train.py exactly)
# ---------------------------------------------------------------------------

PROMPT_TEMPLATE = (
    "USER: <image>\n"
    "Describe this image in one to two sentences as alt-text for a visually impaired person. "
    "Focus on the main subject, context, and any important text visible. "
    "Do not start with 'Image of' or 'Photo of'.\n"
    "ASSISTANT:"
)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments for the evaluation script."""
    parser = argparse.ArgumentParser(
        description="Evaluate an AccessiCaption checkpoint with BLEU / METEOR / ROUGE-L / CIDEr.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the merged model directory, OR the base HF model ID when using --adapter_path.",
    )
    parser.add_argument(
        "--adapter_path",
        type=str,
        default=None,
        help="(Optional) Path to a LoRA adapter directory. When provided, --model_path is treated "
             "as the base model ID and the adapter is loaded on top.",
    )
    parser.add_argument(
        "--eval_data",
        type=str,
        default="../data/processed/eval.json",
        help="Path to the eval JSON file (list of {image_id, image_path, caption} dicts).",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=8,
        help="Inference batch size.",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=128,
        help="Maximum new tokens to generate per sample.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/eval_results.json",
        help="Destination JSON file for numeric metric results.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=10,
        help="Number of sample rows to print to stdout.",
    )
    parser.add_argument(
        "--log_level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_model_and_processor(
    model_path: str,
    adapter_path: str | None = None,
) -> tuple[Any, Any]:
    """Load the evaluation model and its processor.

    Supports two modes:
    1. ``model_path`` points to a merged model directory — loaded directly.
    2. ``model_path`` is a base model ID and ``adapter_path`` is a LoRA
       adapter — the adapter is applied on top of the 4-bit quantised base.

    Args:
        model_path: Merged model directory path or HF base model ID.
        adapter_path: Optional LoRA adapter directory path.

    Returns:
        Tuple ``(model, processor)`` ready for inference.
    """
    from transformers import BitsAndBytesConfig, LlavaForConditionalGeneration, LlavaProcessor

    logger.info("Loading processor from %s …", model_path)
    processor = LlavaProcessor.from_pretrained(model_path)

    if adapter_path:
        logger.info("Loading base model '%s' with 4-bit quantisation …", model_path)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            quantization_config=bnb_config,
            device_map="auto",
            torch_dtype=torch.float16,
        )
        from peft import PeftModel

        logger.info("Applying LoRA adapter from %s …", adapter_path)
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    else:
        logger.info("Loading merged model from %s …", model_path)
        model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.float16,
        )

    model.eval()
    return model, processor


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_eval_data(eval_data_path: str) -> list[dict[str, Any]]:
    """Load evaluation records from a JSON file.

    Expected format: a list of objects, each with at minimum:
    ``{"image_id": ..., "image_path": "...", "caption": "..."}``

    Args:
        eval_data_path: Filesystem path to the eval JSON file.

    Returns:
        List of record dictionaries.
    """
    path = Path(eval_data_path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Eval data not found: {path}")
    with path.open("r", encoding="utf-8") as fh:
        records = json.load(fh)
    logger.info("Loaded %d eval records from %s", len(records), path)
    return records


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

def run_inference(
    model: Any,
    processor: Any,
    records: list[dict[str, Any]],
    batch_size: int = 8,
    max_new_tokens: int = 128,
) -> list[str]:
    """Generate captions for all eval records using batched inference.

    Args:
        model: The loaded LLaVA model (eval mode, on GPU).
        processor: The corresponding ``LlavaProcessor``.
        records: List of eval record dicts (must contain ``image_path``).
        batch_size: Number of images to process in one forward pass.
        max_new_tokens: Maximum number of tokens to generate per image.

    Returns:
        List of generated caption strings, one per record (same order).
    """
    from PIL import Image  # type: ignore

    predictions: list[str] = []
    total = len(records)

    for start in range(0, total, batch_size):
        batch = records[start : start + batch_size]
        logger.debug("Inference batch %d–%d / %d", start + 1, start + len(batch), total)

        images: list[Any] = []
        for rec in batch:
            img_path = Path(rec["image_path"]).resolve()
            try:
                images.append(Image.open(img_path).convert("RGB"))
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not open image %s: %s — using blank placeholder.", img_path, exc)
                images.append(Image.new("RGB", (336, 336), color=(0, 0, 0)))

        prompts = [PROMPT_TEMPLATE] * len(batch)

        inputs = processor(
            text=prompts,
            images=images,
            return_tensors="pt",
            padding=True,
        ).to(model.device)

        with torch.inference_mode():
            generated_ids = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
            )

        # Decode only the newly generated tokens (skip the prompt)
        input_len = inputs["input_ids"].shape[1]
        for gen_ids in generated_ids:
            new_tokens = gen_ids[input_len:]
            caption = processor.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
            predictions.append(caption)

        if (start // batch_size + 1) % 10 == 0:
            logger.info("  … %d / %d samples processed", start + len(batch), total)

    logger.info("Inference complete. %d captions generated.", len(predictions))
    return predictions


# ---------------------------------------------------------------------------
# Metric computation
# ---------------------------------------------------------------------------

def compute_bleu(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute BLEU-1, BLEU-2, and BLEU-4 using sacrebleu.

    Args:
        hypotheses: Model-generated captions.
        references: Ground-truth captions.

    Returns:
        Dictionary with keys ``bleu_1``, ``bleu_2``, ``bleu_4``.
    """
    import sacrebleu  # type: ignore

    # sacrebleu expects references as a list of lists (one list per ref level)
    refs = [references]
    b1 = sacrebleu.corpus_bleu(hypotheses, refs, max_ngram_order=1).score
    b2 = sacrebleu.corpus_bleu(hypotheses, refs, max_ngram_order=2).score
    b4 = sacrebleu.corpus_bleu(hypotheses, refs, max_ngram_order=4).score
    return {"bleu_1": round(b1, 4), "bleu_2": round(b2, 4), "bleu_4": round(b4, 4)}


def compute_meteor(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute corpus-level METEOR using nltk.

    Args:
        hypotheses: Model-generated captions.
        references: Ground-truth captions.

    Returns:
        Dictionary with key ``meteor``.
    """
    import nltk  # type: ignore
    from nltk.translate.meteor_score import meteor_score  # type: ignore

    nltk.download("wordnet", quiet=True)
    nltk.download("omw-1.4", quiet=True)
    nltk.download("punkt", quiet=True)

    scores = [
        meteor_score([ref.split()], hyp.split())
        for hyp, ref in zip(hypotheses, references)
    ]
    return {"meteor": round(sum(scores) / max(len(scores), 1), 4)}


def compute_rouge_l(
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float]:
    """Compute ROUGE-L using rouge_score (falls back to sacrebleu CHRF).

    Args:
        hypotheses: Model-generated captions.
        references: Ground-truth captions.

    Returns:
        Dictionary with key ``rouge_l``.
    """
    try:
        from rouge_score import rouge_scorer  # type: ignore

        scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
        scores = [
            scorer.score(ref, hyp)["rougeL"].fmeasure
            for hyp, ref in zip(hypotheses, references)
        ]
        return {"rouge_l": round(sum(scores) / max(len(scores), 1), 4)}
    except ImportError:
        logger.warning(
            "rouge_score not installed — falling back to sacrebleu CHRF as ROUGE-L proxy."
        )
        import sacrebleu  # type: ignore

        chrf = sacrebleu.corpus_chrf(hypotheses, [references]).score
        return {"rouge_l_chrf_proxy": round(chrf, 4)}


def compute_cider(
    image_ids: list[Any],
    hypotheses: list[str],
    references: list[str],
) -> dict[str, float] | None:
    """Compute CIDEr using pycocoevalcap if available.

    Args:
        image_ids: Image identifiers matching the order of hypotheses/references.
        hypotheses: Model-generated captions.
        references: Ground-truth captions.

    Returns:
        Dictionary with key ``cider``, or ``None`` if library is not installed.
    """
    try:
        from pycocoevalcap.cider.cider import Cider  # type: ignore

        gts: dict[Any, list[dict]] = {
            img_id: [{"caption": ref}] for img_id, ref in zip(image_ids, references)
        }
        res: dict[Any, list[dict]] = {
            img_id: [{"caption": hyp}] for img_id, hyp in zip(image_ids, hypotheses)
        }
        cider_scorer = Cider()
        score, _ = cider_scorer.compute_score(gts, res)
        return {"cider": round(float(score), 4)}
    except ImportError:
        logger.warning("pycocoevalcap not installed — skipping CIDEr computation.")
        return None


# ---------------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------------

def print_samples(
    records: list[dict[str, Any]],
    predictions: list[str],
    n: int = 10,
) -> None:
    """Print ``n`` sample rows of (image_id, ground_truth, predicted) to stdout.

    Args:
        records: Eval records list.
        predictions: Corresponding generated captions.
        n: How many samples to display.
    """
    print("\n" + "=" * 80)
    print(f"{'SAMPLE PREDICTIONS':^80}")
    print("=" * 80)
    for i, (rec, pred) in enumerate(zip(records[:n], predictions[:n])):
        image_id = rec.get("image_id", f"sample_{i}")
        gt = rec.get("caption", "").strip()
        print(f"\n[{i+1}] image_id : {image_id}")
        print(f"     ground truth : {gt}")
        print(f"     predicted    : {pred}")
    print("=" * 80 + "\n")


def save_results(metrics: dict[str, float], output_path: str) -> None:
    """Serialise metric results to a JSON file.

    Args:
        metrics: Dictionary of metric name → score.
        output_path: Destination file path.
    """
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)
    logger.info("Eval results saved to %s", out)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: load model, run inference, compute metrics, save results."""
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # ------------------------------------------------------------------ model
    model, processor = load_model_and_processor(args.model_path, args.adapter_path)

    # ------------------------------------------------------------------ data
    records = load_eval_data(args.eval_data)

    # --------------------------------------------------------------- infer
    predictions = run_inference(
        model,
        processor,
        records,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )

    references = [rec.get("caption", "") for rec in records]
    image_ids = [rec.get("image_id", i) for i, rec in enumerate(records)]

    # --------------------------------------------------------------- metrics
    all_metrics: dict[str, float] = {}

    logger.info("Computing BLEU …")
    all_metrics.update(compute_bleu(predictions, references))

    logger.info("Computing METEOR …")
    all_metrics.update(compute_meteor(predictions, references))

    logger.info("Computing ROUGE-L …")
    all_metrics.update(compute_rouge_l(predictions, references))

    logger.info("Computing CIDEr …")
    cider_result = compute_cider(image_ids, predictions, references)
    if cider_result:
        all_metrics.update(cider_result)

    # --------------------------------------------------------------- report
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("=" * 50)
    for metric, score in all_metrics.items():
        print(f"  {metric:<20}: {score:.4f}")
    print("=" * 50 + "\n")

    print_samples(records, predictions, n=args.num_samples)
    save_results(all_metrics, args.output)


if __name__ == "__main__":
    main()
