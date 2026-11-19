"""
train.py — QLoRA fine-tuning script for AccessiCaption.

Fine-tunes llava-hf/llava-1.5-7b-hf with 4-bit quantization (QLoRA)
using PEFT LoRA adapters and HuggingFace Trainer.

Usage:
    python train.py --config ../configs/config.yaml
    python train.py --config ../configs/config.yaml --resume_from_checkpoint ./outputs/checkpoint-500
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def setup_logging(level: str = "INFO") -> None:
    """Configure root logger with a consistent format."""
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_config(path: str) -> dict[str, Any]:
    """Load and return a YAML config file as a plain dict.

    Args:
        path: Filesystem path to the YAML config file.

    Returns:
        Parsed configuration dictionary.
    """
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with config_path.open("r") as fh:
        config = yaml.safe_load(fh)
    logger.info("Loaded config from %s", config_path)
    return config


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tuning for AccessiCaption (LLaVA-1.5-7B)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=str,
        default="../configs/config.yaml",
        help="Path to the YAML configuration file.",
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help="Path to a previous checkpoint directory to resume from.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Model & quantization
# ---------------------------------------------------------------------------

def build_bnb_config() -> "BitsAndBytesConfig":  # noqa: F821
    """Build the 4-bit BitsAndBytes quantization configuration.

    Returns:
        A ``BitsAndBytesConfig`` ready for passing to ``from_pretrained``.
    """
    from transformers import BitsAndBytesConfig

    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True,
    )


def build_lora_config(cfg: dict[str, Any]) -> "LoraConfig":  # noqa: F821
    """Construct a PEFT LoraConfig from the parsed YAML config.

    Args:
        cfg: The top-level config dictionary (expects a ``lora`` sub-key).

    Returns:
        A ``LoraConfig`` instance.
    """
    from peft import LoraConfig, TaskType

    lora_cfg = cfg.get("lora", {})
    return LoraConfig(
        r=lora_cfg.get("r", 16),
        lora_alpha=lora_cfg.get("alpha", 32),
        target_modules=lora_cfg.get(
            "target_modules",
            ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        ),
        lora_dropout=lora_cfg.get("dropout", 0.05),
        bias=lora_cfg.get("bias", "none"),
        task_type=TaskType.CAUSAL_LM,
    )


def load_model_and_processor(
    model_id: str,
    bnb_config: "BitsAndBytesConfig",  # noqa: F821
) -> tuple[Any, Any]:
    """Load the base LLaVA model with 4-bit quantization and its processor.

    Args:
        model_id: HuggingFace model identifier (e.g. ``llava-hf/llava-1.5-7b-hf``).
        bnb_config: Pre-built ``BitsAndBytesConfig``.

    Returns:
        Tuple of ``(model, processor)``.
    """
    from transformers import LlavaForConditionalGeneration, LlavaProcessor

    logger.info("Loading processor from %s …", model_id)
    processor = LlavaProcessor.from_pretrained(model_id)

    logger.info("Loading model from %s with 4-bit quantization …", model_id)
    model = LlavaForConditionalGeneration.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
    )
    model.config.use_cache = False  # required for gradient checkpointing
    logger.info("Model loaded. Parameter count: %s", f"{sum(p.numel() for p in model.parameters()):,}")
    return model, processor


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def build_datasets(
    cfg: dict[str, Any],
    processor: Any,
) -> tuple[Any, Any]:
    """Instantiate train and eval ``AccessiCaptionDataset`` instances.

    Args:
        cfg: Top-level config dictionary (expects ``data`` sub-key).
        processor: The LlavaProcessor used for tokenisation.

    Returns:
        Tuple of ``(train_dataset, eval_dataset)``.
    """
    # Ensure the data module is importable regardless of CWD
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    from data.collator import AccessiCaptionDataset  # type: ignore[import]

    data_cfg = cfg.get("data", {})
    train_json = data_cfg.get("train_json", "../data/processed/train.json")
    eval_json = data_cfg.get("eval_json", "../data/processed/eval.json")
    image_dir = data_cfg.get("image_dir", "../data/images")

    logger.info("Building train dataset from %s …", train_json)
    train_dataset = AccessiCaptionDataset(
        annotations_file=train_json,
        image_dir=image_dir,
        processor=processor,
    )
    logger.info("Building eval dataset from %s …", eval_json)
    eval_dataset = AccessiCaptionDataset(
        annotations_file=eval_json,
        image_dir=image_dir,
        processor=processor,
    )
    logger.info(
        "Dataset sizes — train: %d  |  eval: %d",
        len(train_dataset),
        len(eval_dataset),
    )
    return train_dataset, eval_dataset


# ---------------------------------------------------------------------------
# Training arguments
# ---------------------------------------------------------------------------

def build_training_args(cfg: dict[str, Any], output_dir: str) -> "TrainingArguments":  # noqa: F821
    """Build HuggingFace ``TrainingArguments`` from config.

    Args:
        cfg: Top-level config dictionary (expects ``training`` sub-key).
        output_dir: Directory where checkpoints will be written.

    Returns:
        A populated ``TrainingArguments`` instance.
    """
    from transformers import TrainingArguments

    t = cfg.get("training", {})
    return TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=t.get("epochs", 3),
        per_device_train_batch_size=t.get("per_device_train_batch_size", 2),
        per_device_eval_batch_size=t.get("per_device_eval_batch_size", 2),
        gradient_accumulation_steps=t.get("gradient_accumulation_steps", 8),
        learning_rate=float(t.get("learning_rate", 2e-4)),
        lr_scheduler_type=t.get("lr_scheduler_type", "cosine"),
        warmup_ratio=t.get("warmup_ratio", 0.03),
        fp16=True,
        bf16=False,
        dataloader_num_workers=t.get("dataloader_num_workers", 2),
        evaluation_strategy=t.get("evaluation_strategy", "steps"),
        eval_steps=t.get("eval_steps", 200),
        save_strategy=t.get("save_strategy", "steps"),
        save_steps=t.get("save_steps", 200),
        save_total_limit=t.get("save_total_limit", 3),
        load_best_model_at_end=t.get("load_best_model_at_end", True),
        logging_dir=os.path.join(output_dir, "logs"),
        logging_steps=t.get("logging_steps", 10),
        report_to=["wandb"] if t.get("use_wandb", True) else ["none"],
        run_name=t.get("wandb_run_name", "accessicaption-qlora"),
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        ddp_find_unused_parameters=False,
        group_by_length=True,
        remove_unused_columns=False,  # required for multimodal datasets
    )


# ---------------------------------------------------------------------------
# WandB setup
# ---------------------------------------------------------------------------

def setup_wandb(cfg: dict[str, Any]) -> None:
    """Initialise Weights & Biases if enabled in config.

    Args:
        cfg: Top-level config dictionary (expects ``wandb`` sub-key).
    """
    if not cfg.get("training", {}).get("use_wandb", True):
        os.environ["WANDB_DISABLED"] = "true"
        logger.info("WandB logging disabled via config.")
        return

    try:
        import wandb  # type: ignore

        wandb_cfg = cfg.get("wandb", {})
        wandb.init(
            project=wandb_cfg.get("project", "accessicaption"),
            entity=wandb_cfg.get("entity", None),
            name=wandb_cfg.get("run_name", cfg.get("training", {}).get("wandb_run_name", "accessicaption-qlora")),
            tags=wandb_cfg.get("tags", ["qlora", "llava", "alt-text", "wcag"]),
            config=cfg,
        )
        logger.info("WandB run initialised: %s", wandb.run.name)
    except ImportError:
        logger.warning("wandb not installed — skipping WandB initialisation.")
        os.environ["WANDB_DISABLED"] = "true"


# ---------------------------------------------------------------------------
# Post-training: merge adapter → full model
# ---------------------------------------------------------------------------

def merge_and_save(
    peft_model: Any,
    merged_output_dir: str,
    processor: Any,
) -> None:
    """Merge LoRA weights into the base model and persist to disk.

    Args:
        peft_model: The PEFT-wrapped model after training.
        merged_output_dir: Destination directory for the merged model.
        processor: LlavaProcessor to save alongside the merged model.
    """
    logger.info("Merging LoRA adapter into base model …")
    from peft import PeftModel  # noqa: F401

    merged_model = peft_model.merge_and_unload()
    Path(merged_output_dir).mkdir(parents=True, exist_ok=True)
    merged_model.save_pretrained(merged_output_dir)
    processor.save_pretrained(merged_output_dir)
    logger.info("Merged model saved to: %s", merged_output_dir)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Entry point: orchestrate the full QLoRA training pipeline."""
    args = parse_args()
    cfg = load_config(args.config)

    setup_logging(cfg.get("logging", {}).get("level", "INFO"))
    setup_wandb(cfg)

    # ------------------------------------------------------------------ paths
    output_root = cfg.get("output", {}).get("root", "outputs/accessicaption-v1")
    adapter_dir = os.path.join(output_root, "final_adapter")
    merged_dir = os.path.join(output_root, "merged")
    checkpoint_dir = os.path.join(output_root, "checkpoints")

    # --------------------------------------------------------------- quantise
    bnb_config = build_bnb_config()

    # ------------------------------------------------------------------ model
    model_id = cfg.get("model", {}).get("base_model_id", "llava-hf/llava-1.5-7b-hf")
    model, processor = load_model_and_processor(model_id, bnb_config)

    # ------------------------------------------------------------------- LoRA
    from peft import get_peft_model, prepare_model_for_kbit_training

    logger.info("Preparing model for k-bit training …")
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    lora_config = build_lora_config(cfg)
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ---------------------------------------------------------------- dataset
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))

    train_dataset, eval_dataset = build_datasets(cfg, processor)

    from data.collator import DataCollatorForVLMCaptioning  # type: ignore[import]

    collator = DataCollatorForVLMCaptioning(processor=processor)

    # ------------------------------------------------------- training args
    training_args = build_training_args(cfg, checkpoint_dir)

    # ---------------------------------------------------------------- Trainer
    from transformers import Trainer

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
    )

    # ------------------------------------------------------------------ train
    logger.info("Starting training …")
    try:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    except torch.cuda.OutOfMemoryError as oom_err:
        logger.error(
            "CUDA OutOfMemoryError encountered during training!\n"
            "  Suggestions:\n"
            "    • Reduce per_device_train_batch_size in config.yaml\n"
            "    • Increase gradient_accumulation_steps to compensate\n"
            "    • Reduce LoRA rank (r) in config.yaml\n"
            "    • Enable gradient checkpointing (already on by default)\n"
            "Original error: %s",
            oom_err,
        )
        sys.exit(1)

    # --------------------------------------------------------- save adapter
    logger.info("Saving final LoRA adapter to %s …", adapter_dir)
    Path(adapter_dir).mkdir(parents=True, exist_ok=True)
    model.save_pretrained(adapter_dir)
    processor.save_pretrained(adapter_dir)

    # --------------------------------------------------------- merge & save
    merge_and_save(model, merged_dir, processor)

    logger.info("Training complete. ✓")
    logger.info("  Adapter  → %s", adapter_dir)
    logger.info("  Merged   → %s", merged_dir)


if __name__ == "__main__":
    main()
