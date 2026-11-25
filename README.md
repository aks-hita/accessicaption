# AccessiCaption 🦾

> **WCAG-compliant image alt-text generation — built for visual disability aid**

A fine-tuned Vision-Language Model (VLM) that takes any image and generates descriptive alt-text captions conforming to WCAG 2.1/2.2 §1.1.1, with multi-level detail tiers and confidence scoring.

## Architecture

```
Image Input
    │
    ▼
LLaVA-1.5-7B + QLoRA Adapter (fine-tuned on COCO + VizWiz)
    │
    ▼
WCAG Normalizer (strip "image of", active voice, length clamp)
    │
    ▼
Confidence Estimator (logit entropy → 0–1 score)
    │
    ▼
CaptionResult {caption, confidence, flagged, latency_ms}
```

## Project Structure

```
accessicaption/
├── configs/
│   └── config.yaml          # All hyperparameters
├── data/
│   ├── prepare_dataset.py   # Download & process COCO + VizWiz
│   ├── wcag_normalize.py    # WCAG 2.1/2.2 caption normalization
│   ├── dataset_stats.py     # Dataset analysis
│   └── collator.py          # PyTorch Dataset + DataCollator
├── training/
│   ├── train.py             # QLoRA fine-tuning
│   └── evaluate.py          # BLEU / CIDEr / METEOR / ROUGE-L
├── inference/
│   ├── captioner.py         # Captioner class (inference wrapper)
│   └── confidence.py        # Confidence estimation methods
├── spaces/
│   └── app.py               # HuggingFace Spaces Gradio demo
├── tests/
│   └── test_core.py         # Core test suite
├── main.py                  # CLI demo entry point
├── requirements.txt
├── .env.example
├── .gitignore
└── README.md
```

## Quickstart

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare data

```bash
python data/prepare_dataset.py --output_dir ./data/processed --cache_dir ./data/cache
```

### 3. Fine-tune the model

```bash
python training/train.py --config configs/config.yaml
```

### 4. Evaluate

```bash
python training/evaluate.py \
  --model_path ./outputs/accessicaption-v1/merged \
  --eval_data ./data/processed/eval.json
```

### 5. Run CLI inference

```bash
python main.py --image /path/to/image.jpg --model_path llava-hf/llava-1.5-7b-hf
```

### 6. Run the Gradio demo

```bash
python spaces/app.py
```

## Model Details

| Property | Value |
|---|---|
| Base model | LLaVA-1.5-7B (`llava-hf/llava-1.5-7b-hf`) |
| Fine-tuning | QLoRA (4-bit NF4 + LoRA r=16, α=32) |
| Training data | COCO Captions 2017 + VizWiz (2× oversampled) |
| Output | 1–2 sentence WCAG-compliant alt-text |
| Languages | English (v1) |

## Target Metrics (v1)

| Metric | Target |
|---|---|
| CIDEr | ≥ 1.20 |
| BLEU-4 | ≥ 0.32 |
| METEOR | ≥ 0.28 |
| ROUGE-L | ≥ 0.52 |
| p95 Latency | ≤ 2 s (cloud GPU) |

## Compliance & Standards

- WCAG 2.1 / 2.2 §1.1.1 (alt-text guidelines)
- CVAA (21st Century Communications and Video Accessibility Act)
- Privacy-first: images processed in-memory only, never stored

## License

Apache 2.0
