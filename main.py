"""
AccessiCaption — entry point for local testing.
Loads the captioner and runs it on a sample image URL.
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))


def demo_caption(image_source: str, model_path: str) -> None:
    """Run a single captioning demo."""
    try:
        from inference.captioner import Captioner, CaptionOptions
    except ImportError as e:
        print(f"[ERROR] Could not import Captioner: {e}")
        print("Make sure dependencies are installed: pip install -r requirements.txt")
        sys.exit(1)

    print(f"\n🔍 Loading model from: {model_path}")
    captioner = Captioner(model_path=model_path)
    captioner.warmup()

    print(f"📷 Captioning: {image_source}\n")

    import PIL.Image
    import requests
    from io import BytesIO

    if image_source.startswith("http"):
        headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
        response = requests.get(image_source, headers=headers, timeout=15)
        response.raise_for_status()
        image = PIL.Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = PIL.Image.open(image_source).convert("RGB")

    start = time.perf_counter()
    result = captioner.caption(image, CaptionOptions())
    elapsed = (time.perf_counter() - start) * 1000

    print("=" * 60)
    print(f"📝 Caption:    {result.caption}")
    print(f"🎯 Confidence: {result.confidence:.3f}")
    print(f"⚠️  Flagged:    {result.flagged}")
    print(f"⏱️  Latency:    {result.latency_ms:.1f} ms")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="AccessiCaption — WCAG-compliant image captioning demo"
    )
    parser.add_argument(
        "--image",
        type=str,
        default="https://upload.wikimedia.org/wikipedia/commons/thumb/4/47/PNG_transparency_demonstration_1.png/280px-PNG_transparency_demonstration_1.png",
        help="Image path or URL to caption",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="./outputs/accessicaption-v1/merged",
        help="Path to fine-tuned model (or HF model ID)",
    )
    args = parser.parse_args()
    demo_caption(args.image, args.model_path)


if __name__ == "__main__":
    main()
