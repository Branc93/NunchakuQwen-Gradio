"""Example script for generating images using quantized Qwen-Image models.

This example loads one of the quantized Qwen-Image models included in this
repository and generates a single image from a text prompt. The script mirrors
https://github.com/nunchaku-tech/nunchaku/blob/main/examples/v1/qwen-image.py
but is adapted to use the ``NunchakuModelLoader`` helper from this project and
will automatically download the model file if it is missing.

Usage:
    python examples/v1/qwen-image.py --prompt "A cute robot painting" \\
        --model svdq-int4_r32 --output output.png
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import requests
from tqdm import tqdm

# Allow running from repository root
sys.path.append(str(Path(__file__).resolve().parents[2]))

from model_loader import NunchakuModelLoader


def generate(prompt: str, model_key: str, output: Path, steps: int,
             width: int, height: int, cfg: float, seed: int) -> None:
    """Load the requested model and create an image."""

    loader = NunchakuModelLoader()

    model_info = loader.available_models.get(model_key)
    if model_info is None:
        raise ValueError(f"Unknown model '{model_key}'. Available: {list(loader.available_models)}")

    model_path = loader.models_dir / model_info["file"]
    if not model_path.exists():
        url = model_info.get("url")
        if url is None:
            raise FileNotFoundError(
                f"Model file '{model_path}' not found and no download URL available."
            )
        model_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Downloading {model_key} model...")
        response = requests.get(url, stream=True)
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        with open(model_path, "wb") as f, tqdm(
            total=total, unit="B", unit_scale=True, desc=model_info["file"]
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

    if not loader.load_model(model_key, model_path):
        raise RuntimeError(f"Failed to load model '{model_key}'.")

    status, image = loader.generate_image(
        prompt=prompt,
        model_key=model_key,
        inference_steps=steps,
        width=width,
        height=height,
        cfg_value=cfg,
        seed=seed,
    )

    if image is None:
        raise RuntimeError(status)

    image.save(output)
    print(status)
    print(f"Image saved to {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a quantized Qwen-Image model")
    parser.add_argument("--prompt", required=True, help="Text prompt for image generation")
    parser.add_argument("--model", default="svdq-int4_r32",
                        help="Model key to use (default: svdq-int4_r32)")
    parser.add_argument("--output", type=Path, default=Path("generated.png"),
                        help="Output image file")
    parser.add_argument("--steps", type=int, default=20, help="Number of denoising steps")
    parser.add_argument("--width", type=int, default=512, help="Image width")
    parser.add_argument("--height", type=int, default=512, help="Image height")
    parser.add_argument("--cfg", type=float, default=7.5,
                        help="Classifier-free guidance value")
    parser.add_argument("--seed", type=int, default=-1, help="Random seed (-1 for random)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    generate(
        prompt=args.prompt,
        model_key=args.model,
        output=args.output,
        steps=args.steps,
        width=args.width,
        height=args.height,
        cfg=args.cfg,
        seed=args.seed,
    )
