"""Precompute and cache frozen CLIP ViT-B/32 patch-grid image features.

Used by 01_lightweight_tagger so CLIP never needs to run during training
(it's frozen anyway) -- one forward pass per unique image, cached to disk,
then looked up by `image_name` during training/inference. ~2,500 images x 50
tokens x 768 dims in fp16 is roughly 190 MB, small enough to load whole into
memory as a plain dict.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

DEFAULT_MODEL_NAME = "openai/clip-vit-base-patch32"


def build_image_cache(image_names, image_dir=data_io.IMAGE_DIR, output_path=None,
                       model_name=DEFAULT_MODEL_NAME, device=None, batch_size=16,
                       dtype=torch.float16):
    """Runs the frozen CLIP vision tower once per unique image in `image_names`.

    Returns (and optionally saves to `output_path`) a dict
    {image_name: Tensor[n_patch_tokens+1, hidden_dim]} (patch tokens + CLS,
    as returned by CLIPVisionModel's last_hidden_state).
    """
    from transformers import CLIPImageProcessor, CLIPVisionModel

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    processor = CLIPImageProcessor.from_pretrained(model_name)
    model = CLIPVisionModel.from_pretrained(model_name).to(device).eval()

    names = sorted(set(image_names))
    cache = {}
    unresolved = []

    from tqdm.auto import tqdm

    with torch.no_grad():
        pbar = tqdm(range(0, len(names), batch_size), desc="CLIP image cache", unit="batch", dynamic_ncols=True)
        for i in pbar:
            batch_names = names[i:i + batch_size]
            images, resolved_names = [], []
            for name in batch_names:
                path = data_io.resolve_image_path(name, image_dir)
                if path is None:
                    unresolved.append(name)
                    continue
                try:
                    images.append(Image.open(path).convert("RGB"))
                    resolved_names.append(name)
                except Exception:
                    unresolved.append(name)
            if not images:
                continue
            inputs = processor(images=images, return_tensors="pt").to(device)
            out = model(**inputs)
            patch_tokens = out.last_hidden_state  # [B, n_tokens, hidden]
            for name, tokens in zip(resolved_names, patch_tokens):
                cache[name] = tokens.to(dtype).cpu()
            pbar.set_postfix({"cached": len(cache), "unresolved": len(unresolved)})

    if unresolved:
        print(f"  WARNING: {len(unresolved)} images could not be resolved/opened, e.g. {unresolved[:5]}")

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(cache, output_path)
        print(f"Saved image cache ({len(cache)} images) -> {output_path}")

    return cache


def load_image_cache(path):
    return torch.load(path, map_location="cpu")


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--output", default=str(_REPO_ROOT / "solutions" / "_cache" / "clip_image_cache.pt"))
    p.add_argument("--limit", type=int, default=None, help="Only cache the first N unique images (smoke testing).")
    args = p.parse_args()

    train = data_io.load_jsonl(data_io.TRAIN_FILE)
    val = data_io.load_jsonl(data_io.VAL_FILE)
    names = sorted({r["image_name"] for r in train} | {r["image_name"] for r in val})
    if args.limit:
        names = names[: args.limit]
    print(f"Building CLIP image cache for {len(names)} unique images...")
    build_image_cache(names, output_path=args.output)
