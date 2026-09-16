"""End-to-end pipeline for solution 01 (lightweight tagger): build image
cache (if missing) -> train -> predict val -> score val -> write results.

Usage:
    python run_pipeline.py                 # full run
    python run_pipeline.py --smoke_test     # tiny run to sanity-check the pipeline
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from solutions.common import data_io, scorer_local
from solutions.common.image_cache import build_image_cache


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    image_cache_path = _THIS_DIR / "image_cache.pt"
    if not image_cache_path.exists():
        print("Building CLIP image cache (one-time)...")
        train = data_io.load_jsonl(data_io.TRAIN_FILE)
        val = data_io.load_jsonl(data_io.VAL_FILE)
        names = {r["image_name"] for r in train} | {r["image_name"] for r in val}
        if args.smoke_test:
            names = set(list(names)[:20])
        build_image_cache(names, output_path=image_cache_path)
    else:
        print(f"Reusing existing image cache: {image_cache_path}")

    train_cmd = [
        sys.executable, "-u", str(_THIS_DIR / "train.py"),
        "--image_cache", str(image_cache_path),
        "--output_dir", str(_THIS_DIR / "checkpoint"),
        "--epochs", str(args.epochs),
        "--batch_size", str(args.batch_size),
    ]
    if args.gpu_index is not None:
        train_cmd += ["--gpu_index", str(args.gpu_index)]
    if args.smoke_test:
        train_cmd += ["--smoke_test"]

    predict_cmd = [
        sys.executable, "-u", str(_THIS_DIR / "predict.py"),
        "--checkpoint", str(_THIS_DIR / "checkpoint"),
        "--image_cache", str(image_cache_path),
        "--output", str(_THIS_DIR / "val_predictions.jsonl"),
    ]
    if args.gpu_index is not None:
        predict_cmd += ["--gpu_index", str(args.gpu_index)]
    if args.smoke_test:
        predict_cmd += ["--smoke_test"]

    import subprocess
    print("=== Training ===")
    subprocess.run(train_cmd, check=True)
    print("=== Predicting val ===")
    subprocess.run(predict_cmd, check=True)

    print("=== Scoring val ===")
    gold = data_io.load_jsonl(data_io.VAL_FILE)
    preds = data_io.load_jsonl(_THIS_DIR / "val_predictions.jsonl")
    if args.smoke_test:
        pred_ids = {p["id"] for p in preds}
        gold = [g for g in gold if g["id"] in pred_ids]
    scores = scorer_local.score_predictions(gold, preds, verbose=True)
    with open(_THIS_DIR / "val_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved -> {_THIS_DIR / 'val_scores.json'}")


if __name__ == "__main__":
    main()
