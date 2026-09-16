"""Seed-variant train -> predict -> score runner, for the multi-seed reruns
in thesis Aim 1 (separating training-seed variance from the example-level
variance the paper's paired bootstrap already captures).

Unlike every run_pipeline.py in this repo, this script:
  (a) actually exposes --seed through to train_lora.py -- none of the
      per-solution run_pipeline.py scripts do (checked 02/03/04/05/06), so
      rerunning any of them retrains at train_lora.py's hardcoded default
      of 42 every time, regardless of how many times you rerun them;
  (b) takes an already-existing --train_file directly and never generates
      synthetic data itself, so repeated calls with different --seed
      values retrain on byte-identical data -- isolating pure
      training-seed variance (LoRA init + example-order shuffle) from a
      second, currently out-of-scope kind of variance (synthetic-
      generation randomness).

Writes to '<checkpoint_dir>_seed<seed>/' -- adapter files land directly in
that folder (same layout train_lora.py's own --output_dir always produces),
alongside val_predictions.jsonl and val_scores.json -- so repeated seeds
never collide with each other or with the solution's own ./checkpoint from
its original default-seed run.

Reuses solution 02's train_lora.py / predict.py unchanged via subprocess,
same as every run_pipeline.py in this repo -- only the seed plumbing and
the skipped data-generation step are new.

Usage:
    # baseline (solution 02), an extra seed
    python run_seed_variant.py --train_file data/train.jsonl \\
      --checkpoint_dir solutions/02_vlm_lora_real_data/checkpoint \\
      --seed 123 --gpu_index 1

    # tuned (solution 05), reusing its existing train_augmented.jsonl unchanged
    python run_seed_variant.py \\
      --train_file solutions/05_vlm_lora_synthetic_tuned/train_augmented.jsonl \\
      --checkpoint_dir solutions/05_vlm_lora_synthetic_tuned/checkpoint \\
      --seed 123 --gpu_index 2
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent  # solutions/common
_REPO_ROOT = _THIS_DIR.parent.parent
_SOLUTION_02_DIR = _REPO_ROOT / "solutions" / "02_vlm_lora_real_data"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, scorer_local


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", required=True,
                    help="An already-built training file -- data/train.jsonl for the real-data "
                         "baseline, or an existing <solution>/train_augmented.jsonl to reuse a "
                         "prior run's synthetic data unchanged. This script never generates "
                         "synthetic data itself.")
    p.add_argument("--checkpoint_dir", required=True,
                    help="Base path; the actual run writes to '<checkpoint_dir>_seed<seed>/' so "
                         "repeated seeds never collide with each other or with the original "
                         "solution's own ./checkpoint.")
    p.add_argument("--seed", type=int, required=True,
                    help="Passed through to train_lora.py's --seed (LoRA init + example-order "
                         "shuffle) -- this is the whole point of this script, see module docstring.")
    p.add_argument("--model_id", default=None)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--target_modules", default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--resume", action="store_true",
                    help="Pass-through to train_lora.py's --resume. Safe to always pass -- a no-op "
                         "on a fresh/empty run directory.")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    run_dir = Path(f"{args.checkpoint_dir}_seed{args.seed}")
    run_dir.mkdir(parents=True, exist_ok=True)

    train_cmd = [
        sys.executable, "-u", str(_SOLUTION_02_DIR / "train_lora.py"),
        "--train_file", args.train_file,
        "--output_dir", str(run_dir),
        "--epochs", str(args.epochs),
        "--grad_accum", str(args.grad_accum),
        "--seed", str(args.seed),
    ]
    predict_cmd = [
        sys.executable, "-u", str(_SOLUTION_02_DIR / "predict.py"),
        "--checkpoint", str(run_dir),
        "--output", str(run_dir / "val_predictions.jsonl"),
    ]
    if args.model_id is not None:
        train_cmd += ["--model_id", args.model_id]
        predict_cmd += ["--base_model_id", args.model_id]
    if args.lora_r is not None:
        train_cmd += ["--lora_r", str(args.lora_r)]
    if args.target_modules is not None:
        train_cmd += ["--target_modules", args.target_modules]
    if args.max_seq_len is not None:
        train_cmd += ["--max_seq_len", str(args.max_seq_len)]
    if args.save_steps is not None:
        train_cmd += ["--save_steps", str(args.save_steps)]
    if args.resume:
        train_cmd += ["--resume"]
    for cmd in (train_cmd, predict_cmd):
        if args.gpu_index is not None:
            cmd += ["--gpu_index", str(args.gpu_index)]
        if args.no_4bit:
            cmd += ["--no_4bit"]
        if args.max_pixels is not None:
            cmd += ["--max_pixels", str(args.max_pixels)]
        if args.min_pixels is not None:
            cmd += ["--min_pixels", str(args.min_pixels)]
        if args.smoke_test:
            cmd += ["--smoke_test"]

    print(f"=== Training (seed={args.seed}, train_file={args.train_file}) -> {run_dir} ===")
    subprocess.run(train_cmd, check=True)
    print("=== Predicting val ===")
    subprocess.run(predict_cmd, check=True)

    print("=== Scoring val ===")
    gold = data_io.load_jsonl(data_io.VAL_FILE)
    preds = data_io.load_jsonl(run_dir / "val_predictions.jsonl")
    if args.smoke_test:
        pred_ids = {p["id"] for p in preds}
        gold = [g for g in gold if g["id"] in pred_ids]
    scores = scorer_local.score_predictions(gold, preds, verbose=True)
    # Provenance fields so Phase 3's aggregation can tell runs apart without
    # relying on folder naming alone -- score_predictions never returns keys
    # starting with "_", so these can't collide with its own output.
    scores["_seed"] = args.seed
    scores["_train_file"] = args.train_file
    with open(run_dir / "val_scores.json", "w") as f:
        json.dump(scores, f, indent=2)
    print(f"Saved -> {run_dir / 'val_scores.json'}")


if __name__ == "__main__":
    main()
