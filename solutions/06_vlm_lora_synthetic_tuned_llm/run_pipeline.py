"""End-to-end pipeline for solution 06 (VLM LoRA fine-tune, synthetic
augmented -- tuned upsampling, WITH LLM-assisted injection): generate
synthetic data -> train -> predict val -> score val -> write results.

This is solution 05 plus method (c) (LLM-assisted injection) included by
default -- a genuinely separate solution folder, not a config toggle within
solution 05, specifically so the two can run in two parallel Colab sessions
without ever touching the same Drive paths. See solution 05's
generate_synthetic.py docstring for the diagnosis behind the tuned
upsampling multiplier both solutions share.

Reuses solution 05's generate_synthetic.py directly (subprocess, unchanged
-- same --max_upsample_multiplier fix), and solution 02's train_lora.py /
predict.py, same as every other solution in this repo. Only the checkpoint/
output paths and the omission of --skip_llm differ from solution 05's own
run_pipeline.py.

Usage:
    python run_pipeline.py --gpu_index 0
    python run_pipeline.py --smoke_test --gpu_index 0
    ANTHROPIC_API_KEY=... python run_pipeline.py --gpu_index 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
_SOLUTION_02_DIR = _REPO_ROOT / "solutions" / "02_vlm_lora_real_data"
_SOLUTION_05_DIR = _REPO_ROOT / "solutions" / "05_vlm_lora_synthetic_tuned"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, scorer_local


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_id", default=None,
                    help="Defaults to train_lora.py's DEFAULT_MODEL_ID (Qwen3-VL-8B-Instruct, dense) "
                         "if omitted. Pass e.g. Qwen/Qwen3-VL-30B-A3B-Instruct for the bigger MoE "
                         "alternative (needs a 24GB 3090 Ti specifically, see train_lora.py's docstring).")
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--no_4bit", action="store_true")
    p.add_argument("--max_upsample_multiplier", type=int, default=None,
                    help="Pass-through to solution 05's generate_synthetic.py --max_upsample_"
                         "multiplier (default 3 there). Keep this the same as whatever solution 05 "
                         "actually used, so the only real difference between the two solutions is "
                         "method (c), not also a different multiplier.")
    p.add_argument("--lora_r", type=int, default=None)
    p.add_argument("--target_modules", default=None)
    p.add_argument("--max_seq_len", type=int, default=None)
    p.add_argument("--max_pixels", type=int, default=None)
    p.add_argument("--min_pixels", type=int, default=None)
    p.add_argument("--checkpoint_dir", default=None,
                    help="Where the LoRA adapter is saved (train) and loaded from (predict). "
                         "Defaults to a local ./checkpoint next to this script. Point this at a "
                         "Drive-mounted path on Colab so periodic --save_steps checkpoints survive "
                         "a session being killed.")
    p.add_argument("--save_steps", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip_ocr_extract", action="store_true", help="Reuse an existing ocr_cache.json.")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else str(_THIS_DIR / "checkpoint")
    train_augmented_path = _THIS_DIR / "train_augmented.jsonl"

    gen_cmd = [
        sys.executable, "-u", str(_SOLUTION_05_DIR / "generate_synthetic.py"),
        "--output", str(train_augmented_path),
        "--ocr_cache", str(_THIS_DIR / "ocr_cache.json"),
        "--llm_cache", str(_THIS_DIR / "llm_synth_cache.json"),
    ]
    if args.max_upsample_multiplier is not None:
        gen_cmd += ["--max_upsample_multiplier", str(args.max_upsample_multiplier)]
    # Deliberately never pass --skip_llm here -- this solution's whole point is including method
    # (c). It still no-ops cleanly (zero API calls) if ANTHROPIC_API_KEY isn't set, same as always.
    if args.skip_ocr_extract:
        gen_cmd += ["--skip_ocr_extract"]
    if args.smoke_test:
        gen_cmd += ["--smoke_test"]

    print("=== Generating synthetic training data (a [tuned] + b + c) ===")
    subprocess.run(gen_cmd, check=True)

    train_cmd = [
        sys.executable, "-u", str(_SOLUTION_02_DIR / "train_lora.py"),
        "--train_file", str(train_augmented_path),
        "--output_dir", checkpoint_dir,
        "--epochs", str(args.epochs),
        "--grad_accum", str(args.grad_accum),
    ]
    predict_cmd = [
        sys.executable, "-u", str(_SOLUTION_02_DIR / "predict.py"),
        "--checkpoint", checkpoint_dir,
        "--output", str(_THIS_DIR / "val_predictions.jsonl"),
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

    print("=== Training (LoRA, synthetic augmented, tuned + LLM) ===")
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
