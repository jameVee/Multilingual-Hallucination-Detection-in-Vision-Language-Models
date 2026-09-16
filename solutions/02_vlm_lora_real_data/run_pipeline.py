"""End-to-end pipeline for solution 02 (VLM LoRA fine-tune, real data only):
train -> predict val -> score val -> write results.

Usage:
    python run_pipeline.py --gpu_index 0
    python run_pipeline.py --smoke_test --gpu_index 0
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
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
    p.add_argument("--lora_r", type=int, default=None,
                    help="Pass-through to train_lora.py's --lora_r (default 16 there). Lower it "
                         "(e.g. 8) to shave a bit of optimizer/gradient memory on tight GPUs.")
    p.add_argument("--target_modules", default=None,
                    help="Pass-through to train_lora.py's --target_modules (comma-separated, or "
                         "'all-linear'). Default for a dense model like the 8B is "
                         "q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj -- the "
                         "gate/up/down (MLP) projections are the largest linear layers in the "
                         "model, so their LoRA adapters dominate backward-pass gradient memory. "
                         "Pass 'q_proj,k_proj,v_proj,o_proj' (attention-only) to cut trainable "
                         "params substantially on an 11GB card, at some cost to adaptation quality.")
    p.add_argument("--max_seq_len", type=int, default=None,
                    help="Pass-through to train_lora.py's --max_seq_len (default 4096 there). "
                         "Training examples longer than this are skipped rather than OOMing the "
                         "run -- lower it if you hit a single-step CUDA OOM well into an otherwise "
                         "stable run (a sign one pathologically long example slipped through).")
    p.add_argument("--max_pixels", type=int, default=None,
                    help="Pass-through to train_lora.py/predict.py's --max_pixels (default "
                         "1024*28*28 there). Lower this first on an 11GB card (e.g. 2080 Ti) if "
                         "you hit a CUDA OOM during the backward pass -- it caps image resolution, "
                         "which is the main lever on vision-token activation memory.")
    p.add_argument("--min_pixels", type=int, default=None,
                    help="Pass-through to train_lora.py/predict.py's --min_pixels (default 256*28*28).")
    p.add_argument("--checkpoint_dir", default=None,
                    help="Where the LoRA adapter is saved (train) and loaded from (predict). "
                         "Defaults to a local ./checkpoint next to this script. Point this at a "
                         "Drive-mounted path on Colab (e.g. via a %%cd'd notebook) so periodic "
                         "--save_steps checkpoints survive a session being killed -- local /content "
                         "disk is wiped when that happens, Drive isn't.")
    p.add_argument("--save_steps", type=int, default=None,
                    help="Pass-through to train_lora.py's --save_steps (default 250 there). Saves "
                         "the adapter + resume state every N optimizer steps during training, not "
                         "just once at the end -- combine with --checkpoint_dir pointed at Drive and "
                         "--resume so a killed Colab session loses at most one save interval's worth "
                         "of progress instead of the entire run.")
    p.add_argument("--resume", action="store_true",
                    help="Pass-through to train_lora.py's --resume: if --checkpoint_dir already has "
                         "a saved adapter + train_state.json, continue from there instead of "
                         "starting fresh. Safe to always pass -- it's a no-op on a fresh/empty "
                         "checkpoint dir.")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    checkpoint_dir = args.checkpoint_dir if args.checkpoint_dir is not None else str(_THIS_DIR / "checkpoint")
    train_cmd = [
        sys.executable, "-u", str(_THIS_DIR / "train_lora.py"),
        "--output_dir", checkpoint_dir,
        "--epochs", str(args.epochs),
        "--grad_accum", str(args.grad_accum),
    ]
    predict_cmd = [
        sys.executable, "-u", str(_THIS_DIR / "predict.py"),
        "--checkpoint", checkpoint_dir,
        "--output", str(_THIS_DIR / "val_predictions.jsonl"),
    ]
    if args.save_steps is not None:
        train_cmd += ["--save_steps", str(args.save_steps)]
    if args.resume:
        train_cmd += ["--resume"]
    if args.model_id is not None:
        train_cmd += ["--model_id", args.model_id]
        predict_cmd += ["--base_model_id", args.model_id]
    if args.lora_r is not None:
        train_cmd += ["--lora_r", str(args.lora_r)]
    if args.target_modules is not None:
        train_cmd += ["--target_modules", args.target_modules]
    if args.max_seq_len is not None:
        train_cmd += ["--max_seq_len", str(args.max_seq_len)]
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

    print("=== Training (LoRA, real data only) ===")
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
