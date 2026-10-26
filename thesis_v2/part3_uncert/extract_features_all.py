"""Part 3.0: extract judge features for the CLEAN train/val records (fixes the Aim 3 leak).

solutions/07_uncertainty_quantification/extract_features.py (lines 561-562) extracted
train/val features only for records WITH labels. Records without features defaulted to
empty predictions, and the official scorer gives empty gold + empty prediction a
perfect 1.0 -- so the 859 clean val records (28%) scored perfectly for free in every
Aim 3/4 val number.

This script reuses the old extractor's per-record function unchanged (same judge
Qwen3-VL-8B, same features, same .npz format) and runs it on exactly the records the
old run skipped: clean train (3,436) and clean val (859). The existing labeled shards
stay valid; features.load_split() merges both and asserts the id set equals the
full split.

Usage (nlp2):
    python -m thesis_v2.part3_uncert.extract_features_all --split val --gpu_index 0 --dry_run
    python -m thesis_v2.part3_uncert.extract_features_all --split val --gpu_index 0
    python -m thesis_v2.part3_uncert.extract_features_all --split train --shard_id 0 --num_shards 2 --gpu_index 0
"""
from __future__ import annotations

import argparse
import importlib.util

import numpy as np
import torch

from thesis_v2.common.paths import OLD_SOLUTIONS, RUNS_DIR
from solutions.common import data_io

CLEAN_FEATURES_DIR = RUNS_DIR / "part3" / "features_clean"


def load_old_extractor():
    """The old module lives in a folder whose name starts with a digit (not importable
    as a package), so it is loaded from its path -- read-only, never edited."""
    path = OLD_SOLUTIONS / "07_uncertainty_quantification" / "extract_features.py"
    spec = importlib.util.spec_from_file_location("old_extract_features", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main():
    old = load_old_extractor()
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--model_id", default=old.DEFAULT_MODEL_ID)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--hidden_layers", default="-1")   # same as the old run
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--save_every", type=int, default=200)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)
    model = old.resolve_model_class(args.model_id).from_pretrained(
        args.model_id, dtype=old.best_dtype(device), device_map={"": device} if device != "cpu" else None,
        attn_implementation="eager")
    model.eval()
    image_token_id = old.find_image_token_id(model, processor)

    path = data_io.TRAIN_FILE if args.split == "train" else data_io.VAL_FILE
    records = [r for r in data_io.load_jsonl(path) if not r.get("labels")]   # exactly what the old run skipped
    records = records[args.shard_id::args.num_shards]
    if args.dry_run:
        records = records[:3]
    print(f"{len(records)} clean {args.split} records (shard {args.shard_id}/{args.num_shards})")

    CLEAN_FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    out_path = CLEAN_FEATURES_DIR / f"{args.split}_shard{args.shard_id}of{args.num_shards}.npz"
    keys = ["entropy", "nll", "label_id", "offset_start", "offset_end"]
    acc = {k: [] for k in keys} | {"language": [], "record_id": [], "approx_alignment": [], "lookback": []}
    hidden = {i: [] for i in hidden_layers}
    lookback_cols = [None]
    skipped = []

    def save():
        kw = {k: np.concatenate(acc[k]) if acc[k] else np.zeros(0) for k in keys}
        kw |= {"language": np.array(acc["language"], dtype=object),
               "record_id": np.array(acc["record_id"], dtype=object),
               "approx_alignment": np.array(acc["approx_alignment"], dtype=bool),
               "lookback": np.concatenate(acc["lookback"], axis=0) if acc["lookback"] else np.zeros((0, 0)),
               "lookback_columns": np.array(lookback_cols[0] or [], dtype=object),
               # records the judge could not process (missing image, too long, OOM): the
               # loader still requires them to be covered, by an explicit empty prediction
               "skipped_ids": np.array(skipped, dtype=object)}
        for i in hidden_layers:
            kw[f"hidden_{i}"] = np.concatenate(hidden[i], axis=0) if hidden[i] else np.zeros((0, 0))
        np.savez_compressed(out_path, **kw)

    from tqdm.auto import tqdm
    n = 0
    for rec in tqdm(records, desc=f"clean[{args.split}]"):
        res = old.process_record(rec, model, processor, device, image_token_id, hidden_layers,
                                 max_seq_len=args.max_seq_len or None, verbose=args.dry_run)
        if res is None:
            skipped.append(rec["id"])
            continue
        if args.dry_run:
            print(f"  id={res['record_id']} n_tokens={res['n_tokens']} positives="
                  f"{int((res['label_id'] != old.CLEAN_LABEL_ID).sum())} (must be 0)")
            continue
        lookback_cols[0] = lookback_cols[0] or res["lookback_columns"]
        for k in keys:
            acc[k].append(res[k])
        acc["language"].extend([res["language"]] * res["n_tokens"])
        acc["record_id"].extend([res["record_id"]] * res["n_tokens"])
        acc["approx_alignment"].extend([res["approx_alignment"]] * res["n_tokens"])
        acc["lookback"].append(res["lookback"])
        for i in hidden_layers:
            hidden[i].append(res["hidden"][str(i)])
        n += 1
        if args.save_every and n % args.save_every == 0:
            save()
    if not args.dry_run:
        save()
        print(f"saved {n} records ({len(skipped)} skipped, recorded in the file) -> {out_path}")


if __name__ == "__main__":
    main()
