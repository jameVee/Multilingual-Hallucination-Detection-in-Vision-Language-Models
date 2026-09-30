"""Aim 4 prerequisite: add per-token character offsets to existing Aim 3
shard files, WITHOUT re-running the frozen-model forward pass.

extract_features.py's shards never saved each token's (start, end)
character position within its response -- they only saved derived features
(entropy, hidden states, lookback ratios). Aim 4's span-assembly step needs
those offsets to turn per-token positive/negative predictions back into
char-offset spans the official scorer can score. Re-running the full
extraction (a ~2-hour GPU job) just to add two integers per token would be
wasteful -- offsets only depend on tokenization, not on the model's
weights or a forward pass, so this script loads ONLY the processor
(tokenizer + image preprocessing, no model, no GPU/CUDA at all) and
replays exactly the same tokenization steps extract_features.py used, then
adds the results into each shard's own .npz file (new keys: `offset_start`,
`offset_end`; everything else preserved as-is).

Safety: recomputes each record's token ordering from data/{split}.jsonl
using the IDENTICAL filtering and shard-striping extract_features.py used
(`records = [r for r in records if r.get("labels")]`, then
`records[shard_id::num_shards]`), then asserts the resulting per-record
token-count run-lengths EXACTLY match what's already stored in the shard's
own `record_id` array before writing anything -- if they don't line up
(e.g. because --max_seq_len or a caught OOM skipped a different set of
records than expected), this errors out loudly instead of silently writing
misaligned offsets.

Usage:
    python compute_offsets.py --split val --shard_id 0 --num_shards 1
    python compute_offsets.py --split train --shard_id 0 --num_shards 3
    python compute_offsets.py --split train --shard_id 1 --num_shards 3
    python compute_offsets.py --split train --shard_id 2 --num_shards 3
"""
from __future__ import annotations

import argparse
import sys
from itertools import groupby
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io
from extract_features import DEFAULT_MODEL_ID, build_plain_conversation, token_char_spans_from_ids


def compute_record_offsets(record, processor, max_pixels, min_pixels):
    """Mirrors process_record's tokenization steps exactly (same functions,
    same call shapes), minus everything that needs the actual model:
    no model(**batch), no hooks, no torch/CUDA. Returns (offsets,
    approx_alignment, n_real_tokens) or None if the record can't be built
    at all (unresolvable image -- should be rare/never, since the original
    extraction already succeeded on this exact record set)."""
    conv = build_plain_conversation(record)
    if conv is None:
        return None
    messages = conv["messages"]
    user_only = [messages[0]]
    image = Image.open(conv["_image_path"]).convert("RGB")

    full_text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
    prompt_text = processor.apply_chat_template(user_only, tokenize=False, add_generation_prompt=True)

    full_inputs = processor(text=[full_text], images=[image], padding=True, return_tensors="pt")
    prompt_inputs = processor(text=[prompt_text], images=[image], padding=True, return_tensors="pt")
    prompt_len = prompt_inputs["input_ids"].shape[1]
    seq_len = full_inputs["input_ids"].shape[1]
    if seq_len <= prompt_len:
        return None

    input_ids_cpu = full_inputs["input_ids"][0]
    response_ids_superset = input_ids_cpu[prompt_len:].tolist()

    offsets, mismatch_text, n_real_tokens = token_char_spans_from_ids(
        processor.tokenizer, response_ids_superset, record["response"]
    )
    approx_alignment = offsets is None
    if approx_alignment:
        n_real_tokens = len(response_ids_superset)
        resp_char_len = len(record["response"])
        step = resp_char_len / max(1, n_real_tokens)
        offsets = [(int(i * step), int((i + 1) * step)) for i in range(n_real_tokens)]

    return offsets, approx_alignment, n_real_tokens


def run_length_groups(record_ids):
    """[a,a,a,b,b,c,...] -> [('a',3), ('b',2), ('c',1), ...] -- same shape
    as how extract_features.py's main() wrote record_id (one contiguous
    run per successfully-processed record, in processing order)."""
    return [(key, sum(1 for _ in group)) for key, group in groupby(record_ids)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=["train", "val"], required=True)
    p.add_argument("--shard_id", type=int, default=0)
    p.add_argument("--num_shards", type=int, default=1)
    p.add_argument("--model_id", default=DEFAULT_MODEL_ID)
    p.add_argument("--max_pixels", type=int, default=1024 * 28 * 28)
    p.add_argument("--min_pixels", type=int, default=256 * 28 * 28)
    p.add_argument("--features_dir", default=str(_THIS_DIR / "features"))
    args = p.parse_args()

    shard_path = Path(args.features_dir) / f"{args.split}_shard{args.shard_id}of{args.num_shards}.npz"
    if not shard_path.exists():
        raise FileNotFoundError(f"{shard_path} not found -- run extract_features.py for this shard first.")

    existing = dict(np.load(shard_path, allow_pickle=True))
    if "offset_start" in existing:
        print(f"{shard_path} already has offsets -- nothing to do.")
        return

    stored_record_ids = existing["record_id"]
    stored_groups = run_length_groups(stored_record_ids.tolist())
    print(f"Shard has {len(stored_groups)} records, {len(stored_record_ids)} tokens total.")

    print(f"Loading processor for {args.model_id} (tokenizer + image preprocessing only, no model weights)...")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(args.model_id, min_pixels=args.min_pixels, max_pixels=args.max_pixels)

    path = data_io.TRAIN_FILE if args.split == "train" else data_io.VAL_FILE
    records = data_io.load_jsonl(path)
    records = [r for r in records if r.get("labels")]
    records = records[args.shard_id::args.num_shards]
    by_id = {r["id"]: r for r in records}

    from tqdm.auto import tqdm
    all_starts, all_ends = [], []
    for record_id, expected_count in tqdm(stored_groups, desc=f"offsets[{args.split}]", unit="rec"):
        record = by_id.get(record_id)
        if record is None:
            raise RuntimeError(
                f"Record {record_id!r} is in the shard but not found in {path} under this "
                f"--split/--shard_id/--num_shards -- re-check those match the original extraction run."
            )
        result = compute_record_offsets(record, processor, args.max_pixels, args.min_pixels)
        if result is None:
            raise RuntimeError(
                f"Record {record_id!r} failed to tokenize here but succeeded in the original shard -- "
                f"something about the environment differs from the original extraction run."
            )
        offsets, approx_alignment, n_real_tokens = result
        if n_real_tokens != expected_count:
            raise RuntimeError(
                f"Token-count mismatch for {record_id!r}: shard has {expected_count} tokens, "
                f"recomputed {n_real_tokens}. Refusing to write misaligned offsets -- investigate "
                f"before rerunning (e.g. a --max_pixels/--min_pixels mismatch vs. the original run)."
            )
        for start, end in offsets:
            all_starts.append(start)
            all_ends.append(end)

    assert len(all_starts) == len(stored_record_ids), (
        f"Total offset count {len(all_starts)} != shard token count {len(stored_record_ids)}"
    )
    existing["offset_start"] = np.array(all_starts, dtype=np.int32)
    existing["offset_end"] = np.array(all_ends, dtype=np.int32)
    np.savez_compressed(shard_path, **existing)
    print(f"Added offsets for {len(all_starts)} tokens -> {shard_path}")


if __name__ == "__main__":
    main()
