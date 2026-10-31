"""Predict with a trained adapter on val or test, greedy or sampled.

Greedy (default) writes {"id", "labels"} per record -- the submission format.
Sampled (--num_samples N, Part 3a) writes {"id", "labels": [], "samples": [span lists]}.

Always predicts on EVERY record of the chosen file (val: all 3,020, scored per half
later; test: all 4,898). Output is written incrementally and --resume skips ids
already written, so an nlp2 crash costs minutes, not the whole run.

Usage:
    python -m thesis_v2.part1_models.predict --backbone qwen3vl8b \
        --adapter thesis_v2/runs/part1/qwen3vl8b_seed42 --split val --gpu_index 1
    python -m thesis_v2.part1_models.predict ... --split test
    python -m thesis_v2.part1_models.predict ... --split val --num_samples 8 --temperature 0.7
"""
from __future__ import annotations

import os

# must be set before torch initializes CUDA: reduces fragmentation OOMs on the 24 GB cards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
from pathlib import Path

import torch

from thesis_v2.common.paths import REPO_ROOT
from thesis_v2.common import backbones, vlm_runtime as vr
from solutions.common import data_io, vlm_io


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(backbones.BACKBONES))
    p.add_argument("--adapter", required=True)
    p.add_argument("--split", choices=["val", "test"], required=True)
    p.add_argument("--output", default=None)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--num_samples", type=int, default=1)
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.95)
    p.add_argument("--sample_seed", type=int, default=0)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--limit", type=int, default=None, help="Only the first N records (smoke test).")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.set_defaults(load_in_4bit=True)
    args = p.parse_args()

    bb = backbones.get(args.backbone)
    adapter = Path(args.adapter) if Path(args.adapter).is_absolute() else REPO_ROOT / args.adapter
    sampled = args.num_samples > 1
    default_name = (f"{args.split}_samples{args.num_samples}_T{args.temperature}.jsonl" if sampled
                    else f"{args.split}_predictions.jsonl")
    out_path = Path(args.output) if args.output else adapter / default_name

    src = data_io.VAL_FILE if args.split == "val" else data_io.TEST_FILE
    records = data_io.load_jsonl(src)
    if args.limit:
        records = records[:args.limit]

    done = set()
    if args.resume and out_path.exists():
        done = {json.loads(l)["id"] for l in out_path.read_text(encoding="utf-8").splitlines() if l.strip()}
        print(f"--resume: {len(done)} already written")
    elif out_path.exists():
        raise SystemExit(f"{out_path} exists -- pass --resume or delete it deliberately")

    device = vr.resolve_device(args.gpu_index)
    processor = vr.load_processor(bb, source=str(adapter))
    model, dtype = vr.load_model(bb, device, args.load_in_4bit)
    from peft import PeftModel
    model = PeftModel.from_pretrained(model, str(adapter))
    model.eval()
    pad_id = vr.pad_token_id(processor)
    torch.manual_seed(args.sample_seed)

    from tqdm.auto import tqdm
    n_unresolvable = 0
    with open(out_path, "a", encoding="utf-8") as fout, torch.no_grad():
        for rec in tqdm(records, desc=f"[{bb.name}] {args.split}{' sampled' if sampled else ''}", dynamic_ncols=True):
            if rec["id"] in done:
                continue
            conv = vlm_io.build_conversation(rec, include_target=False)
            if conv is None:
                # Never silently drop a record: an empty prediction is written and counted,
                # so coverage stays complete and the count is visible in the log.
                n_unresolvable += 1
                row = {"id": rec["id"], "labels": []} | ({"samples": []} if sampled else {})
                fout.write(json.dumps(row, ensure_ascii=False) + "\n")
                continue
            inputs = vr.build_inputs(processor, bb, conv["messages"], conv["_image_path"],
                                     add_generation_prompt=True).to(device)
            # the vision tower stays in `dtype` here (training's kbit preparation makes it
            # float32); Pixtral, unlike the others, does not cast pixel_values itself
            inputs["pixel_values"] = inputs["pixel_values"].to(dtype)
            gen = dict(max_new_tokens=args.max_new_tokens, pad_token_id=pad_id)
            if sampled:
                gen |= dict(do_sample=True, temperature=args.temperature, top_p=args.top_p,
                            num_return_sequences=args.num_samples)
            else:
                gen |= dict(do_sample=False)
            out = model.generate(**inputs, **gen)
            texts = processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
            span_lists = [vlm_io.predict_spans_from_raw_output(rec["response"], t.strip()) for t in texts]
            row = ({"id": rec["id"], "labels": [], "samples": span_lists} if sampled
                   else {"id": rec["id"], "labels": span_lists[0]})
            fout.write(json.dumps(row, ensure_ascii=False) + "\n")
            fout.flush()
    print(f"wrote -> {out_path}  (unresolvable images: {n_unresolvable})")


if __name__ == "__main__":
    main()
