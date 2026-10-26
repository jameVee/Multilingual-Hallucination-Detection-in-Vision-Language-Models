"""QLoRA fine-tuning of any registered backbone on one or more training files.

Generalized from solutions/02_vlm_lora_real_data/train_lora.py (same loop, recipe,
periodic checkpointing and --resume). Used unchanged by Part 2: a Part 2 condition
is just a different --train_files list on the Part 1 winner backbone.

Fixed recipe (never tuned on validation): 4-bit NF4 QLoRA, r16/alpha16, lr 2e-4,
linear warmup 50 then linear decay, 2 epochs, grad_accum 8, batch 1, max_seq_len 4096.

Usage:
    python -m thesis_v2.part1_models.train --backbone qwen3vl8b --seed 42 --gpu_index 1
    python -m thesis_v2.part1_models.train --backbone gemma3_12b --max_steps 20 --gpu_index 1   # feasibility
    python -m thesis_v2.part1_models.train --backbone qwen3vl8b --seed 42 \
        --train_files data/train.jsonl thesis_v2/part2_synth/data/C2_heuristic.jsonl \
        --output_dir thesis_v2/runs/part2/C2_seed42
"""
from __future__ import annotations

import os

# must be set before torch initializes CUDA: reduces fragmentation OOMs on the 24 GB cards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import hashlib
import json
import random
import time
from pathlib import Path

import torch

from thesis_v2.common.paths import REPO_ROOT, RUNS_DIR
from thesis_v2.common import backbones, vlm_runtime as vr
from solutions.common import data_io, vlm_io


def file_sha1(path):
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(backbones.BACKBONES))
    p.add_argument("--train_files", nargs="+", default=[str(data_io.TRAIN_FILE)])
    p.add_argument("--output_dir", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=2)
    p.add_argument("--grad_accum", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora_r", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--warmup_steps", type=int, default=50)
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--save_steps", type=int, default=250)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max_steps", type=int, default=None,
                   help="Stop after this many optimizer steps (feasibility gate / smoke test).")
    p.add_argument("--no_4bit", dest="load_in_4bit", action="store_false")
    p.set_defaults(load_in_4bit=True)
    args = p.parse_args()

    bb = backbones.get(args.backbone)
    out = Path(args.output_dir) if args.output_dir else RUNS_DIR / "part1" / f"{bb.name}_seed{args.seed}"
    if not args.resume and (out / "adapter_config.json").exists():
        raise SystemExit(f"{out} already holds an adapter -- pass --resume to continue it, "
                         f"or choose another --output_dir (never overwrite a finished run)")
    out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(args.seed)
    device = vr.resolve_device(args.gpu_index)

    records = []
    provenance = []
    for f in args.train_files:
        path = Path(f) if Path(f).is_absolute() else REPO_ROOT / f
        recs = data_io.load_jsonl(path)
        records.extend(recs)
        provenance.append({"file": str(path.relative_to(REPO_ROOT)), "n": len(recs), "sha1": file_sha1(path)})
    examples = [c for c in (vlm_io.build_conversation(r, include_target=True) for r in records) if c is not None]
    print(f"[{bb.name}] {len(records)} records from {len(args.train_files)} file(s) -> {len(examples)} usable")
    random.Random(args.seed).shuffle(examples)

    state_path = out / "train_state.json"
    done, gstep0 = 0, 0
    if args.resume and state_path.exists():
        st = json.loads(state_path.read_text())
        if st.get("complete"):
            print("Run already complete -- nothing to do."); return
        done, gstep0 = st["examples_done"], st["global_step"]
        print(f"--resume: skipping {done} examples, global_step={gstep0}")

    processor = vr.load_processor(bb)
    model, dtype = vr.load_model(bb, device, args.load_in_4bit)
    if args.resume and (out / "adapter_config.json").exists():
        from peft import PeftModel, prepare_model_for_kbit_training
        if args.load_in_4bit:
            model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
        model = PeftModel.from_pretrained(model, str(out), is_trainable=True)
        n_lora = sum(1 for n, _ in model.named_modules() if n.endswith("lora_A"))
    else:
        model, n_lora = vr.apply_lora(model, bb, args.lora_r, args.lora_alpha, True, args.load_in_4bit)
    print(f"[{bb.name}] LoRA on {n_lora} LM projection modules; dtype={dtype}")
    model.print_trainable_parameters()

    optimizer = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(examples) // args.grad_accum) * args.epochs)

    def lr_lambda(step):
        if step < args.warmup_steps:
            return step / max(1, args.warmup_steps)
        return max(0.0, 1.0 - (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    for _ in range(gstep0):
        scheduler.step()

    def save(cum, complete=False):
        model.save_pretrained(out)
        processor.save_pretrained(out)
        (out / "train_config.json").write_text(json.dumps(
            vars(args) | {"backbone_model_id": bb.model_id, "provenance": provenance}, indent=2))
        state_path.write_text(json.dumps({"examples_done": cum, "global_step": gstep, "complete": complete}))

    from tqdm.auto import tqdm
    model.train()
    gstep, skipped, seq_lens = gstep0, 0, []
    t0 = time.time()
    stop = False
    cum = done
    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pbar = tqdm(range(len(examples)), desc=f"[{bb.name}] epoch {epoch}", unit="ex", dynamic_ncols=True)
        for step in pbar:
            cum = epoch * len(examples) + step
            if cum < done:
                continue
            batch, _ = vr.build_training_batch(processor, bb, examples[step])
            seq_len = batch["input_ids"].shape[1]
            if seq_len > args.max_seq_len:
                skipped += 1
                continue
            seq_lens.append(seq_len)
            batch = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in batch.items()}
            loss = model(**batch).loss / args.grad_accum
            loss.backward()
            if (step + 1) % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step(); scheduler.step(); optimizer.zero_grad()
                gstep += 1
                if args.save_steps and gstep % args.save_steps == 0:
                    save(cum + 1)
                if args.max_steps and gstep >= args.max_steps:
                    stop = True
                    break
            pbar.set_postfix(loss=f"{loss.item() * args.grad_accum:.4f}", step=f"{gstep}/{total_steps}")
        if stop:
            break

    elapsed = time.time() - t0
    stats = {
        "backbone": bb.name, "seconds": round(elapsed, 1), "examples_seen": len(seq_lens),
        "sec_per_example": round(elapsed / max(1, len(seq_lens)), 3), "skipped_over_max_seq_len": skipped,
        "seq_len_mean": round(sum(seq_lens) / max(1, len(seq_lens)), 1), "seq_len_max": max(seq_lens or [0]),
        "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device != "cpu" else None,
        "optimizer_steps": gstep, "total_steps_planned": total_steps,
    }
    (out / "train_stats.json").write_text(json.dumps(stats, indent=2))
    save(len(examples) * args.epochs if not stop else cum + 1, complete=not stop)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
