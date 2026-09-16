"""Training loop for the lightweight CLIP+XLM-R span tagger (solution 01).

Usage:
    python train.py --epochs 3 --batch_size 8
    python train.py --smoke_test               # ~30 examples, 2 steps, verifies the pipeline

Requires an image cache built first via:
    python -m solutions.common.image_cache --output ./image_cache.pt
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from solutions.common import data_io
from solutions.common.image_cache import load_image_cache
from model import SpanTagger
from dataset import SpanTaggingDataset, collate_fn


def compute_pos_weights(records, cap=30.0):
    counts = data_io.class_counts(records)["span_counts"]
    total = sum(counts.get(l, 0) for l in data_io.LABEL_ORDER) or 1
    weights = []
    for label in data_io.LABEL_ORDER:
        n = counts.get(label, 0)
        w = (total / max(n, 1)) / len(data_io.LABEL_ORDER)
        weights.append(min(cap, max(1.0, w)))
    return torch.tensor(weights, dtype=torch.float32)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train_file", default=str(data_io.TRAIN_FILE))
    p.add_argument("--image_cache", default=str(_THIS_DIR / "image_cache.pt"))
    p.add_argument("--output_dir", default=str(_THIS_DIR / "checkpoint"))
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--grad_accum", type=int, default=2)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--max_length", type=int, default=512)
    p.add_argument("--warmup_ratio", type=float, default=0.06)
    p.add_argument("--gpu_index", type=int, default=None,
                    help="Select a specific CUDA device index (avoid broken indices on shared machines).")
    p.add_argument("--smoke_test", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)

    records = data_io.load_jsonl(args.train_file)
    if args.smoke_test:
        records = records[:30]
        args.epochs = 1
    print(f"Loaded {len(records)} training records from {args.train_file} (device={device})")

    image_cache = load_image_cache(args.image_cache)
    print(f"Loaded image cache: {len(image_cache)} images")

    from transformers import XLMRobertaTokenizerFast
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    dataset = SpanTaggingDataset(records, tokenizer, image_cache, max_length=args.max_length)
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
    )

    model = SpanTagger().to(device)
    pos_weight = compute_pos_weights(records).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    total_steps = max(1, (len(loader) // args.grad_accum) * args.epochs)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return max(0.0, 1.0 - progress)

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    use_amp = device.startswith("cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    model.train()
    global_step = 0
    start_time = time.time()
    max_optim_steps = 2 if args.smoke_test else None
    stop = False

    from tqdm.auto import tqdm

    for epoch in range(args.epochs):
        optimizer.zero_grad()
        pbar = tqdm(loader, desc=f"epoch {epoch}/{args.epochs - 1}", unit="batch", dynamic_ncols=True)
        for step, batch in enumerate(pbar):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            targets = batch["targets"].to(device)
            resp_mask = batch["resp_token_mask"].to(device)
            image_tokens = batch["image_tokens"].to(device)

            with torch.autocast(device_type="cuda" if use_amp else "cpu",
                                 enabled=use_amp, dtype=torch.float16):
                logits = model(input_ids, attention_mask, image_tokens)
                loss_raw = F.binary_cross_entropy_with_logits(
                    logits, targets, pos_weight=pos_weight, reduction="none"
                )
                mask = resp_mask.unsqueeze(-1).float()
                loss = (loss_raw * mask).sum() / mask.sum().clamp(min=1.0)
                loss = loss / args.grad_accum

            scaler.scale(loss).backward()

            if (step + 1) % args.grad_accum == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()
                global_step += 1

            elapsed = time.time() - start_time
            pbar.set_postfix({
                "loss": f"{loss.item() * args.grad_accum:.4f}",
                "opt_step": f"{global_step}",
                "elapsed": f"{elapsed / 60:.1f}m",
            })

            if max_optim_steps is not None and global_step >= max_optim_steps:
                stop = True
                break
        if stop:
            break

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_dir / "model.pt")
    with open(output_dir / "config.json", "w") as f:
        json.dump({"max_length": args.max_length}, f)
    print(f"Saved checkpoint -> {output_dir}")


if __name__ == "__main__":
    main()
