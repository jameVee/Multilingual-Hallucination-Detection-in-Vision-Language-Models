"""Run a trained lightweight tagger checkpoint over data/val.jsonl and write
val_predictions.jsonl in the official submission schema.

Usage:
    python predict.py --checkpoint ./checkpoint --image_cache ./image_cache.pt \
        --output ./val_predictions.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from solutions.common import data_io
from solutions.common.image_cache import load_image_cache
from model import SpanTagger, decode_spans, DEFAULT_THRESHOLDS
from dataset import SpanTaggingDataset, collate_fn


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val_file", default=str(data_io.VAL_FILE))
    p.add_argument("--checkpoint", default=str(_THIS_DIR / "checkpoint"))
    p.add_argument("--image_cache", default=str(_THIS_DIR / "image_cache.pt"))
    p.add_argument("--output", default=str(_THIS_DIR / "val_predictions.jsonl"))
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--thresholds", type=str, default=None,
                    help="JSON file overriding per-label decision thresholds.")
    p.add_argument("--smoke_test", action="store_true")
    args = p.parse_args()

    device = "cpu"
    if torch.cuda.is_available():
        device = f"cuda:{args.gpu_index}" if args.gpu_index is not None else "cuda"
        if args.gpu_index is not None:
            torch.cuda.set_device(args.gpu_index)

    checkpoint_dir = Path(args.checkpoint)
    config = json.loads((checkpoint_dir / "config.json").read_text())

    records = data_io.load_jsonl(args.val_file)
    if args.smoke_test:
        records = records[:20]
    print(f"Loaded {len(records)} val records (device={device})")

    image_cache = load_image_cache(args.image_cache)

    from transformers import XLMRobertaTokenizerFast
    tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")

    dataset = SpanTaggingDataset(records, tokenizer, image_cache, max_length=config["max_length"])
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=lambda b: collate_fn(b, pad_token_id=tokenizer.pad_token_id),
        num_workers=0,
    )

    model = SpanTagger().to(device)
    state = torch.load(checkpoint_dir / "model.pt", map_location=device)
    model.load_state_dict(state)
    model.eval()

    thresholds = DEFAULT_THRESHOLDS
    if args.thresholds:
        thresholds = {**DEFAULT_THRESHOLDS, **json.loads(Path(args.thresholds).read_text())}

    from tqdm.auto import tqdm

    predictions = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="predicting", unit="batch", dynamic_ncols=True):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            image_tokens = batch["image_tokens"].to(device)
            logits = model(input_ids, attention_mask, image_tokens)
            probs = torch.sigmoid(logits).float().cpu()

            for i, meta in enumerate(batch["meta"]):
                token_probs = probs[i, :meta["length"]]
                spans = decode_spans(token_probs, meta["offsets"], meta["seq_ids"], thresholds=thresholds)
                predictions.append({"id": meta["id"], "labels": spans})

    data_io.write_jsonl(args.output, predictions)
    print(f"Wrote {len(predictions)} predictions -> {args.output}")


if __name__ == "__main__":
    main()
