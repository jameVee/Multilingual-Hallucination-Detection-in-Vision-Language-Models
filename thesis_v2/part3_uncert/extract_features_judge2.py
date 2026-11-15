"""R7: probe features from a SECOND frozen judge of another family (Gemma-3-12B).

Same features as the Qwen probe (entropy, nll, attention look-back per head, last-layer
hidden state), computed by the old extractor's per-record function unchanged
(solutions/07_uncertainty_quantification/extract_features.py, loaded read-only), with
the same plain VQA framing: the frozen, NOT fine-tuned model is given (image, question)
and the real answer as its own assistant turn under teacher forcing.

Differences from the Qwen run, all forced by the model (results/preregistration.md, R7):
  - Gemma-3-12B does not fit in bf16 next to eager attention on a 24 GB card, so the
    language model is loaded in 4-bit NF4 (vision tower in bf16), as in its Part 1 run.
  - Gemma's processor wants nested images ([[img]]); a thin wrapper does that.
  - Gemma's chat template already starts with <bos>; the wrapper then tokenizes without
    adding a second one.
  - Every record of train/val/test is extracted here (the Qwen run was split in two).

Output: RUNS_DIR/part3/features_<backbone>/{split}_chunk{NNN}.npz, CHUNK records each, in
the old shard format. Finished chunks are skipped, so a crashed run resumes by rerunning.

Usage (nlp2):
    python -m thesis_v2.part3_uncert.extract_features_judge2 --split val --gpu_index 0 --dry_run
    python -m thesis_v2.part3_uncert.extract_features_judge2 --split train --gpu_index 0
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import torch

from thesis_v2.common import backbones
from thesis_v2.common.paths import RUNS_DIR
from thesis_v2.common.vlm_runtime import load_model, load_processor, resolve_device
from thesis_v2.part3_uncert.extract_features_all import load_old_extractor
from solutions.common import data_io

CHUNK = 500
SPLIT_FILES = {"train": data_io.TRAIN_FILE, "val": data_io.VAL_FILE, "test": data_io.TEST_FILE}


def features_dir(backbone_name):
    return RUNS_DIR / "part3" / f"features_{backbone_name}"


class ProcessorAdapter:
    """Presents the interface process_record expects: apply_chat_template, tokenizer,
    and __call__(text=[...], images=[img]) -- nesting images and avoiding a double BOS."""

    def __init__(self, processor, bb):
        self.p, self.bb = processor, bb
        self.tokenizer = getattr(processor, "tokenizer", processor)

    def apply_chat_template(self, *a, **k):
        return self.p.apply_chat_template(*a, **k)

    def __call__(self, text, images, **k):
        if self.bb.nested_images:
            images = [images]
        bos = self.tokenizer.bos_token
        if bos and all(t.startswith(bos) for t in text):
            k["add_special_tokens"] = False
        return self.p(text=text, images=images, **k)


def save_chunk(path, results, skipped, hidden_layers):
    keys = ["entropy", "nll", "label_id", "offset_start", "offset_end"]
    kw = {k: np.concatenate([r[k] for r in results]) if results else np.zeros(0) for k in keys}
    per_tok = lambda f: [f(r) for r in results for _ in range(r["n_tokens"])]
    kw |= {"language": np.array(per_tok(lambda r: r["language"]), dtype=object),
           "record_id": np.array(per_tok(lambda r: r["record_id"]), dtype=object),
           "approx_alignment": np.array(per_tok(lambda r: r["approx_alignment"]), dtype=bool),
           "lookback": np.concatenate([r["lookback"] for r in results], axis=0) if results else np.zeros((0, 0)),
           "lookback_columns": np.array(results[0]["lookback_columns"] if results else [], dtype=object),
           "skipped_ids": np.array(skipped, dtype=object)}
    for i in hidden_layers:
        kw[f"hidden_{i}"] = (np.concatenate([r["hidden"][str(i)] for r in results], axis=0).astype(np.float16)
                             if results else np.zeros((0, 0)))
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **kw)
    tmp.replace(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--split", choices=list(SPLIT_FILES), required=True)
    p.add_argument("--backbone", default="gemma3_12b")
    p.add_argument("--gpu_index", type=int, default=None)
    p.add_argument("--hidden_layers", default="-1")
    p.add_argument("--max_seq_len", type=int, default=4096)
    p.add_argument("--dry_run", action="store_true")
    args = p.parse_args()
    hidden_layers = [int(x) for x in args.hidden_layers.split(",")]
    old = load_old_extractor()
    bb = backbones.get(args.backbone)

    device = resolve_device(args.gpu_index)
    processor = ProcessorAdapter(load_processor(bb), bb)
    # eager attention: the look-back feature needs the attention weights
    model, _ = load_model(bb, device, load_in_4bit=True, attn_implementation="eager")
    model.eval()
    image_token_id = old.find_image_token_id(model, processor.p)
    print(f"{bb.name}: image_token_id={image_token_id}  device={device}")

    records = data_io.load_jsonl(SPLIT_FILES[args.split])
    out_dir = features_dir(bb.name)
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        for rec in records[:3]:
            t0 = time.time()
            res = old.process_record(rec, model, processor, device, image_token_id, hidden_layers,
                                     max_seq_len=args.max_seq_len, verbose=True)
            if res is None:
                print(f"  id={rec['id']} skipped")
                continue
            lb = res["lookback"]
            print(f"  id={res['record_id']} n_tokens={res['n_tokens']} approx_alignment={res['approx_alignment']} "
                  f"lookback={lb.shape} (mean {lb.mean():.3f}, share of zero cols {(lb.max(0) == 0).mean():.2f}) "
                  f"hidden={res['hidden']['-1'].shape} entropy_mean={res['entropy'].mean():.2f} "
                  f"labels={int((res['label_id'] >= 0).sum())} {time.time() - t0:.1f}s  "
                  f"max_mem={torch.cuda.max_memory_allocated() / 2**30:.1f} GiB")
        enc = processor.p.apply_chat_template([{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
                                              tokenize=False, add_generation_prompt=True)
        print("chat template head:", repr(enc[:40]))
        return

    n_chunks = (len(records) + CHUNK - 1) // CHUNK
    from tqdm.auto import tqdm
    for c in range(n_chunks):
        path = out_dir / f"{args.split}_chunk{c:03d}.npz"
        if path.exists():
            continue
        results, skipped = [], []
        for rec in tqdm(records[c * CHUNK:(c + 1) * CHUNK], desc=f"{args.split} chunk {c + 1}/{n_chunks}"):
            res = old.process_record(rec, model, processor, device, image_token_id, hidden_layers,
                                     max_seq_len=args.max_seq_len)
            if res is None:
                skipped.append(rec["id"])
            else:
                results.append(res)
        save_chunk(path, results, skipped, hidden_layers)
        n_approx = sum(r["approx_alignment"] for r in results)
        print(f"saved {path.name}: {len(results)} records, {len(skipped)} skipped, {n_approx} approx-aligned",
              flush=True)


if __name__ == "__main__":
    main()
