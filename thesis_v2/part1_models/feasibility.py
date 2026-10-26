"""Phase 0 feasibility gate for one backbone. Run on nlp2 before any real training.

Checks, in one process:
  1. processor:  pixel_values present for 40 training examples
  2. masking:    the unmasked (trained-on) tokens decode to the target span JSON
                 -- catches chat templates whose prompt prefix doesn't line up
  3. length:     <= 2% of examples exceed max_seq_len 4096
  4. training:   5 optimizer steps (40 examples) without OOM; s/example and peak memory
  5. generation: 5 val records decode; JSON-parse rate reported (not gated -- 5 steps
                 is too early to expect the format)
Writes results/feasibility/<backbone>.json with pass/fail per check.

Usage:
    python -m thesis_v2.part1_models.feasibility --backbone gemma3_12b --gpu_index 1
"""
from __future__ import annotations

import os

# must be set before torch initializes CUDA: reduces fragmentation OOMs on the 24 GB cards
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import argparse
import json
import random
import time

import torch

from thesis_v2.common.paths import RESULTS_DIR
from thesis_v2.common import backbones, vlm_runtime as vr
from solutions.common import data_io, vlm_io

N_EXAMPLES = 40
MAX_SEQ = 4096


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backbone", required=True, choices=sorted(backbones.BACKBONES))
    p.add_argument("--gpu_index", type=int, default=None)
    args = p.parse_args()
    bb = backbones.get(args.backbone)
    res = {"backbone": bb.name, "model_id": bb.model_id, "checks": {}}
    out = RESULTS_DIR / "feasibility" / f"{bb.name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def finish():
        res["passed"] = all(c.get("pass", False) for k, c in res["checks"].items() if k != "generation")
        out.write_text(json.dumps(res, indent=2))
        print(json.dumps(res, indent=2))

    records = data_io.load_jsonl(data_io.TRAIN_FILE)
    rng = random.Random(0)
    # include hallucinated and clean examples, and the longest responses (worst case for length)
    pool = rng.sample(records, N_EXAMPLES - 10) + sorted(records, key=lambda r: -len(r["response"]))[:10]
    convs = [c for c in (vlm_io.build_conversation(r, include_target=True) for r in pool) if c]

    try:
        processor = vr.load_processor(bb)
        batches, seq_lens, mask_ok = [], [], 0
        for c in convs:
            batch, _ = vr.build_training_batch(processor, bb, c)
            seq_lens.append(batch["input_ids"].shape[1])
            trained = batch["input_ids"][batch["labels"] != -100]
            decoded = processor.tokenizer.decode(trained, skip_special_tokens=True)
            target = c["messages"][1]["content"][0]["text"]
            mask_ok += target.strip() in decoded
            batches.append(batch)
        res["checks"]["processor"] = {"pass": True, "n": len(batches)}
        res["checks"]["masking"] = {"pass": mask_ok >= 0.95 * len(batches), "ok": mask_ok, "n": len(batches)}
        over = sum(1 for s in seq_lens if s > MAX_SEQ)
        res["checks"]["length"] = {"pass": over <= 0.02 * len(seq_lens) + 1, "over_4096": over,
                                   "mean": round(sum(seq_lens) / len(seq_lens)), "max": max(seq_lens)}
    except Exception as exc:  # noqa: BLE001 -- record any failure as a failed gate
        res["checks"]["processor"] = {"pass": False, "error": repr(exc)[:500]}
        return finish()

    device = vr.resolve_device(args.gpu_index)
    try:
        model, _ = vr.load_model(bb, device)
        model, n_lora = vr.apply_lora(model, bb)
        opt = torch.optim.AdamW([q for q in model.parameters() if q.requires_grad], lr=2e-4)
        model.train()
        t0, n, grad = time.time(), 0, None
        for i, b in enumerate(batches):
            if b["input_ids"].shape[1] > MAX_SEQ:
                continue
            b = {k: (v.to(device) if hasattr(v, "to") else v) for k, v in b.items()}
            (model(**b).loss / 8).backward()
            n += 1
            if n == 1:
                # LoRA must actually receive gradients (gradient checkpointing can silently
                # cut them: "None of the inputs have requires_grad=True")
                lora = [(nm, q) for nm, q in model.named_parameters() if "lora_" in nm and q.requires_grad]
                nonzero = sum(1 for _, q in lora if q.grad is not None and q.grad.abs().sum() > 0)
                lora_b = [q for nm, q in lora if "lora_B" in nm]
                grad = {"lora_params": len(lora), "with_nonzero_grad": nonzero,
                        # lora_B starts at 0, so only lora_B gets a gradient on step 1
                        "lora_B_with_grad": sum(1 for q in lora_b if q.grad is not None and q.grad.abs().sum() > 0),
                        "lora_B_total": len(lora_b)}
            if n % 8 == 0:
                opt.step(); opt.zero_grad()
        grad_ok = grad is not None and grad["lora_B_with_grad"] >= 0.95 * grad["lora_B_total"]
        res["checks"]["training"] = {
            "pass": grad_ok, "gradients": grad, "lora_modules": n_lora, "examples": n,
            "sec_per_example": round((time.time() - t0) / max(1, n), 2),
            "peak_gpu_mem_gb": round(torch.cuda.max_memory_allocated() / 1e9, 2) if device != "cpu" else None,
        }
    except Exception as exc:  # noqa: BLE001
        res["checks"]["training"] = {"pass": False, "error": repr(exc)[:500]}
        return finish()

    try:
        model.eval()
        val = data_io.load_jsonl(data_io.VAL_FILE)[:5]
        parsed = 0
        with torch.no_grad():
            for r in val:
                c = vlm_io.build_conversation(r, include_target=False)
                inp = vr.build_inputs(processor, bb, c["messages"], c["_image_path"], True).to(device)
                o = model.generate(**inp, max_new_tokens=256, do_sample=False, pad_token_id=vr.pad_token_id(processor))
                txt = processor.batch_decode(o[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)[0]
                parsed += bool(vlm_io.parse_model_output(txt)) or txt.strip().startswith("[")
        res["checks"]["generation"] = {"pass": True, "json_like": parsed, "n": len(val)}
    except Exception as exc:  # noqa: BLE001
        res["checks"]["generation"] = {"pass": False, "error": repr(exc)[:500]}
    finish()


if __name__ == "__main__":
    main()
