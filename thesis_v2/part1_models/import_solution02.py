"""Reuse solution 02 as the Part 1 Qwen3-VL-8B seed-42 run (no retraining).

Solution 02 used the same recipe as Part 1 (Qwen3-VL-8B-Instruct, 4-bit QLoRA, r16/a16,
lr 2e-4, 2 epochs, grad_accum 8, min/max pixels 256*28^2 / 1024*28^2, seed 42) -- this
script checks that from its train_config.json, then copies into
runs/part1/qwen3vl8b_seed42/ (the path every later step expects):
  adapter + processor files, val_predictions.jsonl, test_predictions.jsonl (the four
  per-language test files merged), and provenance.json (source + sha1 of every file).
Then scores val_select / val_confirm with the official-metric evaluator.

Note: this run came from the old training code (Colab), not part1_models/train.py.
Seeds 123/2024 and every Part 2 condition are trained with the new code, so the
"same pipeline" check is: a new-code Qwen seed must land within seed noise of it.

Usage:
    python -m thesis_v2.part1_models.import_solution02
"""
from __future__ import annotations

import hashlib
import json
import shutil

from thesis_v2.common import evaluate as ev
from thesis_v2.common.paths import OLD_SOLUTIONS, RUNS_DIR
from solutions.common import data_io

SRC = OLD_SOLUTIONS / "02_vlm_lora_real_data"
DST = RUNS_DIR / "part1" / "qwen3vl8b_seed42"
EXPECTED = {"model_id": "Qwen/Qwen3-VL-8B-Instruct", "epochs": 2, "grad_accum": 8, "lr": 0.0002,
            "lora_r": 16, "lora_alpha": 16, "load_in_4bit": True, "seed": 42,
            "min_pixels": 256 * 28 * 28, "max_pixels": 1024 * 28 * 28}


def sha1(path):
    return hashlib.sha1(path.read_bytes()).hexdigest()


def main():
    cfg = json.loads((SRC / "checkpoint" / "train_config.json").read_text())
    diff = {k: (cfg.get(k), v) for k, v in EXPECTED.items() if cfg.get(k) != v}
    if diff:
        raise SystemExit(f"solution 02 does not match the Part 1 recipe: {diff}")
    if DST.exists() and any(DST.iterdir()):
        raise SystemExit(f"{DST} is not empty -- delete it deliberately to re-import")
    DST.mkdir(parents=True, exist_ok=True)

    for f in (SRC / "checkpoint").iterdir():
        if f.is_file():
            shutil.copy2(f, DST / f.name)
    shutil.copy2(SRC / "val_predictions.jsonl", DST / "val_predictions.jsonl")
    test_rows = [r for lg in ("en", "fr", "it", "zh")
                 for r in data_io.load_jsonl(SRC / f"test_predictions.{lg}.jsonl")]
    test_ids = {r["id"] for r in data_io.load_jsonl(data_io.TEST_FILE)}
    if {r["id"] for r in test_rows} != test_ids or len(test_rows) != len(test_ids):
        raise SystemExit("solution 02 test predictions do not cover the test set exactly")
    with open(DST / "test_predictions.jsonl", "w", encoding="utf-8") as fh:
        for r in test_rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")

    prov = {"source": str(SRC.relative_to(OLD_SOLUTIONS.parent)), "imported_from": "solution 02 (no retraining)",
            "train_config": cfg, "files": {f.name: sha1(f) for f in sorted(DST.iterdir()) if f.is_file()}}
    (DST / "provenance.json").write_text(json.dumps(prov, indent=2))

    val = data_io.load_jsonl(DST / "val_predictions.jsonl")
    for split in ("val_select", "val_confirm"):
        res, _ = ev.evaluate(val, split)
        ev.save_scores("p1_qwen3vl8b_seed42", res)
        print(ev.format_result("p1_qwen3vl8b_seed42", res))
    print(f"-> {DST}")


if __name__ == "__main__":
    main()
