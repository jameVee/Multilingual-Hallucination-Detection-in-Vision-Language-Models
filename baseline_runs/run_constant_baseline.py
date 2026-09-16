"""Runs the official participant_kit/baselines/constant_baselines.py against
data/val.jsonl and scores every variant through the official scorer logic
(via solutions/common/scorer_local.py).

Does NOT modify participant_kit/ -- that script expects per-language files
named `shroom-vision.test.<lang>.jsonl` in a directory (its own glob:
`file.name.split('.')[2]` for language), which doesn't match this repo's
merged data/val.jsonl. This script's only job is the prep step: split
val.jsonl by language into that exact naming convention in a staging dir,
invoke the real script unchanged, then score each of its 6 output variants
per language (none, invention, mischaracterization, OCR, miscounting,
other) against the real gold labels.

Usage:
    python baseline_runs/run_constant_baseline.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, scorer_local

STAGING_DIR = _THIS_DIR / "constant_baseline" / "staging"
PRED_DIR = _THIS_DIR / "constant_baseline" / "predictions"
BASELINE_SCRIPT = _REPO_ROOT / "participant_kit" / "baselines" / "constant_baselines.py"

VARIANTS = ["none", "invention", "mischaracterization", "OCR", "miscounting", "other"]


def main():
    STAGING_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    gold = data_io.load_jsonl(data_io.VAL_FILE)
    by_lang = defaultdict(list)
    for r in gold:
        by_lang[r["language"]].append(r)

    print("=== Prep: splitting data/val.jsonl into per-language files constant_baselines.py expects ===")
    for lang, records in by_lang.items():
        out_path = STAGING_DIR / f"shroom-vision.test.{lang}.jsonl"
        with out_path.open("w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"  {out_path.name}: {len(records)} rows")

    print("\n=== Running participant_kit/baselines/constant_baselines.py (unmodified) ===")
    subprocess.run(
        [sys.executable, str(BASELINE_SCRIPT), str(STAGING_DIR), str(PRED_DIR)],
        check=True,
    )

    print("\n=== Scoring each variant against data/val.jsonl (official metric logic) ===")
    print(f"{'variant':22s} {'IoU':>8s} {'Cor':>8s} {'CorLbl':>8s}")
    results = {}
    for variant in VARIANTS:
        preds = []
        for lang in by_lang:
            variant_path = PRED_DIR / f"{lang}.mark_{variant}.jsonl"
            preds.extend(data_io.load_jsonl(variant_path))
        scores = scorer_local.score_predictions(gold, preds, verbose=False)
        results[variant] = scores
        o = scores["overall"]
        print(f"{variant:22s} {o['iou']:8.4f} {o['cor']:8.4f} {o['cor_lbl']:8.4f}")

    out_path = _THIS_DIR / "constant_baseline" / "val_scores.json"
    with out_path.open("w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved full breakdown (per-language, per-label) -> {out_path}")


if __name__ == "__main__":
    main()
