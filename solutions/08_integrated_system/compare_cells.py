"""Aim 4, step 1: which fine-tuned solution (02-06) actually wins each of
the 20 (category, language) cells, on val -- the "best fine-tuned option
per cell" half of Aim 4's selection table (the other half, once available,
is which Aim 3 uncertainty signal wins each cell; see
solutions/07_uncertainty_quantification/aim3_auc_clean_only_logreg_w0.json).

Runs entirely on already-committed local data (no GPU, no nlp2 needed):
each of solutions 02-06 already has a val_predictions.jsonl in the repo.

Per-cell metric: the official Cor+Lbl per-label term (scorer_local.py's
`per_label_language[cell]["cor"]` -- literally score_cor_lbl's own
per-label Spearman correlation, not a re-derived proxy), since that's the
paper's own primary metric and the one Aim 1's val multi-seed analysis
showed to be the metric augmentation reliably does NOT move -- exactly why
the per-cell selection matters instead of assuming one config wins
everywhere.

Usage:
    python compare_cells.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io, scorer_local

SOLUTIONS = {
    "02": "02_vlm_lora_real_data",
    "03": "03_vlm_lora_synthetic_augmented",
    "04": "04_vlm_lora_synthetic_augmented_llm",
    "05": "05_vlm_lora_synthetic_tuned",
    "06": "06_vlm_lora_synthetic_tuned_llm",
}

LABEL_ORDER = data_io.LABEL_ORDER
LANGUAGES = sorted(data_io.VALID_LANGUAGES)


def main():
    gold = data_io.load_jsonl(data_io.VAL_FILE)

    per_solution = {}
    for key, folder in SOLUTIONS.items():
        pred_path = _REPO_ROOT / "solutions" / folder / "val_predictions.jsonl"
        if not pred_path.exists():
            print(f"  WARNING: {pred_path} missing, skipping solution {key}")
            continue
        preds = data_io.load_jsonl(pred_path)
        result = scorer_local.score_predictions(gold, preds, verbose=False)
        per_solution[key] = result["per_label_language"]
        print(f"Scored solution {key} ({folder})")

    print(f"\n{'category':22} {'language':8} {'best':6} {'cor_lbl':>8}   per-solution cor_lbl")
    winners = {}
    for label in LABEL_ORDER:
        for lang in LANGUAGES:
            cell_key = f"{label}/{lang}"
            scores = {sol: per_solution[sol][cell_key]["cor"] for sol in per_solution
                      if cell_key in per_solution[sol]}
            if not scores:
                continue
            best_sol = max(scores, key=scores.get)
            winners[cell_key] = {"best_solution": best_sol, "cor_lbl": scores[best_sol],
                                  "all_solutions": scores}
            detail = "  ".join(f"{sol}={v:.4f}" for sol, v in sorted(scores.items()))
            print(f"{label:22} {lang:8} {best_sol:6} {scores[best_sol]:>8.4f}   {detail}")

    win_counts = {}
    for v in winners.values():
        win_counts[v["best_solution"]] = win_counts.get(v["best_solution"], 0) + 1
    print(f"\nWin counts across all 20 cells: {win_counts}")

    out_path = _THIS_DIR / "finetuned_cell_winners.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(winners, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
