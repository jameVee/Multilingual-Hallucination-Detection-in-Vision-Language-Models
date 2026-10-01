"""Aim 4 test-set benchmark, step 1: merge the five existing fine-tuned
solutions' (02-06) test predictions into ONE combined per-language
prediction set, using finetuned_cell_winners.json's per-(category,
language) selection -- for a given test response, each category's spans
come from whichever solution won that (category, language) cell on val.

Deliberately scoped to the fine-tuned side only. The 2 cells where Aim 3's
uncertainty signal won on val (OCR/zh, mischaracterization/zh) fall back to
the best FINE-TUNED solution here too, same as every other cell -- their
marginal contribution is small (~+0.0025 of the ~+0.0275 total val gain;
see solutions/08_integrated_system/README.md), and testing the real,
primary effect (does best-per-cell fine-tuned selection generalize past
val, where it was both selected AND evaluated) matters more right now than
building the as-yet-nonexistent test-time uncertainty inference pipeline
for a small addition. This produces a complete, valid, all-20-cell
prediction set on its own.

Entirely local, no GPU, no nlp2 -- every solution's test predictions
already exist in this repo.

Usage:
    python merge_finetuned_predictions.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

SOLUTIONS = {
    "02": "02_vlm_lora_real_data",
    "03": "03_vlm_lora_synthetic_augmented",
    "04": "04_vlm_lora_synthetic_augmented_llm",
    "05": "05_vlm_lora_synthetic_tuned",
    "06": "06_vlm_lora_synthetic_tuned_llm",
}
LANGUAGES = sorted(data_io.VALID_LANGUAGES)


def main():
    with open(_THIS_DIR / "finetuned_cell_winners.json", encoding="utf-8") as f:
        winners = json.load(f)

    output_dir = _THIS_DIR / "test_predictions_integrated"
    output_dir.mkdir(exist_ok=True)

    source_usage = {}  # per (category, language) -> which solution actually got used
    for language in LANGUAGES:
        by_solution = {}
        for key, folder in SOLUTIONS.items():
            path = _REPO_ROOT / "solutions" / folder / f"test_predictions.{language}.jsonl"
            records = data_io.load_jsonl(path)
            by_solution[key] = {r["id"]: r for r in records}

        # Sanity: every solution must cover the exact same set of ids for this language --
        # a silent mismatch here would mean some response gets an incomplete merge.
        id_sets = {key: set(recs) for key, recs in by_solution.items()}
        reference_ids = id_sets["02"]
        for key, ids in id_sets.items():
            if ids != reference_ids:
                missing = reference_ids - ids
                extra = ids - reference_ids
                raise RuntimeError(
                    f"Solution {key}/{language}: id mismatch vs solution 02 "
                    f"(missing={len(missing)}, extra={len(extra)}) -- refusing to merge."
                )

        category_sources = {}
        for category in data_io.LABEL_ORDER:
            cell_key = f"{category}/{language}"
            winner_sol = winners.get(cell_key, {}).get("best_solution")
            if winner_sol is None:
                raise RuntimeError(f"No winner recorded for {cell_key} in finetuned_cell_winners.json")
            category_sources[category] = winner_sol
            source_usage[cell_key] = winner_sol

        merged_records = []
        for record_id in sorted(reference_ids):
            merged_labels = []
            for category, winner_sol in category_sources.items():
                source_record = by_solution[winner_sol][record_id]
                merged_labels.extend(
                    span for span in source_record.get("labels", [])
                    if span["label"] == category
                )
            merged_labels.sort(key=lambda s: s["start"])
            merged_records.append({"id": record_id, "labels": merged_labels})

        out_path = output_dir / f"test_predictions_integrated.{language}.jsonl"
        data_io.write_jsonl(out_path, merged_records)
        print(f"  {language}: {len(merged_records)} merged predictions -> {out_path}")
        print(f"    sources used: " + ", ".join(f"{c}={s}" for c, s in category_sources.items()))

    print("\nPer-cell source summary:")
    for cell_key, sol in source_usage.items():
        print(f"  {cell_key:28} -> solution {sol}")


if __name__ == "__main__":
    main()
