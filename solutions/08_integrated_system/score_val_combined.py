"""Per-language IoU / Cor / Cor+Lbl on validation, for every combined-prediction
approach tried in Aim 4 -- the same shape of table Vroom-Vroom's paper reports
(Table 1), not the per-(category, language) cell breakdowns used elsewhere in
this project to pick thresholds.

Four prediction sets are built and scored, each covering every val record
across all 5 categories merged together (the native unit the official scorer
actually works on):

    hard_selection    -- finetuned_cell_winners.json's per-cell winner, spans
                          taken as-is (recomputes solutions/08_integrated_system/
                          val_scores_percell_selection.json's local, already-
                          verified result -- printed side by side as a sanity
                          check; the two should match exactly).
    full_blend        -- the tuned alpha blend (blend_selection_table's own
                          alpha/start/continue/min_span) for every cell.
    patched_blend     -- full_blend, except the 7 cells identified as
                          over-firing on test fall back to pure fine-tuned.
    uncertainty_only  -- alpha=0: blend_selection_table's "uncertainty_only"
                          sub-block for every cell, fine-tuned side ignored.

Needs val feature shards and the GBM-window-1 models -- run on nlp2.

Usage:
    python score_val_combined.py \
        --features_dir ../07_uncertainty_quantification/features \
        --models_dir ../07_uncertainty_quantification/models_clean_and_other_gbm_w1 \
        --blend_table blend_selection_table_models_clean_and_other_gbm_w1.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_UQ_DIR = _REPO_ROOT / "solutions" / "07_uncertainty_quantification"
if str(_UQ_DIR) not in sys.path:
    sys.path.insert(0, str(_UQ_DIR))

from solutions.common import data_io, scorer_local
from train_probes import load_shards
from tune_blend import SOLUTIONS, whitebox_predictions_by_record, char_probs_from_tokens
from span_assembly import merge_positive_tokens_to_spans_hysteresis

BROKEN_CELLS = {
    ("invention", "fr"), ("mischaracterization", "en"), ("mischaracterization", "fr"),
    ("mischaracterization", "it"), ("mischaracterization", "zh"),
    ("OCR", "zh"), ("other", "zh"),
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=str(_UQ_DIR / "features"))
    p.add_argument("--models_dir", required=True)
    p.add_argument("--blend_table", required=True)
    p.add_argument("--finetuned_winners", default=str(_THIS_DIR / "finetuned_cell_winners.json"))
    p.add_argument("--output_prefix", default=str(_THIS_DIR / "val_scores"))
    args = p.parse_args()

    data = load_shards(args.features_dir)
    val_mask = data["source_split"] == "val"
    per_token_keys = [k for k in data if k != "lookback_columns"]
    data = {k: (data[k][val_mask] if k in per_token_keys else data[k]) for k in data}

    gold = data_io.load_jsonl(data_io.VAL_FILE)
    gold_by_id = {r["id"]: r for r in gold}
    text_lens = {r["id"]: len(r["response"]) for r in gold}
    char_offsets = {rid: [(i, i + 1) for i in range(n)] for rid, n in text_lens.items()}

    with open(args.blend_table, encoding="utf-8") as f:
        table = json.load(f)
    with open(args.finetuned_winners, encoding="utf-8") as f:
        winners = json.load(f)
    with open(Path(args.models_dir) / "models_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    bb_cache = {}

    def load_bb(sol):
        if sol not in bb_cache:
            path = _REPO_ROOT / "solutions" / SOLUTIONS[sol] / "val_predictions.jsonl"
            bb_cache[sol] = {r["id"]: r for r in data_io.load_jsonl(path)}
        return bb_cache[sol]

    out = {name: {rid: [] for rid in gold_by_id} for name in
           ("hard_selection", "full_blend", "patched_blend", "uncertainty_only")}

    for category in data_io.LABEL_ORDER:
        for language in sorted(data_io.VALID_LANGUAGES):
            cell_key = f"{category}/{language}"
            cfg = table[cell_key]
            winner_sol = winners[cell_key]["best_solution"]
            bb_by_id = load_bb(winner_sol)
            model_entry = joblib.load(Path(args.models_dir) / f"{category}_{language}.joblib")
            wb_by_record = whitebox_predictions_by_record(data, language, model_entry)

            lang_ids = [rid for rid, r in gold_by_id.items() if r["language"] == language]
            for rid in lang_ids:
                n = text_lens[rid]
                ft_spans = [s for s in bb_by_id.get(rid, {}).get("labels", []) if s["label"] == category]
                out["hard_selection"][rid].extend(ft_spans)

                bb_vec = np.array(data_io.char_labels_to_vector(
                    bb_by_id.get(rid, {}).get("labels", []), n, label=category))
                if rid in wb_by_record:
                    offs, probs = wb_by_record[rid]
                    wb_vec = char_probs_from_tokens(offs, probs, n)
                else:
                    wb_vec = np.zeros(n)

                blended = cfg["alpha"] * bb_vec + (1 - cfg["alpha"]) * wb_vec
                full_spans = merge_positive_tokens_to_spans_hysteresis(
                    char_offsets[rid], blended.tolist(), category,
                    cfg["start_threshold"], cfg["continue_threshold"], min_span_tokens=cfg["min_span_tokens"])
                out["full_blend"][rid].extend(full_spans)
                out["patched_blend"][rid].extend(ft_spans if (category, language) in BROKEN_CELLS else full_spans)

                uq = cfg.get("uncertainty_only")
                if uq:
                    uq_spans = merge_positive_tokens_to_spans_hysteresis(
                        char_offsets[rid], wb_vec.tolist(), category,
                        uq["start_threshold"], uq["continue_threshold"], min_span_tokens=uq["min_span_tokens"])
                    out["uncertainty_only"][rid].extend(uq_spans)

    print(f"{'approach':18}{'lang':6}{'IoU':>9}{'Cor':>9}{'Cor_lbl':>9}")
    for name, by_id in out.items():
        pred_records = [{"id": rid, "labels": sorted(spans, key=lambda s: s["start"])}
                        for rid, spans in by_id.items()]
        result = scorer_local.score_predictions(gold, pred_records, verbose=False)
        for lang, m in sorted(result["per_language"].items()):
            print(f"{name:18}{lang:6}{m['iou']:>9.4f}{m['cor']:>9.4f}{m['cor_lbl']:>9.4f}")
        out_path = f"{args.output_prefix}_{name}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
        print(f"  saved -> {out_path}")

    print("\nSanity check: 'hard_selection' row above should match "
          "solutions/08_integrated_system/val_scores_percell_selection.json exactly "
          "(en 0.3838 / fr 0.3945 / it 0.4197 / zh 0.5010 Cor_lbl) -- if it doesn't, "
          "something about this script's record building diverges from that known-good run.")


if __name__ == "__main__":
    main()
