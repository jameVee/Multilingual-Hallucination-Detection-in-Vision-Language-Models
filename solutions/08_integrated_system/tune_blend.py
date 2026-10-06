"""Aim 4, attempt 2: blend instead of hard-select.

The earlier approach (finetuned_cell_winners.json + merge_finetuned_predictions.py)
hard-picked ONE of 5 fine-tuned solutions per (category, language) cell, based
on which scored highest on a small val slice. That backfired on the real test
set -- it scored worse than every single solution on its own (see
RESULTS_SUMMARY.md / the git history around "aim4-percell-selection-finetuned").
Diagnosis: hard argmax selection over many small, noisy per-cell comparisons
is a textbook "winner's curse" setup -- it selects for luck, not true
superiority, and stitching together 20 such picks compounds the noise instead
of averaging it out.

This script replaces that with a WEIGHTED AVERAGE, matching Rykov et al.
2025's SmurfCat/SemEval-2025 merge formula (the advisor's own prior work):

    p_final = alpha * p_blackbox + (1 - alpha) * p_whitebox

per character, where p_blackbox comes from the already-identified best
fine-tuned solution for that cell (finetuned_cell_winners.json -- unchanged,
still a single fixed choice per cell, not re-litigated here) and p_whitebox
comes from Aim 3's uncertainty probe for that cell. alpha and the span
assembly settings are chosen by the same held-out val scoring already used
throughout this project. Averaging two reasonable estimates reduces variance
instead of amplifying it -- that's the specific, principled fix for what went
wrong.

Character-level, not token-level: this is the native representation
scorer_local.py / the official scorer itself uses (data_io.char_labels_to_vector
already does exactly this accumulation for gold spans), so the blended
probability vector goes straight into the hysteresis span assembly
(span_assembly.py) -- a list of (char_index, char_index+1) "offsets" is all it
needs.

Search per cell (val only), two stages:
  Stage 1: alpha x flat (start == continue) threshold grid. Cheap; used only to
           rank the alphas.
  Stage 2: for the top-K alphas from stage 1 PLUS the two endpoints -- 0.0 (pure
           uncertainty, so the uncertainty-only number here is directly comparable
           to tune_thresholds.py) and 1.0 (pure fine-tuned) -- a hysteresis sweep
           over start_threshold x continue offset x min_span_tokens. The start and
           offset grids match tune_thresholds.py's, extended with 0.999 because the
           uncertainty-only sweep kept selecting the top of its range.
The best config seen across both stages is kept.

Records with NO Aim 3 features at all (clean, zero-hallucination val
records -- extract_features.py never extracted these) get p_whitebox = 0
everywhere, which is the mathematically correct default: a record with no
gold label at all has no gold label for THIS category either, so "predict
nothing" is exactly the right behavior for it, not a gap being papered over.

Usage:
    python tune_blend.py \
        --features_dir ../07_uncertainty_quantification/features \
        --models_dir ../07_uncertainty_quantification/models_gbm
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
from train_probes import load_shards, LABEL_ORDER, LANGUAGES
from tune_thresholds import group_by_record, build_feature_matrix
from span_assembly import merge_positive_tokens_to_spans_hysteresis

SOLUTIONS = {
    "02": "02_vlm_lora_real_data",
    "03": "03_vlm_lora_synthetic_augmented",
    "04": "04_vlm_lora_synthetic_augmented_llm",
    "05": "05_vlm_lora_synthetic_tuned",
    "06": "06_vlm_lora_synthetic_tuned_llm",
}

MIN_CONTINUE_THRESHOLD = 0.05  # same floor tune_thresholds.py applies to continue_threshold


def whitebox_predictions_by_record(data, language, model_entry):
    """{record_id: (offsets, token_probs)} for every val token in this
    language -- same grouping logic as tune_thresholds.py's tune_cell."""
    lang_mask = data["language"] == language
    idx = np.where(lang_mask)[0]
    if len(idx) == 0:
        return {}
    X = build_feature_matrix(data, idx, model_entry)
    Xs = model_entry["scaler"].transform(X)
    probs = model_entry["clf"].predict_proba(Xs)[:, 1]
    record_ids = data["record_id"][idx]
    offset_start = data["offset_start"][idx]
    offset_end = data["offset_end"][idx]
    groups = group_by_record(record_ids.tolist())
    out = {}
    for rid, start, end in groups:
        offsets = list(zip(offset_start[start:end].tolist(), offset_end[start:end].tolist()))
        token_probs = probs[start:end].tolist()
        out[rid] = (offsets, token_probs)
    return out


def char_probs_from_tokens(offsets, token_probs, text_len):
    char_probs = np.zeros(text_len, dtype=np.float64)
    for (s, e), p in zip(offsets, token_probs):
        char_probs[s:min(e, text_len)] = p
    return char_probs


def tune_cell_blend(category, language, bb_by_id, wb_by_record, gold_by_id, alphas,
                    flat_thresholds, start_grid, continue_offsets, min_span_grid, top_k_alphas):
    """Two-stage search described in the module docstring. Returns a dict with the
    chosen config and the uncertainty-only comparator, or None if nothing scored."""
    record_ids = sorted(bb_by_id)  # full language coverage -- every val record this solution predicted on
    if not record_ids:
        return None
    text_lens = {rid: len(gold_by_id[rid]["response"]) for rid in record_ids if rid in gold_by_id}
    record_ids = [rid for rid in record_ids if rid in text_lens]
    if not record_ids:
        return None

    bb_vecs = {}
    wb_vecs = {}
    for rid in record_ids:
        text_len = text_lens[rid]
        bb_vecs[rid] = np.array(
            data_io.char_labels_to_vector(bb_by_id[rid].get("labels", []), text_len, label=category)
        )
        if rid in wb_by_record:
            offsets, token_probs = wb_by_record[rid]
            wb_vecs[rid] = char_probs_from_tokens(offsets, token_probs, text_len)
        else:
            wb_vecs[rid] = np.zeros(text_len)  # never extracted -> genuinely clean record, 0 is correct

    char_offsets = {rid: [(i, i + 1) for i in range(text_lens[rid])] for rid in record_ids}
    gold_subset = [gold_by_id[rid] for rid in record_ids]
    cell_key = f"{category}/{language}"

    def blend_all(alpha):
        return {rid: alpha * bb_vecs[rid] + (1 - alpha) * wb_vecs[rid] for rid in record_ids}

    def score(blended, start, cont, min_span):
        pred_records = []
        for rid in record_ids:
            spans = merge_positive_tokens_to_spans_hysteresis(
                char_offsets[rid], blended[rid].tolist(), category, start, cont, min_span_tokens=min_span)
            pred_records.append({"id": rid, "labels": spans})
        result = scorer_local.score_predictions(gold_subset, pred_records, verbose=False)
        return result["per_label_language"].get(cell_key, {}).get("cor", float("nan"))

    # Each candidate: (cor, alpha, start_threshold, continue_threshold, min_span_tokens)
    stage1 = []
    for alpha in alphas:
        blended = blend_all(alpha)
        for t in flat_thresholds:
            stage1.append((score(blended, t, t, 1), alpha, t, t, 1))

    def stage1_best(alpha):
        vals = [c[0] for c in stage1 if c[1] == alpha and c[0] == c[0]]
        return max(vals) if vals else float("-inf")

    ranked = sorted(alphas, key=stage1_best, reverse=True)
    stage2_alphas = []
    for a in ranked[:top_k_alphas] + [0.0, 1.0]:
        if a in alphas and a not in stage2_alphas:
            stage2_alphas.append(a)

    seen = {(a, round(t, 6), round(c, 6), m) for _, a, t, c, m in stage1}
    stage2 = []
    for alpha in stage2_alphas:
        blended = blend_all(alpha)
        for start in start_grid:
            for offset in continue_offsets:
                cont = round(start - offset, 6)
                if cont < MIN_CONTINUE_THRESHOLD:
                    continue
                for min_span in min_span_grid:
                    key = (alpha, round(start, 6), cont, min_span)
                    if key in seen:
                        continue
                    seen.add(key)
                    stage2.append((score(blended, start, cont, min_span), alpha, start, cont, min_span))

    valid = [c for c in stage1 + stage2 if c[0] == c[0]]
    if not valid:
        return None
    # max() returns the FIRST maximal candidate, so ties go to the earlier (simpler,
    # stage-1) config -- a deliberate bias toward the least complicated explanation.
    best = max(valid, key=lambda c: c[0])
    uncert = [c for c in stage2 if c[1] == 0.0 and c[0] == c[0]]
    uncert_best = max(uncert, key=lambda c: c[0]) if uncert else None

    cor, alpha, start, cont, min_span = best
    return {
        "alpha": alpha, "start_threshold": start, "continue_threshold": cont,
        "min_span_tokens": min_span, "cor_lbl": cor, "n_records": len(record_ids),
        "n_configs_evaluated": len(stage1) + len(stage2),
        "uncertainty_only": (
            {"cor_lbl": uncert_best[0], "start_threshold": uncert_best[2],
             "continue_threshold": uncert_best[3], "min_span_tokens": uncert_best[4]}
            if uncert_best else None),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=str(_UQ_DIR / "features"))
    p.add_argument("--models_dir", default=str(_UQ_DIR / "models"))
    p.add_argument("--finetuned_winners", default=str(_THIS_DIR / "finetuned_cell_winners.json"))
    p.add_argument("--alphas", default="0.0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                    help="Blend weight for the fine-tuned side; 1.0 = pure fine-tuned, "
                         "0.0 = pure uncertainty.")
    p.add_argument("--flat_thresholds", default="0.3,0.5,0.7,0.9",
                    help="Stage 1: flat start==continue thresholds used only to rank alphas.")
    p.add_argument("--start_thresholds", default="0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.995,0.999",
                    help="Stage 2: candidate start_threshold values.")
    p.add_argument("--continue_offsets", default="0,0.1,0.2,0.3",
                    help="Stage 2: candidate (start - continue) gaps; 0 = flat for that start.")
    p.add_argument("--min_span_tokens_grid", default="1,2,3",
                    help="Stage 2: candidate minimum span lengths in tokens.")
    p.add_argument("--top_k_alphas", type=int, default=3,
                    help="Stage 2 refines this many best alphas from stage 1 (plus 0.0 and 1.0).")
    p.add_argument("--output", default=str(_THIS_DIR / "blend_selection_table.json"))
    args = p.parse_args()

    alphas = [float(x) for x in args.alphas.split(",")]
    flat_thresholds = [float(x) for x in args.flat_thresholds.split(",")]
    start_grid = [float(x) for x in args.start_thresholds.split(",")]
    continue_offsets = [float(x) for x in args.continue_offsets.split(",")]
    min_span_grid = [int(x) for x in args.min_span_tokens_grid.split(",")]

    data = load_shards(args.features_dir)
    if "offset_start" not in data:
        raise RuntimeError("Shards have no offsets -- run compute_offsets.py for every shard first.")
    val_mask = data["source_split"] == "val"
    per_token_keys = [k for k in data if k != "lookback_columns"]
    data = {k: (data[k][val_mask] if k in per_token_keys else data[k]) for k in data}

    gold = data_io.load_jsonl(data_io.VAL_FILE)
    gold_by_id = {r["id"]: r for r in gold}

    with open(args.finetuned_winners, encoding="utf-8") as f:
        finetuned_winners = json.load(f)

    manifest_path = Path(args.models_dir) / "models_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    # Cache: solution_key -> language -> {id: record}, loaded on demand (only the
    # solutions that actually win at least one cell need to be read).
    bb_cache = {}

    def load_bb(solution_key, language):
        cache_key = (solution_key, language)
        if cache_key not in bb_cache:
            path = _REPO_ROOT / "solutions" / SOLUTIONS[solution_key] / f"val_predictions.jsonl"
            records = data_io.load_jsonl(path)
            bb_cache[cache_key] = {r["id"]: r for r in records}
        return bb_cache[cache_key]

    selection = {}
    print(f"\n{'category':22} {'language':8} {'alpha':>6} {'start':>6} {'cont':>6} {'minsp':>5} "
          f"{'blend':>8} {'uncert':>8} {'pure_ft':>8} {'improved':>9}")
    for category in LABEL_ORDER:
        for language in LANGUAGES:
            cell_key = f"{category}/{language}"
            winner_sol = finetuned_winners.get(cell_key, {}).get("best_solution")
            pure_ft_cor = finetuned_winners.get(cell_key, {}).get("cor_lbl", float("nan"))
            if winner_sol is None or cell_key not in manifest:
                print(f"{category:22} {language:8}   missing winner or model -- skipping")
                continue

            bb_by_id = load_bb(winner_sol, language)
            model_entry = joblib.load(Path(args.models_dir) / f"{category}_{language}.joblib")
            wb_by_record = whitebox_predictions_by_record(data, language, model_entry)

            result = tune_cell_blend(category, language, bb_by_id, wb_by_record, gold_by_id,
                                      alphas, flat_thresholds, start_grid, continue_offsets,
                                      min_span_grid, args.top_k_alphas)
            if result is None:
                print(f"{category:22} {language:8}   no data -- skipping")
                continue
            blend_cor = result["cor_lbl"]
            improved = blend_cor > pure_ft_cor
            uncert = result["uncertainty_only"]["cor_lbl"] if result["uncertainty_only"] else float("nan")
            print(f"{category:22} {language:8} {result['alpha']:>6.1f} {result['start_threshold']:>6.3f} "
                  f"{result['continue_threshold']:>6.3f} {result['min_span_tokens']:>5d} "
                  f"{blend_cor:>8.4f} {uncert:>8.4f} {pure_ft_cor:>8.4f} {'YES' if improved else 'no':>9}")

            selection[cell_key] = {
                "finetuned_solution": winner_sol,
                "alpha": result["alpha"],
                "start_threshold": result["start_threshold"],
                "continue_threshold": result["continue_threshold"],
                "min_span_tokens": result["min_span_tokens"],
                "blend_cor_lbl": blend_cor,
                "uncertainty_only": result["uncertainty_only"],
                "pure_finetuned_cor_lbl": pure_ft_cor,
                "improved_over_pure_finetuned": improved,
                "n_records": result["n_records"],
                "n_configs_evaluated": result["n_configs_evaluated"],
            }

    n_improved = sum(1 for v in selection.values() if v["improved_over_pure_finetuned"])
    print(f"\nBlending improved {n_improved}/{len(selection)} cells over the pure fine-tuned baseline.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
