"""Aim 4: for each (category, language) cell, apply that cell's persisted
Aim 3 probe model to every held-out val token in that language, assemble
spans at a sweep of candidate thresholds, score each with the OFFICIAL
scorer, keep the threshold that maximizes held-out Cor+Lbl for that cell --
then compare the uncertainty system's best achievable score against the
best fine-tuned solution's score for the same cell
(finetuned_cell_winners.json) to decide the final per-cell source. This is
Aim 4's "selects whichever signal... performed best on validation, locks
in its tuned threshold" step.

**Scoring scope**: a cell's comparison runs over every val record extracted
for that language (anything with >=1 gold label of ANY category -- exactly
what extract_features.py kept), not just records with a gold span of the
exact category being scored. scorer_local.py's own `per_label_language`
logic then correctly skips a record from the average only when BOTH its
gold and predicted labels for this category are empty -- the same
treatment the fine-tuned side already gets from being run over the whole
val set. (An earlier version restricted to gold-span-present records only,
which let the uncertainty side's false positives on other-category records
go completely unscored -- fixed; see git history for the real run that
exposed it.)

**Remaining scope caveat, stated plainly**: Aim 3's feature shards only
cover LABELED records (extract_features.py filters to `r.get("labels")`)
-- clean (zero-hallucination) val records were never extracted. This
script still cannot measure how often the uncertainty system would fire a
false positive on a genuinely clean response, since none were ever run
through it at all. Worth closing before trusting this beyond the cell-vs-
cell comparison it's built for.

**Hysteresis span assembly** (added after the first real evaluation showed
flat-threshold assembly losing 19/20 cells even at a near-1.0 threshold):
sweeps (start_threshold, continue_threshold, min_span_tokens) instead of a
single flat threshold, via span_assembly.py's
merge_positive_tokens_to_spans_hysteresis -- see that function's docstring
for why. Set --continue_offsets 0 --min_span_tokens_grid 1 to recover the
old flat-threshold-only behavior exactly.

Usage:
    python tune_thresholds.py \
        --features_dir ../07_uncertainty_quantification/features \
        --models_dir ../07_uncertainty_quantification/models
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from itertools import groupby
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
from train_probes import load_shards, LABEL_ORDER, LANGUAGES, windowed_features_for
from span_assembly import merge_positive_tokens_to_spans_hysteresis

# Canonical signal order -- MUST match train_probes.py's `signals` list
# construction, since an ensemble model's feature matrix was fit on
# columns concatenated in this exact order.
CANONICAL_SIGNAL_ORDER = ["entropy_nll", "lookback"]


def canonical_order(signal_list, data):
    hidden_signals = sorted(k for k in data if k.startswith("hidden_") and data[k].shape[1] > 0)
    order = CANONICAL_SIGNAL_ORDER + hidden_signals
    return [s for s in order if s in signal_list]


def build_feature_matrix(data, idx, model_entry):
    """Reuses train_probes.py's windowed_features_for (not a local copy) so a
    windowed model gets fed the SAME window size at inference time it was
    fit with -- a mismatch here would silently feed the classifier the
    wrong feature shape/semantics. `model_entry.get("window", 0)` defaults
    to 0 (no windowing) for any model saved before this option existed."""
    window = model_entry.get("window", 0)
    if model_entry["kind"] == "single":
        return windowed_features_for(data, idx, model_entry["signal"], window)
    signals = canonical_order(model_entry["signal"], data)
    return np.concatenate([windowed_features_for(data, idx, s, window) for s in signals], axis=1)


def group_by_record(record_ids):
    """Returns [(record_id, start_idx, end_idx_exclusive), ...] -- contiguous
    runs, same assumption as compute_offsets.py's run_length_groups."""
    groups = []
    pos = 0
    for key, g in groupby(record_ids):
        n = sum(1 for _ in g)
        groups.append((key, pos, pos + n))
        pos += n
    return groups


def tune_cell(data, category, language, model_entry, gold_by_id, configs):
    """`configs`: list of (start_threshold, continue_threshold, min_span_tokens)
    tuples to sweep. Returns (best_config, best_cor_lbl, n_val_records) or
    None if there's not enough val data in this language to tune against at
    all."""
    lang_mask = data["language"] == language
    idx = np.where(lang_mask)[0]
    if len(idx) == 0:
        return None

    X = build_feature_matrix(data, idx, model_entry)
    Xs = model_entry["scaler"].transform(X)
    probs = model_entry["clf"].predict_proba(Xs)[:, 1]

    record_ids = data["record_id"][idx]
    offset_start = data["offset_start"][idx]
    offset_end = data["offset_end"][idx]
    groups = group_by_record(record_ids.tolist())

    # ALL extracted records in this language -- NOT just ones with a gold span
    # of THIS category. An earlier version restricted to gold-span-present
    # records only, which let the uncertainty side's false positives on
    # OTHER-category-but-not-this-one records go completely unscored (those
    # records were simply never included), while the fine-tuned side's score
    # in finetuned_cell_winners.json WAS penalized for the equivalent mistake
    # (its predictions were run over the whole val set). That asymmetry favors
    # the uncertainty side and plausibly explains why its win margins looked
    # implausibly large (e.g. OCR's held-out scores more than doubling the
    # fine-tuned side's) on the first run. scorer_local's own per_label_language
    # logic already correctly skips a record from the average only when BOTH
    # gold and predicted labels for this category are empty for it -- same
    # treatment the fine-tuned side already gets -- so widening this to every
    # extracted record and letting that logic do the filtering makes the two
    # sides comparable on equal terms. (Fully clean, zero-hallucination
    # records are STILL never in here at all -- see the module docstring's
    # remaining scope caveat, which this does not fix.)
    relevant_record_ids = {rid for rid, s, e in groups if rid in gold_by_id}
    if not relevant_record_ids:
        return None

    gold_subset = [gold_by_id[rid] for rid, _, _ in groups if rid in relevant_record_ids]
    best_config, best_cor, best_n_contributing = None, -1.0, 0
    for start_t, continue_t, min_span in configs:
        pred_records = []
        for rid, start, end in groups:
            if rid not in relevant_record_ids:
                continue
            offsets = list(zip(offset_start[start:end].tolist(), offset_end[start:end].tolist()))
            token_probs = probs[start:end].tolist()
            spans = merge_positive_tokens_to_spans_hysteresis(
                offsets, token_probs, category, start_t, continue_t, min_span_tokens=min_span)
            pred_records.append({"id": rid, "labels": spans})

        result = scorer_local.score_predictions(gold_subset, pred_records, verbose=False)
        cell_key = f"{category}/{language}"
        cell_stats = result["per_label_language"].get(cell_key, {})
        cor = cell_stats.get("cor", float("nan"))
        if cor == cor and cor > best_cor:  # NaN-safe (NaN != NaN)
            best_cor, best_config = cor, (start_t, continue_t, min_span)
            best_n_contributing = cell_stats.get("n_records_with_label", 0)

    if best_config is None:
        return None
    # best_n_contributing: how many records actually had a gold OR predicted span of
    # this category at the winning config (what per_label_language itself averaged
    # over) -- a more meaningful count than len(relevant_record_ids), which is every
    # extracted record in this language regardless of whether this category ever
    # appears in it (see the widened-scope comment above).
    return best_config, best_cor, best_n_contributing


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=str(_UQ_DIR / "features"))
    p.add_argument("--models_dir", default=str(_UQ_DIR / "models"))
    p.add_argument("--finetuned_winners", default=str(_THIS_DIR / "finetuned_cell_winners.json"))
    p.add_argument("--start_thresholds",
                    default="0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.995",
                    help="Candidate start_threshold values (the bar to BEGIN a span). "
                         "Densified near 1.0 after a real run picked 0.9 (the old grid's highest "
                         "value) as 'best' for most cells -- the optimum sitting at the edge of the "
                         "range it was given is a sign the true optimum is past that edge.")
    p.add_argument("--continue_offsets",
                    default="0,0.1,0.2,0.3",
                    help="Candidate (start_threshold - continue_threshold) gaps -- 0 recovers flat "
                         "single-threshold assembly for that start value. continue_threshold is "
                         "clipped to >=0.05, and any offset that would push it below that for a given "
                         "start value is skipped for that start value.")
    p.add_argument("--min_span_tokens_grid", default="1,2,3",
                    help="Candidate minimum span lengths (in tokens) to require; shorter assembled "
                         "spans are dropped. 1 keeps every span regardless of length.")
    p.add_argument("--output", default=str(_THIS_DIR / "selection_table.json"))
    args = p.parse_args()

    start_thresholds = [float(x) for x in args.start_thresholds.split(",")]
    continue_offsets = [float(x) for x in args.continue_offsets.split(",")]
    min_span_grid = [int(x) for x in args.min_span_tokens_grid.split(",")]
    configs = []
    for start_t in start_thresholds:
        for offset in continue_offsets:
            continue_t = max(0.05, start_t - offset)
            for min_span in min_span_grid:
                configs.append((start_t, continue_t, min_span))
    configs = sorted(set(configs))  # de-dup configs that collapsed to the same values after clipping
    print(f"Sweeping {len(configs)} (start, continue, min_span_tokens) configs per cell.")

    data = load_shards(args.features_dir)
    if "offset_start" not in data:
        raise RuntimeError(
            "Shards have no offset_start/offset_end -- run compute_offsets.py for every shard first."
        )
    # Only val-split tokens matter here (Aim 4 tunes/evaluates on held-out val,
    # same as every other threshold/selection decision in this project).
    # Explicit key list, not a length-based heuristic -- `lookback_columns` is
    # metadata (column names), not a per-token array, and must stay untouched.
    val_mask = data["source_split"] == "val"
    per_token_keys = [k for k in data if k != "lookback_columns"]
    data = {k: (data[k][val_mask] if k in per_token_keys else data[k]) for k in data}

    gold = data_io.load_jsonl(data_io.VAL_FILE)
    gold_by_id = {r["id"]: r for r in gold}

    with open(args.finetuned_winners, encoding="utf-8") as f:
        finetuned = json.load(f)

    manifest_path = Path(args.models_dir) / "models_manifest.json"
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)

    max_start = max(start_thresholds)
    selection = {}
    print(f"\n{'category':22} {'language':8} {'start':>6} {'cont':>6} {'minsp':>5} "
          f"{'unc_cor':>8} {'ft_cor':>8} {'ft_sol':>7} {'winner':>12}")
    for category in LABEL_ORDER:
        for language in LANGUAGES:
            cell_key = f"{category}/{language}"
            if cell_key not in manifest:
                print(f"{category:22} {language:8}   no persisted model -- skipping")
                continue
            entry = manifest[cell_key]
            model_entry = joblib.load(Path(args.models_dir) / f"{category}_{language}.joblib")

            tuned = tune_cell(data, category, language, model_entry, gold_by_id, configs)
            ft_cor = finetuned.get(cell_key, {}).get("cor_lbl", float("nan"))
            ft_sol = finetuned.get(cell_key, {}).get("best_solution", "?")

            if tuned is None:
                print(f"{category:22} {language:8} {'n/a':>6} {'n/a':>6} {'n/a':>5} {'n/a':>8} {ft_cor:>8.4f} "
                      f"{ft_sol:>7} {'finetuned':>12} (no val data for uncertainty side)")
                selection[cell_key] = {"source": "finetuned", "finetuned_solution": ft_sol, "cor_lbl": ft_cor}
                continue

            (start_t, continue_t, min_span), unc_cor, n_records = tuned
            winner = "uncertainty" if unc_cor > ft_cor else "finetuned"
            edge_flag = "  <-- start_threshold at edge of range, widen --start_thresholds" if start_t == max_start else ""
            print(f"{category:22} {language:8} {start_t:>6.3f} {continue_t:>6.3f} {min_span:>5d} "
                  f"{unc_cor:>8.4f} {ft_cor:>8.4f} {ft_sol:>7} {winner:>12}  (n={n_records}){edge_flag}")

            if winner == "uncertainty":
                selection[cell_key] = {
                    "source": "uncertainty", "start_threshold": start_t, "continue_threshold": continue_t,
                    "min_span_tokens": min_span, "cor_lbl": unc_cor,
                    "signal": entry["signal"], "n_val_records": n_records,
                }
            else:
                selection[cell_key] = {"source": "finetuned", "finetuned_solution": ft_sol, "cor_lbl": ft_cor}

    n_uncertainty_wins = sum(1 for v in selection.values() if v["source"] == "uncertainty")
    print(f"\nUncertainty signal wins {n_uncertainty_wins}/{len(selection)} cells.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(selection, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
