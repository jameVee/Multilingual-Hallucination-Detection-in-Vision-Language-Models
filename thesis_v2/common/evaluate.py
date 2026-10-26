"""Official-metric evaluation for thesis_v2, with the coverage guard that Aim 3/4 lacked.

Every number reported in thesis_v2 comes from here. It scores with the official
scorer functions (via solutions/common/scorer_local.py: score_iou, score_cor,
score_cor_lbl), per record, then aggregates per language. The primary number is
the 4-language mean Cor+Lbl (each language weighted equally, like the per-language
leaderboard).

Coverage guard: a prediction file must contain exactly the ids of the split it is
scored on. Feature-based methods must also pass `feature_ids` -- the ids that
actually had features -- and those must equal the split too. In Aim 3/4, clean
val records had no features, silently became empty predictions, and the official
scorer gives empty-gold + empty-pred a perfect 1.0, so 28% of val scored perfectly
for free. This guard makes that impossible to repeat.

Usage:
    python -m thesis_v2.common.evaluate --pred runs/x/val_predictions.jsonl --split val_select --name x
"""
from __future__ import annotations

import argparse
import json
import random

import numpy as np

from thesis_v2.common.paths import LANGUAGES, RESULTS_DIR
from thesis_v2.common.splits import load_split
from solutions.common import data_io, scorer_local

SCORES_DIR = RESULTS_DIR / "scores"
METRICS = ("cor_lbl", "cor", "iou")


class CoverageError(RuntimeError):
    pass


def _gold_by_id():
    return {r["id"]: r for r in data_io.load_jsonl(data_io.VAL_FILE)}


def check_coverage(pred_ids, split_ids, feature_ids=None, what="predictions"):
    pred_ids, split_ids = set(pred_ids), set(split_ids)
    missing, extra = split_ids - pred_ids, pred_ids - split_ids
    if missing or extra:
        raise CoverageError(f"{what} do not match the split: {len(missing)} missing "
                            f"(e.g. {sorted(missing)[:3]}), {len(extra)} extra (e.g. {sorted(extra)[:3]})")
    if feature_ids is not None:
        fmissing = split_ids - set(feature_ids)
        if fmissing:
            raise CoverageError(f"{len(fmissing)} split records have no features (e.g. {sorted(fmissing)[:3]}) "
                                f"-- their empty predictions would score 1.0 for free on clean records")


def per_record_scores(pred_records, split="val_select", feature_ids=None, gold=None):
    """-> list of {id, language, cor_lbl, cor, iou}, one per split record."""
    gold = gold or _gold_by_id()
    split_ids = load_split(split)
    pred_ids = [r["id"] for r in pred_records]
    if len(pred_ids) != len(set(pred_ids)):
        raise CoverageError("prediction file contains duplicate ids")
    unknown = set(pred_ids) - set(gold)
    if unknown:
        raise CoverageError(f"{len(unknown)} prediction ids are not val records (e.g. {sorted(unknown)[:3]})")
    # A prediction file may cover all of val and be scored on one half; ids outside
    # the split are ignored, but every split id must be present.
    check_coverage(set(pred_ids) & split_ids, split_ids, feature_ids)
    preds = {r["id"]: r for r in pred_records if r["id"] in split_ids}
    rows = []
    for rid in sorted(split_ids):
        ref = scorer_local._to_ref_dict(gold[rid])
        pred = scorer_local._to_pred_dict(preds[rid])
        rows.append({
            "id": rid, "language": gold[rid]["language"],
            "cor_lbl": scorer_local.score_cor_lbl(ref, pred),
            "cor": scorer_local.score_cor(ref, pred),
            "iou": scorer_local.score_iou(ref, pred),
        })
    return rows


def aggregate(rows):
    out = {"per_language": {}}
    for lg in LANGUAGES:
        sub = [r for r in rows if r["language"] == lg]
        out["per_language"][lg] = {m: float(np.mean([r[m] for r in sub])) for m in METRICS} | {"n": len(sub)}
    for m in METRICS:
        out[f"primary_{m}"] = float(np.mean([out["per_language"][lg][m] for lg in LANGUAGES]))
    out["primary"] = out["primary_cor_lbl"]
    return out


def evaluate(pred_records, split="val_select", feature_ids=None):
    gold = _gold_by_id()
    rows = per_record_scores(pred_records, split, feature_ids, gold)
    result = aggregate(rows)
    # per-category diagnostic from the official-wrapper code path (same functions)
    split_ids = load_split(split)
    sub_gold = [gold[i] for i in sorted(split_ids)]
    sub_pred = [r for r in pred_records if r["id"] in split_ids]
    result["per_label"] = scorer_local.score_predictions(sub_gold, sub_pred)["per_label"]
    result["split"] = split
    return result, rows


def paired_bootstrap(rows_a, rows_b, metric="cor_lbl", n_boot=2000, seed=0):
    """Difference (b - a) in the 4-language-mean metric, resampling records within
    each language (keeps languages equally weighted). Both row lists must cover the
    same ids. Returns (observed_diff, ci_low, ci_high)."""
    a = {r["id"]: r for r in rows_a}
    b = {r["id"]: r for r in rows_b}
    if set(a) != set(b):
        raise CoverageError("paired bootstrap needs identical record sets")
    by_lang = {lg: [i for i in a if a[i]["language"] == lg] for lg in LANGUAGES}
    da = {lg: np.array([b[i][metric] - a[i][metric] for i in ids]) for lg, ids in by_lang.items()}
    observed = float(np.mean([d.mean() for d in da.values()]))
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        boots.append(np.mean([d[rng.integers(0, len(d), len(d))].mean() for d in da.values()]))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return observed, float(lo), float(hi)


def save_scores(name, result):
    SCORES_DIR.mkdir(parents=True, exist_ok=True)
    path = SCORES_DIR / f"{name}__{result['split']}.json"
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


def format_result(name, result):
    pl = result["per_language"]
    langs = "  ".join(f"{lg}={pl[lg]['cor_lbl']:.4f}" for lg in LANGUAGES)
    return (f"{name:40} [{result['split']}] Cor+Lbl={result['primary']:.4f}  Cor={result['primary_cor']:.4f}  "
            f"IoU={result['primary_iou']:.4f}  | {langs}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--split", default="val_select", choices=["val_select", "val_confirm", "val_full"])
    p.add_argument("--name", required=True)
    p.add_argument("--feature_ids", default=None, help="Optional JSON list of ids that had features.")
    p.add_argument("--no_save", action="store_true")
    args = p.parse_args()
    preds = data_io.load_jsonl(args.pred)
    fids = json.loads(open(args.feature_ids, encoding="utf-8").read()) if args.feature_ids else None
    result, _ = evaluate(preds, args.split, fids)
    print(format_result(args.name, result))
    if not args.no_save:
        print("saved ->", save_scores(args.name, result))


if __name__ == "__main__":
    main()
