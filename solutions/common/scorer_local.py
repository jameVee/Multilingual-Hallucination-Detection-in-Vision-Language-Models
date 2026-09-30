"""Local scoring wrapper that reuses participant_kit/scorer.py's exact metric
functions (not a reimplementation) so results computed here match the
official scorer bit-for-bit on the two metrics docs/2026.md actually names:
Span IoU and Spearman correlation (overall + per-label).

The official scorer's reference loader (`load_jsonl_file_to_records`) also
requires a `raw_annots` column, used only by the auxiliary `scores_classif`
binary has-hallucination metric. Local data/val.jsonl doesn't have that
field (it isn't part of the released split), so instead of calling
scorer.main() we build ref/pred dicts ourselves (id, labels, text_len) and
call score_iou / score_cor / score_cor_lbl per record directly -- those three
don't touch raw_annots at all.
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io

_SCORER_PATH = _REPO_ROOT / "participant_kit" / "scorer.py"


def _load_official_scorer():
    spec = importlib.util.spec_from_file_location("shroom_official_scorer", _SCORER_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_official = _load_official_scorer()
score_iou = _official.score_iou
score_cor = _official.score_cor
score_cor_lbl = _official.score_cor_lbl


def _to_ref_dict(record: dict) -> dict:
    return {
        "id": record["id"],
        "labels": record.get("labels", []) or [],
        "text_len": len(record["response"]),
    }


def _to_pred_dict(record: dict) -> dict:
    return {"id": record["id"], "labels": record.get("labels", []) or []}


def score_predictions(gold_records, pred_records, verbose=False):
    """Score `pred_records` against `gold_records` with the official metrics.

    Mirrors scorer.py's main(): predictions are matched to gold by id (every
    predicted id must exist in gold), then scored per record and averaged.
    Returns overall + per-language + per-label(diagnostic) breakdowns.
    """
    gold_by_id = {r["id"]: r for r in gold_records}
    pred_ids = [r["id"] for r in pred_records]
    missing = [i for i in pred_ids if i not in gold_by_id]
    if missing:
        raise ValueError(
            f"{len(missing)} prediction ids have no matching gold record, e.g. {missing[:5]}"
        )
    if len(pred_ids) != len(set(pred_ids)):
        dupes = [i for i in set(pred_ids) if pred_ids.count(i) > 1]
        raise ValueError(f"Duplicate prediction ids: {dupes[:5]}")

    lang_by_id = {r["id"]: r.get("language", "unknown") for r in gold_records}

    ref_dicts = [_to_ref_dict(gold_by_id[i]) for i in pred_ids]
    pred_dicts = [_to_pred_dict(r) for r in pred_records]

    per_record = []
    for ref, pred in zip(ref_dicts, pred_dicts):
        per_record.append({
            "id": ref["id"],
            "iou": score_iou(ref, pred),
            "cor": score_cor(ref, pred),
            "cor_lbl": score_cor_lbl(ref, pred),
            "language": lang_by_id.get(ref["id"], "unknown"),
        })

    def _mean(key, subset=None):
        vals = [r[key] for r in per_record if subset is None or r["language"] == subset]
        return float(np.mean(vals)) if vals else float("nan")

    languages = sorted({r["language"] for r in per_record})
    result = {
        "n": len(per_record),
        "overall": {"iou": _mean("iou"), "cor": _mean("cor"), "cor_lbl": _mean("cor_lbl")},
        "per_language": {
            lang: {
                "iou": _mean("iou", lang), "cor": _mean("cor", lang), "cor_lbl": _mean("cor_lbl", lang),
                "n": sum(1 for r in per_record if r["language"] == lang),
            }
            for lang in languages
        },
    }

    # Diagnostic only (not part of the official rubric, which scores IoU
    # label-agnostically): character IoU restricted to spans of a single
    # label, so OCR/miscounting/other performance stays visible on its own
    # rather than being averaged away by the majority classes. The Spearman
    # side (`cor`) below IS exactly score_cor_lbl's own per-label term.
    per_label = {}
    for label in data_io.LABEL_ORDER:
        label_ious, label_cors = [], []
        for ref, pred in zip(ref_dicts, pred_dicts):
            ref_f = {**ref, "labels": [s for s in ref["labels"] if s["label"] == label]}
            pred_f = {**pred, "labels": [s for s in pred["labels"] if s["label"] == label]}
            if not ref_f["labels"] and not pred_f["labels"]:
                continue
            label_ious.append(score_iou(ref_f, pred_f))
            label_cors.append(score_cor(ref, pred, label_filtered_=label))
        per_label[label] = {
            "iou": float(np.mean(label_ious)) if label_ious else float("nan"),
            "cor": float(np.mean(label_cors)) if label_cors else float("nan"),
            "n_records_with_label": len(label_ious),
        }
    result["per_label"] = per_label

    # Category x language cross-breakdown (Aim 4): the per_language and per_label
    # blocks above each collapse the other axis away, but Aim 4's per-cell source
    # selection ("does the fine-tuned model or an Aim 3 probe win THIS category in
    # THIS language") needs both axes held fixed at once.
    per_label_language = {}
    for label in data_io.LABEL_ORDER:
        for lang in languages:
            lang_ids = {r["id"] for r in per_record if r["language"] == lang}
            label_ious, label_cors = [], []
            for ref, pred in zip(ref_dicts, pred_dicts):
                if ref["id"] not in lang_ids:
                    continue
                ref_f = {**ref, "labels": [s for s in ref["labels"] if s["label"] == label]}
                pred_f = {**pred, "labels": [s for s in pred["labels"] if s["label"] == label]}
                if not ref_f["labels"] and not pred_f["labels"]:
                    continue
                label_ious.append(score_iou(ref_f, pred_f))
                label_cors.append(score_cor(ref, pred, label_filtered_=label))
            per_label_language[f"{label}/{lang}"] = {
                "iou": float(np.mean(label_ious)) if label_ious else float("nan"),
                "cor": float(np.mean(label_cors)) if label_cors else float("nan"),
                "n_records_with_label": len(label_ious),
            }
    result["per_label_language"] = per_label_language

    if verbose:
        o = result["overall"]
        print(f"n={result['n']}  IoU={o['iou']:.4f}  Cor={o['cor']:.4f}  CorLbl={o['cor_lbl']:.4f}")
        for lang, m in result["per_language"].items():
            print(f"  [{lang}] n={m['n']:4d}  IoU={m['iou']:.4f}  Cor={m['cor']:.4f}  CorLbl={m['cor_lbl']:.4f}")
        for label, m in result["per_label"].items():
            print(f"  <{label:20s}> n={m['n_records_with_label']:4d}  IoU={m['iou']:.4f}  Cor={m['cor']:.4f}")

    return result


def score_files(gold_path, pred_path, verbose=False):
    gold_records = data_io.load_jsonl(gold_path)
    pred_records = data_io.load_jsonl(pred_path)
    return score_predictions(gold_records, pred_records, verbose=verbose)


if __name__ == "__main__":
    # Sanity check: scoring gold val labels against themselves should give
    # IoU == 1.0 and CorLbl == 1.0 exactly. Overall (unfiltered) Cor will be
    # *slightly* below 1.0 on identity input whenever a response has
    # overlapping spans with different labels at the same characters (real
    # in this dataset, e.g. annotator disagreement over invention vs.
    # mischaracterization on the same text) -- this is inherent to
    # score_cor's own logic, not a bug here: the reference vector SUMS
    # overlapping spans' probabilities per character while the prediction
    # vector OVERWRITES (last span wins), so even a byte-identical
    # prediction can't perfectly reconstruct a summed reference signal.
    # This must pass before any model's val_scores.json is trusted.
    gold = data_io.load_jsonl(data_io.VAL_FILE)
    identity_preds = [{"id": r["id"], "labels": r.get("labels", []) or []} for r in gold]
    print("Sanity check: gold predicted back against itself (expect ~1.0 everywhere)")
    out = score_predictions(gold, identity_preds, verbose=True)
    assert abs(out["overall"]["iou"] - 1.0) < 1e-9, "IoU sanity check failed"
    assert out["overall"]["cor"] > 0.95, "Cor sanity check failed (expected close to but not exactly 1.0)"
    assert abs(out["overall"]["cor_lbl"] - 1.0) < 1e-6, "CorLbl sanity check failed"
    print("OK: scorer_local matches official scorer.py on the identity case.")
