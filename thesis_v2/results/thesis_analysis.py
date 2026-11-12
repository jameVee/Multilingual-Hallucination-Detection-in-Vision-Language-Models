"""Descriptive analyses for the thesis write-up (no decision is made here).

Uses existing prediction files only and writes thesis_v2/results/thesis_analysis.json:
  dataset      record counts per split / language, share of records with each label (train)
  ablation     component chain C0 -> 3a v1 -> 3a v2 (F2) -> F3 -> G1 and the probe alone,
               val_select and val_confirm, mean over the seeds that exist
  per_label    per-category Cor+Lbl (official per_label diagnostic) for the same systems
  h4           per (category, language) cell: Cor+Lbl of 3a v2 and 3b v2 (seed means);
               Spearman correlation of the two across the 20 cells
  record_corr  per-record Cor+Lbl correlation between the detector (C0) and the probe (R1 diagnostic)

Usage:  PYTHONPATH=. python thesis_v2/results/thesis_analysis.py
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np
from scipy.stats import spearmanr

from thesis_v2.common import evaluate as ev
from thesis_v2.common.paths import LANGUAGES, RESULTS_DIR, RUNS_DIR
from thesis_v2.common.splits import load_split
from solutions.common import data_io

SEEDS = (42, 123, 2024)
LABELS = data_io.LABEL_ORDER
P1, P3, INT = RUNS_DIR / "part1", RUNS_DIR / "part3", RUNS_DIR / "integration"


def c0(s):
    return P1 / f"qwen3vl8b_seed{s}" / ("val_predictions_dec2.jsonl" if s == 42 else "val_predictions.jsonl")


SYSTEMS = {
    "C0 greedy": {s: c0(s) for s in SEEDS},
    "3a v1 (single-label, samples only)": {42: P3 / "3a_qwen_seed42" / "val_predictions.jsonl"},
    "3b v2 probe alone": {s: P3 / f"crfml_seed{s}" / "val_predictions.jsonl" for s in SEEDS},
    "F2 = 3a v2": {s: P3 / f"3a2_qwen_seed{s}" / "val_predictions.jsonl" for s in SEEDS},
    "F3 = F2 + probe voter": {s: INT / "final" / f"F3_seed{s}_val_predictions.jsonl" for s in SEEDS},
    "G1 = F3 + graded prob": {s: INT / "graded" / f"G1_seed{s}_val_predictions.jsonl" for s in SEEDS},
}


def dataset():
    out = {}
    for name, f in (("train", data_io.TRAIN_FILE), ("val", data_io.VAL_FILE), ("test", data_io.TEST_FILE)):
        recs = data_io.load_jsonl(f)
        out[name] = {"n": len(recs), "per_language": dict(Counter(r["language"] for r in recs))}
        if name != "test":
            lab = [r for r in recs if r.get("labels")]
            out[name]["with_hallucination"] = len(lab)
            out[name]["label_record_share"] = {
                L: round(sum(any(s["label"] == L for s in r["labels"]) for r in recs) / len(recs), 4) for L in LABELS}
            out[name]["mean_spans_per_labeled_record"] = round(np.mean([len(r["labels"]) for r in lab]), 2)
            out[name]["images"] = len({r.get("image_name") for r in recs})
    out["val_select"] = len(load_split("val_select"))
    out["val_confirm"] = len(load_split("val_confirm"))
    return out


def main():
    res = {"dataset": dataset(), "ablation": {}, "per_label": {}}
    rows_cache = {}
    for name, files in SYSTEMS.items():
        res["ablation"][name], res["per_label"][name] = {}, {}
        for split in ("val_select", "val_confirm"):
            prim, cor, iou, per_lab, per_lang = [], [], [], [], []
            for s, f in files.items():
                r, rows = ev.evaluate(data_io.load_jsonl(f), split)
                rows_cache[(name, s, split)] = rows
                prim.append(r["primary"]); cor.append(r["primary_cor"]); iou.append(r["primary_iou"])
                per_lab.append(r["per_label"]); per_lang.append(r["per_language"])
            res["ablation"][name][split] = {
                "seeds": list(files), "cor_lbl": float(np.mean(prim)),
                "cor_lbl_std": float(np.std(prim, ddof=1)) if len(prim) > 1 else None,
                "cor": float(np.mean(cor)), "iou": float(np.mean(iou)),
                "per_language_cor_lbl": {lg: float(np.mean([p[lg]["cor_lbl"] for p in per_lang])) for lg in LANGUAGES}}
            res["per_label"][name][split] = per_lab[0] if len(per_lab) == 1 else per_lab
        print(name, {k: round(v["cor_lbl"], 4) for k, v in res["ablation"][name].items()}, flush=True)

    # H4: per (category, language) cell, Cor+Lbl of 3a v2 vs 3b v2 on val_select (seed means)
    gold = {r["id"]: r for r in data_io.load_jsonl(data_io.VAL_FILE)}
    from solutions.common import scorer_local

    def cell_scores(files):
        cells = {}
        ids = sorted(load_split("val_select"))
        for s, f in files.items():
            pred = {r["id"]: r for r in data_io.load_jsonl(f)}
            for lg in LANGUAGES:
                sub = [i for i in ids if gold[i]["language"] == lg]
                pl = scorer_local.score_predictions([gold[i] for i in sub], [pred[i] for i in sub])["per_label"]
                for L in LABELS:
                    cells.setdefault((L, lg), []).append(pl[L]["cor"])
        return {k: float(np.mean(v)) for k, v in cells.items()}

    a = cell_scores(SYSTEMS["F2 = 3a v2"])
    b = cell_scores(SYSTEMS["3b v2 probe alone"])
    keys = sorted(a)
    rho, pval = spearmanr([a[k] for k in keys], [b[k] for k in keys])
    res["h4"] = {"cells": {f"{L}|{lg}": {"3a_v2": a[(L, lg)], "3b_v2": b[(L, lg)]} for L, lg in keys},
                 "spearman": float(rho), "p": float(pval)}
    print("H4 spearman across 20 cells", round(rho, 3), "p", round(pval, 4), flush=True)

    # R1 diagnostic: per-record Cor+Lbl correlation detector (C0) vs probe, val_select, per seed
    rc = []
    for s in SEEDS:
        x = [r["cor_lbl"] for r in rows_cache[("C0 greedy", s, "val_select")]]
        y = [r["cor_lbl"] for r in rows_cache[("3b v2 probe alone", s, "val_select")]]
        rc.append(float(np.corrcoef(x, y)[0, 1]))
    res["record_corr_C0_vs_probe"] = rc
    print("record-level Pearson C0 vs probe", [round(v, 3) for v in rc], flush=True)

    (RESULTS_DIR / "thesis_analysis.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("->", RESULTS_DIR / "thesis_analysis.json")


if __name__ == "__main__":
    main()
