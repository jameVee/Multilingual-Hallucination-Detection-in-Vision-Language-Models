"""R7 (results/preregistration.md): a second frozen judge (Gemma-3-12B) for the probe.

Candidates, all per seed on G1's voters (per-label vote, k from the config, graded prob):
  H0  G1 as is: 8 samples + greedy + Qwen probe                (M = 10)
  H1  G1 + Gemma probe as an 11th voter                         (M = 11)
  H2  G1 with the Gemma probe in place of the Qwen probe        (M = 10)
Decision D7_judge2 = highest mean val_select Cor+Lbl over the 3 seeds, logged once.
The chosen candidate is checked against H0 with the "beats baseline" rule (H0 seed std,
val_confirm paired bootstrap on records averaged over seeds). Test files are written
only if H1 or H2 is chosen.

Diagnostics (select nothing): Gemma probe alone; record-level Pearson of per-record
Cor+Lbl on val_select between C0 and each probe and between the probes (H5); Spearman
over the 20 category-language cells between F2 and the Gemma probe (as H4).
Everything goes to results/r7_judge2.json.

Usage:  python -m thesis_v2.integration.r7 --config thesis_v2/integration/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from thesis_v2.common import decision_log, evaluate as ev
from thesis_v2.common.paths import LANGUAGES, RESULTS_DIR, RUNS_DIR
from thesis_v2.common.splits import load_split
from thesis_v2.common.voting import vote
from thesis_v2.integration.build_candidates import _load, write
from solutions.common import data_io, scorer_local

JUDGE2 = "gemma3_12b"
VARIANTS = ("H0", "H1", "H2")
LABELS = data_io.LABEL_ORDER


def judge2_probe(seed, split):
    return RUNS_DIR / "part3" / f"crfml_{JUDGE2}_seed{seed}" / f"{split}_predictions.jsonl"


def build(cfg, seed, split, variant):
    recs = data_io.load_jsonl(data_io.VAL_FILE if split == "val" else data_io.TEST_FILE)
    s = cfg["seeds"][seed]
    samples, greedy = _load(s["samples"][split]), _load(s["f1"][split])
    qwen_probe, gemma_probe = _load(s["probe"][split]), _load(judge2_probe(seed, split))
    out = []
    for r in recs:
        voters = list(samples[r["id"]]["samples"]) + [greedy[r["id"]]["labels"]]
        if variant in ("H0", "H1"):
            voters.append(qwen_probe[r["id"]]["labels"])
        if variant in ("H1", "H2"):
            voters.append(gemma_probe[r["id"]]["labels"])
        out.append({"id": r["id"], "labels": vote(voters, len(r["response"]), cfg["k"],
                                                  per_label=True, graded="prob")})
    return out


def seed_mean_rows(rows_per_seed):
    """Per-record scores averaged over seeds (the bootstrap unit of the protocol)."""
    base = rows_per_seed[0]
    return [{"id": r["id"], "language": r["language"],
             **{m: float(np.mean([rs[i][m] for rs in rows_per_seed])) for m in ("cor_lbl", "cor", "iou")}}
            for i, r in enumerate(base)]


def cell_cor(files):
    gold = {r["id"]: r for r in data_io.load_jsonl(data_io.VAL_FILE)}
    ids = sorted(load_split("val_select"))
    cells = {}
    for f in files:
        pred = {r["id"]: r for r in data_io.load_jsonl(f)}
        for lg in LANGUAGES:
            sub = [i for i in ids if gold[i]["language"] == lg]
            pl = scorer_local.score_predictions([gold[i] for i in sub], [pred[i] for i in sub])["per_label"]
            for L in LABELS:
                cells.setdefault(f"{L}|{lg}", []).append(pl[L]["cor"])
    return {k: float(np.mean(v)) for k, v in cells.items()}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = list(cfg["seeds"])
    res = {"seeds": seeds}

    val = {(v, s): build(cfg, s, "val", v) for v in VARIANTS for s in seeds}
    for s in seeds:   # H0 must be G1 exactly
        g1 = _load(RUNS_DIR / "integration" / "graded" / f"G1_seed{s}_val_predictions.jsonl")
        assert all(r["labels"] == g1[r["id"]]["labels"] for r in val[("H0", s)]), f"H0 != G1 (seed {s})"
    rows ={(v, s, h): ev.evaluate(val[(v, s)], h)[1] for v in VARIANTS for s in seeds
            for h in ("val_select", "val_confirm")}
    score = {k: ev.aggregate(r)["primary"] for k, r in rows.items()}
    for v in VARIANTS:
        print(v, {h: [round(score[(v, s, h)], 4) for s in seeds] for h in ("val_select", "val_confirm")}, flush=True)
    res["scores"] = {v: {h: {s: score[(v, s, h)] for s in seeds} for h in ("val_select", "val_confirm")}
                     for v in VARIANTS}

    sel = {v: float(np.mean([score[(v, s, "val_select")] for s in seeds])) for v in VARIANTS}
    chosen = decision_log.record_decision(
        "D7_judge2", "R7: second frozen judge (Gemma-3-12B) probe on G1 (mean over 3 seeds)", sel)
    conf = [score[(chosen, s, "val_confirm")] for s in seeds]
    decision_log.record_confirmation("D7_judge2", {"val_confirm_primary_mean": float(np.mean(conf)),
                                                   "per_seed": conf})
    res["chosen"] = chosen

    # "beats baseline" check of H1 and H2 against H0 (reported for both; only the chosen one can be claimed)
    h0_std = float(np.std([score[("H0", s, "val_select")] for s in seeds], ddof=1))
    res["vs_H0"] = {"H0_seed_std": h0_std}
    for v in ("H1", "H2"):
        d = {}
        for h in ("val_select", "val_confirm"):
            a = seed_mean_rows([rows[("H0", s, h)] for s in seeds])
            b = seed_mean_rows([rows[(v, s, h)] for s in seeds])
            d[h] = dict(zip(("diff", "ci_low", "ci_high"), ev.paired_bootstrap(a, b)))
        d["per_seed_val_select"] = [score[(v, s, "val_select")] - score[("H0", s, "val_select")] for s in seeds]
        d["beats"] = bool(d["val_select"]["diff"] > h0_std and d["val_confirm"]["diff"] > 0
                          and d["val_confirm"]["ci_low"] > 0)
        res["vs_H0"][v] = d
        print(v, "vs H0", {h: tuple(round(x, 4) for x in d[h].values()) for h in ("val_select", "val_confirm")},
              "beats" if d["beats"] else "does not beat", flush=True)

    # diagnostics
    gemma = {s: _load(judge2_probe(s, "val")) for s in seeds}
    alone = {h: [ev.evaluate(list(gemma[s].values()), h)[0]["primary"] for s in seeds]
             for h in ("val_select", "val_confirm")}
    res["gemma_probe_alone"] = alone
    corr = {"C0_vs_qwen_probe": [], "C0_vs_gemma_probe": [], "qwen_probe_vs_gemma_probe": []}
    for s in seeds:
        c0 = ev.evaluate(list(_load(cfg["seeds"][s]["f1"]["val"]).values()), "val_select")[1]
        qp = ev.evaluate(list(_load(cfg["seeds"][s]["probe"]["val"]).values()), "val_select")[1]
        gp = ev.evaluate(list(gemma[s].values()), "val_select")[1]
        x, y, z = ([r["cor_lbl"] for r in t] for t in (c0, qp, gp))
        corr["C0_vs_qwen_probe"].append(float(np.corrcoef(x, y)[0, 1]))
        corr["C0_vs_gemma_probe"].append(float(np.corrcoef(x, z)[0, 1]))
        corr["qwen_probe_vs_gemma_probe"].append(float(np.corrcoef(y, z)[0, 1]))
    res["record_corr"] = corr
    f2 = cell_cor([RUNS_DIR / "part3" / f"3a2_qwen_seed{s}" / "val_predictions.jsonl" for s in seeds])
    gc = cell_cor([judge2_probe(s, "val") for s in seeds])
    keys = sorted(f2)
    rho, pval = spearmanr([f2[k] for k in keys], [gc[k] for k in keys])
    res["h4_gemma"] = {"spearman": float(rho), "p": float(pval), "cells": {k: [f2[k], gc[k]] for k in keys}}
    print("gemma probe alone", {h: [round(x, 4) for x in v] for h, v in alone.items()})
    print("record corr", {k: [round(x, 3) for x in v] for k, v in corr.items()})
    print("H4 (F2 vs gemma probe) spearman", round(rho, 3))

    out_dir = RUNS_DIR / "integration" / "r7"
    for v in VARIANTS:
        for s in seeds:
            write(val[(v, s)], out_dir / f"{v}_seed{s}_val_predictions.jsonl")
    if chosen != "H0":
        for s in seeds:
            write(build(cfg, s, "test", chosen), out_dir / f"{chosen}_seed{s}_test_predictions.jsonl")
    (RESULTS_DIR / "r7_judge2.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"chosen {chosen}; -> {out_dir}, {RESULTS_DIR / 'r7_judge2.json'}")


if __name__ == "__main__":
    main()
