"""R6 (results/preregistration.md): graded per-character probabilities on top of F3.

Same voters as F3 (8 samples + greedy + crfml probe, per-label vote, k from the config),
so the kept characters -- and IoU -- are identical. Only the prob values change:
  G0  F3 as is (span prob = mean over the span)
  G1  graded="prob"       per-character prob_L(c), rounded to 2 decimals
  G2  graded="vote_frac"  per-character votes_L(c) / N
Decision D6_graded = highest mean val_select over the seeds. val_confirm is attached,
and test files are written for every seed of the chosen variant.

Usage:  python -m thesis_v2.integration.graded --config thesis_v2/integration/config.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from thesis_v2.common import decision_log, evaluate as ev
from thesis_v2.common.paths import RUNS_DIR
from thesis_v2.common.voting import vote
from thesis_v2.integration.build_candidates import _load, write
from solutions.common import data_io

VARIANTS = {"G0": None, "G1": "prob", "G2": "vote_frac"}


def build(cfg, seed, split, graded):
    recs = data_io.load_jsonl(data_io.VAL_FILE if split == "val" else data_io.TEST_FILE)
    s = cfg["seeds"][seed]
    samples, greedy, probe = _load(s["samples"][split]), _load(s["f1"][split]), _load(s["probe"][split])
    out = []
    for r in recs:
        voters = list(samples[r["id"]]["samples"]) + [greedy[r["id"]]["labels"], probe[r["id"]]["labels"]]
        out.append({"id": r["id"], "labels": vote(voters, len(r["response"]), cfg["k"],
                                                  per_label=True, graded=graded)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    seeds = list(cfg["seeds"])

    val = {(g, s): build(cfg, s, "val", mode) for g, mode in VARIANTS.items() for s in seeds}
    sel = {}
    for g in VARIANTS:
        scores = []
        for s in seeds:
            res, _ = ev.evaluate(val[(g, s)], "val_select")
            scores.append(res["primary"])
            print(ev.format_result(f"R6 {g} seed{s}", res), flush=True)
        sel[g] = float(np.mean(scores))
    chosen = decision_log.record_decision(
        "D6_graded", "R6 post-hoc: graded per-character probabilities on F3 (mean over 3 seeds)", sel)
    conf = [ev.evaluate(val[(chosen, s)], "val_confirm")[0]["primary"] for s in seeds]
    decision_log.record_confirmation("D6_graded", {"val_confirm_primary_mean": float(np.mean(conf)),
                                                   "per_seed": conf})
    print(f"chosen {chosen}: val_select {sel[chosen]:.4f}, val_confirm {np.mean(conf):.4f}", flush=True)

    out_dir = RUNS_DIR / "integration" / "graded"
    for g in VARIANTS:
        for s in seeds:
            write(val[(g, s)], out_dir / f"{g}_seed{s}_val_predictions.jsonl")
    if chosen != "G0":
        for s in seeds:
            write(build(cfg, s, "test", VARIANTS[chosen]), out_dir / f"{chosen}_seed{s}_test_predictions.jsonl")
    print(f"-> {out_dir}")


if __name__ == "__main__":
    main()
