"""Integration: the pre-registered combined systems and Decision 4.

Candidates (fixed in advance -- no other combinations are tried):
  F1  Part-2 best condition, greedy decoding
  F2  F1's adapter + 3a sampling vote (k from Decision 3)
  F3  F2 + the 3b CRF probe as ONE extra voter (same k; the probe counts like one decode)
  F4  Part-1 committee: one greedy prediction per backbone, majority vote
      (k = ceil(n/2), not tuned). User-approved optional candidate, reported separately.
With "per_label": true and "greedy_voter": true (R3 / D3_3a2, results/preregistration.md R5)
every vote is per label, and F2/F3 also count F1's greedy prediction as one voter, so F2
reproduces 3a v2 exactly.
Each of F1-F3 is built per seed; Decision 4 uses the mean val_select primary over
seeds, the choice is confirmed on val_confirm, and test predictions are written for
every seed of the chosen system (the user submits them).

Config (JSON), e.g. thesis_v2/integration/config.json:
{
  "name": "final",
  "k": 3,
  "seeds": {
    "42": {"f1":      {"val": "runs/part2/C4_seed42/val_predictions.jsonl", "test": "..."},
           "samples": {"val": ".../val_samples8_T0.7.jsonl", "test": "..."},
           "probe":   {"val": "runs/part3/crf_seed42/val_predictions.jsonl", "test": "..."}},
    "123": {...}, "2024": {...}
  },
  "committee": {"val": ["runs/part1/qwen3vl8b_seed42/val_predictions.jsonl", ...], "test": [...]}
}

Usage:
    python -m thesis_v2.integration.build_candidates --config thesis_v2/integration/config.json
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np

from thesis_v2.common import decision_log, evaluate as ev
from thesis_v2.common.paths import REPO_ROOT, RUNS_DIR
from thesis_v2.common.voting import vote
from solutions.common import data_io


def _load(path):
    path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    return {r["id"]: r for r in data_io.load_jsonl(path)}


def combine(records, sample_file, k, probe_file=None, greedy_file=None, per_label=False):
    samples, probe = _load(sample_file), (_load(probe_file) if probe_file else None)
    greedy = _load(greedy_file) if greedy_file else None
    out = []
    for r in records:
        voters = list(samples[r["id"]]["samples"])
        if greedy is not None:
            voters.append(greedy[r["id"]]["labels"])
        if probe is not None:
            voters.append(probe[r["id"]]["labels"])
        out.append({"id": r["id"], "labels": vote(voters, len(r["response"]), k, per_label=per_label)})
    return out


def committee(records, files, per_label=False):
    preds = [_load(f) for f in files]
    k = math.ceil(len(files) / 2)
    return [{"id": r["id"], "labels": vote([p[r["id"]]["labels"] for p in preds], len(r["response"]), k,
                                           per_label=per_label)}
            for r in records]


def build(cfg, split, seed=None):
    recs = data_io.load_jsonl(data_io.VAL_FILE if split == "val" else data_io.TEST_FILE)
    out = {}
    pl = cfg.get("per_label", False)
    if seed is not None:
        s = cfg["seeds"][seed]
        g = s["f1"][split] if cfg.get("greedy_voter") else None
        out["F1"] = list(_load(s["f1"][split]).values())
        out["F2"] = combine(recs, s["samples"][split], cfg["k"], greedy_file=g, per_label=pl)
        if "probe" in s:
            out["F3"] = combine(recs, s["samples"][split], cfg["k"], s["probe"][split], greedy_file=g, per_label=pl)
    elif cfg.get("committee"):
        out["F4"] = committee(recs, cfg["committee"][split], per_label=pl)
    return out


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    args = p.parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))
    name, seeds = cfg["name"], list(cfg["seeds"])

    per_seed = {}   # candidate -> [val_select primary per seed]
    val_preds = {}
    for seed in seeds:
        for cand, rows in build(cfg, "val", seed).items():
            res, _ = ev.evaluate(rows, "val_select")
            per_seed.setdefault(cand, []).append(res["primary"])
            val_preds[(cand, seed)] = rows
            print(ev.format_result(f"{name} {cand} seed{seed}", res))
    f4 = build(cfg, "val").get("F4")
    if f4:
        res, _ = ev.evaluate(f4, "val_select")
        per_seed["F4"] = [res["primary"]]
        val_preds[("F4", None)] = f4
        print(ev.format_result(f"{name} F4 committee", res))

    cand = {c: float(np.mean(v)) for c, v in per_seed.items()}
    chosen = decision_log.record_decision(f"D4_{name}", "integration F1-F4 (mean over seeds)", cand)
    conf_seeds = [None] if chosen == "F4" else seeds
    confirm = [ev.evaluate(val_preds[(chosen, s)], "val_confirm")[0]["primary"] for s in conf_seeds]
    decision_log.record_confirmation(f"D4_{name}", {"val_confirm_primary_mean": float(np.mean(confirm)),
                                                    "per_seed": confirm})
    print(f"chosen {chosen}: val_select {cand[chosen]:.4f}, val_confirm {np.mean(confirm):.4f}")

    out_dir = RUNS_DIR / "integration" / name
    for s in conf_seeds:
        tag = "committee" if s is None else f"seed{s}"
        write(val_preds[(chosen, s)], out_dir / f"{chosen}_{tag}_val_predictions.jsonl")
        test_rows = build(cfg, "test", s)[chosen]
        write(test_rows, out_dir / f"{chosen}_{tag}_test_predictions.jsonl")
    print(f"-> {out_dir}  (submit each *_test_predictions.jsonl after format_checker)")


if __name__ == "__main__":
    main()
