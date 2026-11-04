"""Part 3a: sampling-vote detector (training-free consistency).

Input: the sampled prediction files written by part1_models/predict.py with
--num_samples 8 --temperature 0.7 (one per split), i.e. rows
{"id", "labels": [], "samples": [span list, ...]}. A character is kept if at least k of
the N decodes flag it (common/voting.py); its prob is the mean flag x prob over the
decodes, so prob already reflects how consistently the model asserts the span.

Steps (all on val_select / val_confirm -- never on test):
  --diversity   mean pairwise Jaccard between the decodes' flagged-character sets on
                50 val_select records. Pre-registered rule: if > 0.95 the decodes are
                near-identical and sampling is redone once at T=1.0.
  --decide      Decision 3: k in {2, 3, 4} (3 candidates) on val_select, logged in the
                decision log; the chosen k is then scored on val_confirm and the
                outcome attached to the decision. Writes val and test predictions.
                Rerun with --test once test sampling is done: the logged k is reused.

Usage:
    python -m thesis_v2.part3_uncert.sample_vote --val runs/.../val_samples8_T0.7.jsonl --diversity
    python -m thesis_v2.part3_uncert.sample_vote --val ... --test runs/.../test_samples8_T0.7.jsonl \
        --name 3a_qwen_seed42 --decide
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
from pathlib import Path

import numpy as np

from thesis_v2.common import decision_log, evaluate as ev
from thesis_v2.common.paths import REPO_ROOT, RUNS_DIR
from thesis_v2.common.splits import load_split
from thesis_v2.common.voting import vote
from solutions.common import data_io

K_CANDIDATES = (2, 3, 4)
DIVERSITY_LIMIT = 0.95


def load_samples(path):
    path = Path(path) if Path(path).is_absolute() else REPO_ROOT / path
    return {r["id"]: r["samples"] for r in data_io.load_jsonl(path)}


def predictions(samples, records, k):
    out = []
    for r in records:
        spans = samples.get(r["id"])
        if spans is None:
            raise ev.CoverageError(f"no samples for {r['id']}")
        out.append({"id": r["id"], "labels": vote(spans, len(r["response"]), k)})
    return out


def char_set(spans):
    return {c for s in spans for c in range(int(s["start"]), int(s["end"]))}


def diversity(samples, n_records=50, seed=0):
    ids = sorted(load_split("val_select") & set(samples))
    ids = random.Random(seed).sample(ids, min(n_records, len(ids)))
    sims = []
    for rid in ids:
        sets = [char_set(s) for s in samples[rid]]
        for a, b in itertools.combinations(sets, 2):
            sims.append(1.0 if not a and not b else len(a & b) / len(a | b))
    return float(np.mean(sims))


def write(rows, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--val", required=True)
    p.add_argument("--test", default=None)
    p.add_argument("--name", default="3a")
    p.add_argument("--diversity", action="store_true")
    p.add_argument("--decide", action="store_true")
    args = p.parse_args()

    val_samples = load_samples(args.val)
    if args.diversity:
        d = diversity(val_samples)
        print(f"mean pairwise Jaccard of decodes (50 val_select records): {d:.3f} -> "
              f"{'too similar: resample once at T=1.0' if d > DIVERSITY_LIMIT else 'ok'}")
        return
    if not args.decide:
        p.error("pass --diversity or --decide")

    val = data_io.load_jsonl(data_io.VAL_FILE)
    done = decision_log.get_decision(f"D3_{args.name}")
    if done:   # decided earlier (e.g. before test sampling finished): reuse k, never re-decide
        k = int(done["chosen"].split("=")[1])
        val_pred = predictions(val_samples, val, k)
        print(f"D3_{args.name} already decided: k={k}")
    else:
        cand = {}
        for k in K_CANDIDATES:
            res, _ = ev.evaluate(predictions(val_samples, val, k), "val_select")
            cand[f"k={k}"] = res["primary"]
            print(ev.format_result(f"{args.name} k={k}", res))
        chosen = decision_log.record_decision(f"D3_{args.name}",
                                              "3a sampling vote: votes threshold k", cand)
        k = int(chosen.split("=")[1])
        val_pred = predictions(val_samples, val, k)
        conf, _ = ev.evaluate(val_pred, "val_confirm")
        decision_log.record_confirmation(f"D3_{args.name}", {"val_confirm_primary": conf["primary"]})
        print(ev.format_result(f"{args.name} k={k}", conf))
        ev.save_scores(args.name, ev.evaluate(val_pred, "val_select")[0])
        ev.save_scores(args.name, conf)

    out_dir = RUNS_DIR / "part3" / args.name
    write(val_pred, out_dir / "val_predictions.jsonl")
    if args.test:
        test = data_io.load_jsonl(data_io.TEST_FILE)
        write(predictions(load_samples(args.test), test, k), out_dir / "test_predictions.jsonl")
    print(f"chosen k={k} -> {out_dir}")


if __name__ == "__main__":
    main()
