"""R4 (results/preregistration.md): prior-matched synthetic conditions.

Every synthetic / upsampled record carries a hallucination, so adding them raises the share
of hallucinated training records above the real train share (71.6%) and the model flags
more clean records. R4 adds clean duplicates so that the share stays exactly at the real
one; training is held at C0's optimizer steps (train.py --step_budget).

    n_clean = round(n_added * n_clean_real / n_labeled_real)

C4x-b: clean duplicates are the clean base records the C4/C4x transplants were made from
       (same image and prompt seen with and without the hallucination); sampled without
       replacement, seed 0.
C1-b:  C1 duplicates no particular record pairing, so clean duplicates are drawn from all
       clean train records, without replacement, seed 0.

Writes thesis_v2/part2_synth/data/R4_C4xb_clean.jsonl and R4_C1b_clean.jsonl.

Usage:  python -m thesis_v2.part2_synth.build_r4
"""
from __future__ import annotations

import json
import random

from thesis_v2.common.paths import REPO_ROOT
from solutions.common import data_io

DATA = REPO_ROOT / "thesis_v2" / "part2_synth" / "data"


def load(name):
    return data_io.load_jsonl(DATA / f"{name}.jsonl")


def n_clean_for(n_added, n_labeled, n_clean):
    return round(n_added * n_clean / n_labeled)


def duplicates(records, n, tag, seed=0):
    if n > len(records):
        raise ValueError(f"{tag}: need {n} clean records, only {len(records)} available")
    pick = random.Random(seed).sample(sorted(records, key=lambda r: r["id"]), n)
    return [dict(r, id=f"{r['id']}__{tag}", labels=[]) for r in pick]


def share(n_lab, n_all):
    return n_lab / n_all


def write(rows, name):
    with open(DATA / f"{name}.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"-> {DATA / name}.jsonl  ({len(rows)} records)")


def main():
    train = data_io.load_jsonl(data_io.TRAIN_FILE)
    by_id = {r["id"]: r for r in train}
    clean = [r for r in train if not r.get("labels")]
    n_lab, n_cl = len(train) - len(clean), len(clean)
    print(f"real train: {len(train)} records, {n_lab} with a hallucination ({share(n_lab, len(train)):.4f})")

    added = load("C4_candidates") + load("C4x_claims")
    assert all(r.get("labels") for r in added)
    bases = sorted({r["_meta"]["base_id"] for r in added})
    assert all(not by_id[b].get("labels") for b in bases), "a transplant base record is not clean"
    n = n_clean_for(len(added), n_lab, n_cl)
    c4xb = duplicates([by_id[b] for b in bases], n, "r4clean")
    write(c4xb, "R4_C4xb_clean")

    up = load("C1_upsample")
    assert all(r.get("labels") for r in up)
    n1 = n_clean_for(len(up), n_lab, n_cl)
    c1b = duplicates(clean, n1, "r4clean")
    write(c1b, "R4_C1b_clean")

    for name, k_add, k_cl in (("C4x", len(added), 0), ("C4x-b", len(added), len(c4xb)),
                              ("C1", len(up), 0), ("C1-b", len(up), len(c1b))):
        tot = len(train) + k_add + k_cl
        print(f"{name:6} {tot:6} records, hallucinated share {share(n_lab + k_add, tot):.4f}, "
              f"2-epoch steps {(tot // 8) * 2}")


if __name__ == "__main__":
    main()
