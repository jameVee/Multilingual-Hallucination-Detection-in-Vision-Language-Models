"""Classifier two-sample test: can text alone tell a method's spans from real spans?

For one category, positives are the synthetic spans and negatives are an equal number
of real train spans per language. A TF-IDF + logistic-regression classifier sees the
span and its surrounding text; it is cross-validated with folds grouped by image, so
a C4 span and its same-image donor are never split across train and test.

AUC 0.5 = indistinguishable from real. A high AUC means the method leaves a text-only
pattern a detector could learn instead of checking the image. The score says nothing
about whether a span is actually wrong for the image, or whether training on it helps;
those need image review and a training run. Train data only; costs nothing.

Usage:
    python -m thesis_v2.part2_synth.c2st
"""
from __future__ import annotations

import json
import random

import numpy as np
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from thesis_v2.common.paths import OLD_SOLUTIONS, PART2_DATA_DIR, RESULTS_DIR
from thesis_v2.part2_synth import retrieval_inject
from solutions.common import data_io

CONTEXT = 40
SEEDS = (0, 1, 2)
MIN_SPANS = 40


def units(records, category):
    """One row per span of `category`: span text, left and right context, language, image."""
    out = []
    for r in records:
        t = r["response"]
        for s in r.get("labels") or []:
            if s["label"] == category and s["end"] > s["start"]:
                out.append({"span": t[s["start"]:s["end"]], "left": t[max(0, s["start"] - CONTEXT):s["start"]],
                            "right": t[s["end"]:s["end"] + CONTEXT], "lang": r["language"],
                            "image": r["image_name"]})
    return out


def auc_once(pos, real, seed):
    rng = random.Random(seed)
    neg = []
    for lg in {u["lang"] for u in pos}:
        pool = [u for u in real if u["lang"] == lg]
        n = sum(u["lang"] == lg for u in pos)
        neg += rng.sample(pool, min(n, len(pool)))
    rows = pos + neg
    y = np.array([1] * len(pos) + [0] * len(neg))
    groups = np.array([u["image"] for u in rows])
    scores = {}
    fields = {"span+context": ("span", "left", "right"), "span only": ("span",), "context only": ("left", "right")}
    for name, keys in fields.items():
        pred = np.zeros(len(rows))
        for tr, te in GroupKFold(n_splits=5).split(rows, y, groups):
            xtr, xte = [], []
            for k in keys:
                vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), min_df=2, sublinear_tf=True)
                xtr.append(vec.fit_transform([rows[i][k] for i in tr]))
                xte.append(vec.transform([rows[i][k] for i in te]))
            clf = LogisticRegression(max_iter=2000, C=1.0)
            clf.fit(hstack(xtr), y[tr])
            pred[te] = clf.decision_function(hstack(xte))
        scores[name] = roc_auc_score(y, pred)
    return scores


def method_records(train):
    methods = {}
    cache = json.loads((OLD_SOLUTIONS / "04_vlm_lora_synthetic_augmented_llm" / "llm_synth_cache.json")
                       .read_text(encoding="utf-8"))
    methods["old LLM prompt"] = [r for v in cache.values() for r in v]
    aug = data_io.load_jsonl(OLD_SOLUTIONS / "05_vlm_lora_synthetic_tuned" / "train_augmented.jsonl")
    methods["old heuristic"] = [r for r in aug if str(r["id"]).startswith("synth") and "llm" not in r["id"]]
    methods["C2 new heuristic"] = data_io.load_jsonl(PART2_DATA_DIR / "C2_candidates.jsonl")
    methods["C4 retrieval"] = data_io.load_jsonl(PART2_DATA_DIR / "C4_candidates.jsonl")
    extra = retrieval_inject.build_candidates(train, ["invention", "mischaracterization"], 7)
    methods["C4 extended"] = [r for v in extra.values() for r in v]
    return methods


def main():
    train = data_io.load_jsonl(data_io.TRAIN_FILE)
    results = {}
    for name, recs in method_records(train).items():
        for cat in data_io.LABEL_ORDER:
            pos = units(recs, cat)
            if len(pos) < MIN_SPANS:
                continue
            # a C4 span and its exact donor would share a test fold and near-identical
            # scores, pulling AUC toward 0.5: the donor spans leave the real side
            copied = {(u["image"], u["span"]) for u in pos}
            real = [u for u in units(train, cat) if (u["image"], u["span"]) not in copied]
            runs = [auc_once(pos, real, s) for s in SEEDS]
            row = {k: round(float(np.mean([r[k] for r in runs])), 3) for k in runs[0]} | {"n_spans": len(pos)}
            results[f"{name} / {cat}"] = row
            print(f"{name:18} {cat:20} n={len(pos):5}  " + "  ".join(f"{k}={row[k]:.3f}" for k in runs[0]), flush=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "part2_c2st.json").write_text(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
