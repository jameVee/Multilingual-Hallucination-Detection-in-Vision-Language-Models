# -*- coding: utf-8 -*-
"""Aim 2 preliminary work: can a classifier distinguish synthetic from
natural hallucination spans of the same category? (Hypothesis H2)

Extends the paper's span-LENGTH-only comparison (Appendix, Table:
Synthetic vs Natural Span Length) to lexical form: character composition
(alpha/digit ratio, word count) plus character-trigram Jaccard similarity
to a held-out reference pool of natural spans of the same category (never
including the instance being scored, avoiding self-match leakage for the
natural class).

Data:
  - Natural spans: data/train.jsonl's own gold `labels` (the "original"
    records already deduplicated -- upsampled duplicates in
    train_augmented.jsonl are the same underlying spans, so using the
    unaugmented file avoids double-counting).
  - Synthetic spans: solutions/05_vlm_lora_synthetic_tuned/
    train_augmented.jsonl's synth_* -tagged records (tuned r_max=3 config
    -- arbitrary choice among 03-06 since the injection heuristics
    themselves are identical across all four, only upsampling/LLM differ).

Per category: logistic regression (5-fold CV, class_weight='balanced'),
report mean AUC -- a high AUC means synthetic and natural spans of that
category are lexically distinguishable by these specific features.

Result on a first run (see Aim2_SyntheticVsNatural_Findings.md for the
full writeup): all 5 categories separate well above chance, but NOT in
the direction a naive reading of H2 predicts -- OCR is actually the
*least* separable by these features (AUC 0.726, tied with invention at
0.727), despite being the one category augmentation robustly hurts
(RESULTS_SUMMARY.md's multi-seed analysis). mischaracterization (0.859)
and other (0.828) are the most separable, driven almost entirely by the
`length` feature -- consistent with the paper's own Appendix span-length
table showing those two have the largest natural-vs-synthetic length gaps,
while OCR/miscounting are comparatively length-matched. For OCR
specifically, `max_trigram_jaccard` (character-level lexical form, not
length) is the dominant signal instead -- see the findings doc for what
this does and doesn't establish about H2.
"""
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_FILE = _REPO_ROOT / "data" / "train.jsonl"
_AUGMENTED_FILE = _REPO_ROOT / "solutions" / "05_vlm_lora_synthetic_tuned" / "train_augmented.jsonl"
_OUTPUT_FILE = Path(__file__).resolve().parent / "aim2_lexical_classifier_results.json"

RNG = random.Random(42)

TAG_TO_LABEL = {
    "ocrgrnd": "OCR",
    "cntword": "miscounting",
    "cntdigit": "miscounting",
    "attrswap": "mischaracterization",
    "stateswap": "other",
    "reployop": "other",
    "objadd": "invention",
}

LABELS = ["invention", "mischaracterization", "OCR", "miscounting", "other"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def extract_natural_spans(records):
    """label -> list of span texts, from gold `labels` on unaugmented records."""
    spans = defaultdict(list)
    for r in records:
        resp = r["response"]
        for lab in r.get("labels", []):
            text = resp[lab["start"]:lab["end"]]
            if text:
                spans[lab["label"]].append(text)
    return spans


def extract_synthetic_spans(records):
    """label -> list of span texts, from synth_*-tagged records only."""
    spans = defaultdict(list)
    for r in records:
        if not r["id"].startswith("synth_"):
            continue
        tag = r["id"].split("_")[1]
        label = TAG_TO_LABEL.get(tag)
        if label is None:
            continue
        resp = r["response"]
        for lab in r.get("labels", []):
            text = resp[lab["start"]:lab["end"]]
            if text:
                spans[label].append(text)
    return spans


def char_trigrams(s):
    s = s.lower()
    if len(s) < 3:
        return {s} if s else set()
    return {s[i:i + 3] for i in range(len(s) - 2)}


def jaccard(a, b):
    if not a and not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def lexical_features(text, reference_trigram_sets):
    n = len(text)
    n_alpha = sum(c.isalpha() for c in text)
    n_digit = sum(c.isdigit() for c in text)
    n_space = sum(c.isspace() for c in text)
    word_count = len(text.split())
    tri = char_trigrams(text)
    max_jac = max((jaccard(tri, ref) for ref in reference_trigram_sets), default=0.0)
    return [
        n,
        n_alpha / n if n else 0.0,
        n_digit / n if n else 0.0,
        n_space / n if n else 0.0,
        word_count,
        max_jac,
    ]


def main():
    train_records = load_jsonl(_TRAIN_FILE)
    aug_records = load_jsonl(_AUGMENTED_FILE)

    natural = extract_natural_spans(train_records)
    synthetic = extract_synthetic_spans(aug_records)

    print(f"{'category':22} {'n_natural':>10} {'n_synthetic':>12}")
    for label in LABELS:
        print(f"{label:22} {len(natural[label]):>10} {len(synthetic.get(label, [])):>12}")
    print()

    feature_names = ["length", "alpha_ratio", "digit_ratio", "space_ratio", "word_count", "max_trigram_jaccard"]
    results = {}

    for label in LABELS:
        nat_texts = natural[label][:]
        syn_texts = synthetic.get(label, [])
        if len(syn_texts) < 20 or len(nat_texts) < 40:
            print(f"[{label}] skipped -- not enough examples (nat={len(nat_texts)}, syn={len(syn_texts)})")
            continue

        RNG.shuffle(nat_texts)
        # Reference pool: held out, used only to compute similarity features,
        # never used as a classifier instance -- avoids self-match leakage.
        ref_pool_size = min(300, len(nat_texts) // 3)
        ref_pool = nat_texts[:ref_pool_size]
        ref_trigrams = [char_trigrams(t) for t in ref_pool]

        eval_natural = nat_texts[ref_pool_size:ref_pool_size + 1500]
        eval_synthetic = syn_texts[:1500]

        X, y = [], []
        for t in eval_natural:
            X.append(lexical_features(t, ref_trigrams))
            y.append(0)
        for t in eval_synthetic:
            X.append(lexical_features(t, ref_trigrams))
            y.append(1)

        X = np.array(X)
        y = np.array(y)

        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        # Fit on everything to inspect which features matter (standardized coefficients)
        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        clf.fit(Xs, y)

        results[label] = {
            "n_natural": len(eval_natural),
            "n_synthetic": len(eval_synthetic),
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std()),
            "coefs": dict(zip(feature_names, clf.coef_[0].tolist())),
        }
        print(f"[{label:20}] n_nat={len(eval_natural):5} n_syn={len(eval_synthetic):5}  "
              f"AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

    print()
    print("Standardized logistic-regression coefficients (which features drive separability):")
    for label, r in results.items():
        print(f"  {label}:")
        for fname, coef in sorted(r["coefs"].items(), key=lambda x: -abs(x[1])):
            print(f"    {fname:22} {coef:+.3f}")

    with open(_OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {_OUTPUT_FILE}")


if __name__ == "__main__":
    main()
