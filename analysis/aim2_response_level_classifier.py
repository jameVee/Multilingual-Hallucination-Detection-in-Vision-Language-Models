# -*- coding: utf-8 -*-
"""Aim 2, response-level classifier: can a classifier tell a synthetic
hallucinated RESPONSE from a natural hallucinated response of the same
category, using the whole response text rather than just the isolated
injected span (aim2_synthetic_vs_natural_classifier.py does the latter)?

This is the broader question the thesis proposal actually asks -- a span in
isolation loses surrounding context (e.g. an invention injection's
templated sentence sitting next to genuinely-varied real prose might stand
out at the response level even if the span text alone looks plausible).

Reuses the same feature primitives as the span-level script (imported, not
duplicated): length/alpha/digit/space ratios, word count, is_alpha_only,
character-trigram Jaccard vs. a held-out natural reference pool, and (for
OCR) overlap against that image's own cached EasyOCR text.

Data:
  - Natural responses: one response per gold-labeled record in
    data/train.jsonl that has >=1 span of the target category (whole
    response, not the span).
  - Synthetic responses: one response per synth_*-tagged record of that
    category in the chosen solution's train_augmented.jsonl (whole
    corrupted response, i.e. the real seed response plus/with the
    injection already applied).

Usage:
    python analysis/aim2_response_level_classifier.py --solution 05
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

from aim2_synthetic_vs_natural_classifier import (
    _REPO_ROOT, _TRAIN_FILE, _SOLUTION_DIRS, LABELS, RNG,
    load_jsonl, load_ocr_cache, char_trigrams, jaccard, lexical_features, FEATURE_NAMES,
)


def extract_natural_responses(records):
    """label -> list of {"text", "image_name"} -- one entry per record that
    has >=1 gold span of that label, text = the FULL response."""
    out = defaultdict(list)
    for r in records:
        labels_present = {lb["label"] for lb in r.get("labels", []) or []}
        for label in labels_present:
            out[label].append({"text": r["response"], "image_name": r.get("image_name")})
    return out


def extract_synthetic_responses(records, source="all"):
    """label -> list of {"text", "image_name"} -- same id-parsing rules as
    extract_synthetic_spans, but keeps the FULL (already-corrupted) response
    instead of slicing out just the injected span."""
    from aim2_synthetic_vs_natural_classifier import TAG_TO_LABEL
    out = defaultdict(list)
    for r in records:
        if not r["id"].startswith("synth_"):
            continue
        parts = r["id"].split("_")
        tag = parts[1]
        is_llm = tag == "llm"
        if source == "heuristic" and is_llm:
            continue
        if source == "llm" and not is_llm:
            continue
        if is_llm:
            label = parts[2] if len(parts) > 2 and parts[2] in LABELS else None
        else:
            label = TAG_TO_LABEL.get(tag)
        if label is None:
            continue
        out[label].append({"text": r["response"], "image_name": r.get("image_name")})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solution", choices=sorted(_SOLUTION_DIRS), default="05")
    p.add_argument("--source", choices=["all", "heuristic", "llm"], default="all")
    args = p.parse_args()

    augmented_file = _REPO_ROOT / "solutions" / _SOLUTION_DIRS[args.solution] / "train_augmented.jsonl"
    suffix = f"sol{args.solution}" + ("" if args.source == "all" else f"_{args.source}")
    output_file = Path(__file__).resolve().parent / f"aim2_response_level_results_{suffix}.json"

    train_records = load_jsonl(_TRAIN_FILE)
    aug_records = load_jsonl(augmented_file)
    ocr_cache = load_ocr_cache(args.solution)
    print(f"Response-level: solutions/{_SOLUTION_DIRS[args.solution]}/train_augmented.jsonl "
          f"(source={args.source})\n")

    natural = extract_natural_responses(train_records)
    synthetic = extract_synthetic_responses(aug_records, source=args.source)

    print(f"{'category':22} {'n_natural':>10} {'n_synthetic':>12}")
    for label in LABELS:
        print(f"{label:22} {len(natural[label]):>10} {len(synthetic.get(label, [])):>12}")
    print()

    results = {}
    for label in LABELS:
        nat_entries = natural[label][:]
        syn_entries = synthetic.get(label, [])
        if len(syn_entries) < 20 or len(nat_entries) < 40:
            print(f"[{label}] skipped -- not enough examples (nat={len(nat_entries)}, syn={len(syn_entries)})")
            continue

        RNG.shuffle(nat_entries)
        ref_pool_size = min(300, len(nat_entries) // 3)
        ref_pool = nat_entries[:ref_pool_size]
        ref_trigrams = [char_trigrams(e["text"]) for e in ref_pool]

        eval_natural = nat_entries[ref_pool_size:ref_pool_size + 1500]
        eval_synthetic = syn_entries[:1500]

        X, y = [], []
        for e in eval_natural:
            X.append(lexical_features(e["text"], ref_trigrams, e["image_name"], ocr_cache))
            y.append(0)
        for e in eval_synthetic:
            X.append(lexical_features(e["text"], ref_trigrams, e["image_name"], ocr_cache))
            y.append(1)

        X = np.array(X)
        y = np.array(y)

        clf = LogisticRegression(class_weight="balanced", max_iter=2000)
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        aucs = cross_val_score(clf, X, y, cv=cv, scoring="roc_auc")

        from sklearn.preprocessing import StandardScaler
        Xs = StandardScaler().fit_transform(X)
        clf.fit(Xs, y)

        results[label] = {
            "n_natural": len(eval_natural),
            "n_synthetic": len(eval_synthetic),
            "auc_mean": float(aucs.mean()),
            "auc_std": float(aucs.std()),
            "coefs": dict(zip(FEATURE_NAMES, clf.coef_[0].tolist())),
        }
        print(f"[{label:20}] n_nat={len(eval_natural):5} n_syn={len(eval_synthetic):5}  "
              f"AUC = {aucs.mean():.3f} +/- {aucs.std():.3f}")

    print()
    print("Standardized logistic-regression coefficients (which features drive separability):")
    for label, r in results.items():
        print(f"  {label}:")
        for fname, coef in sorted(r["coefs"].items(), key=lambda x: -abs(x[1])):
            print(f"    {fname:22} {coef:+.3f}")

    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved -> {output_file}")


if __name__ == "__main__":
    main()
