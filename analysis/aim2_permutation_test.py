# -*- coding: utf-8 -*-
"""Aim 2: is the heuristic-vs-LLM separability gap (found in
aim2_synthetic_vs_natural_classifier.py / aim2_response_level_classifier.py,
solution 04) actually significant, or could a gap this size arise just from
which ~150-450 examples happened to land in each group?

Only OCR/miscounting/other have LLM-assisted spans at all (llm_inject.py's
TARGET_CATEGORIES) -- invention/mischaracterization are skipped.

Method: label permutation test.
  1. Pool ALL synthetic entries (heuristic + LLM) for a category together --
     feature vectors don't depend on the heuristic/llm label, so they're
     computed once and reused across every permutation (fast).
  2. Observed statistic: AUC(natural vs the true-heuristic subset) minus
     AUC(natural vs the true-LLM subset), both fit+scored on the full
     subset (resubstitution AUC, not 5-fold CV -- this keeps the loop fast
     over hundreds of permutations; the point of this test is ONLY whether
     the heuristic/llm labeling matters, not to re-estimate generalization
     AUC, which the main classifier scripts already report via proper CV).
  3. Null distribution: shuffle which pooled entries are called "heuristic"
     vs "llm" (preserving the true group sizes), recompute the same
     statistic, repeat N times.
  4. Two-sided p-value = fraction of |null diff| >= |observed diff|.

Usage:
    python analysis/aim2_permutation_test.py --level span
    python analysis/aim2_permutation_test.py --level response
"""
import argparse
import random

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from aim2_synthetic_vs_natural_classifier import (
    _REPO_ROOT, _TRAIN_FILE, _SOLUTION_DIRS, RNG,
    load_jsonl, load_ocr_cache, char_trigrams, lexical_features,
    extract_natural_spans, extract_synthetic_spans,
)
from aim2_response_level_classifier import extract_natural_responses, extract_synthetic_responses

CATEGORIES_WITH_LLM = ["OCR", "miscounting", "other"]
SOLUTION = "04"  # the only solution with LLM-assisted spans among 03-06's non-tuned pair
N_PERM = 999


def resubstitution_auc(X_natural, X_group):
    X = np.vstack([X_natural, X_group])
    y = np.array([0] * len(X_natural) + [1] * len(X_group))
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xs, y)
    probs = clf.predict_proba(Xs)[:, 1]
    from sklearn.metrics import roc_auc_score
    return roc_auc_score(y, probs)


def run_test(label, nat_entries, heur_entries, llm_entries, ocr_cache, rng, n_perm):
    RNG.shuffle(nat_entries)
    ref_pool_size = min(300, len(nat_entries) // 3)
    ref_pool = nat_entries[:ref_pool_size]
    ref_trigrams = [char_trigrams(e["text"]) for e in ref_pool]
    eval_natural = nat_entries[ref_pool_size:ref_pool_size + 1500]

    X_natural = np.array([lexical_features(e["text"], ref_trigrams, e["image_name"], ocr_cache)
                           for e in eval_natural])

    pooled = heur_entries + llm_entries
    n_heur = len(heur_entries)
    X_pooled = np.array([lexical_features(e["text"], ref_trigrams, e["image_name"], ocr_cache)
                          for e in pooled])

    def diff_for_assignment(idx_heur, idx_llm):
        auc_h = resubstitution_auc(X_natural, X_pooled[idx_heur])
        auc_l = resubstitution_auc(X_natural, X_pooled[idx_llm])
        return auc_h - auc_l

    all_idx = np.arange(len(pooled))
    observed_idx_heur = all_idx[:n_heur]
    observed_idx_llm = all_idx[n_heur:]
    observed_diff = diff_for_assignment(observed_idx_heur, observed_idx_llm)

    null_diffs = []
    for _ in range(n_perm):
        perm = rng.permutation(all_idx)
        null_diffs.append(diff_for_assignment(perm[:n_heur], perm[n_heur:]))
    null_diffs = np.array(null_diffs)

    p_value = (np.sum(np.abs(null_diffs) >= abs(observed_diff)) + 1) / (n_perm + 1)
    return observed_diff, p_value, null_diffs.std()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--level", choices=["span", "response"], default="span")
    p.add_argument("--n_perm", type=int, default=N_PERM)
    args = p.parse_args()
    n_perm = args.n_perm

    augmented_file = _REPO_ROOT / "solutions" / _SOLUTION_DIRS[SOLUTION] / "train_augmented.jsonl"
    train_records = load_jsonl(_TRAIN_FILE)
    aug_records = load_jsonl(augmented_file)
    ocr_cache = load_ocr_cache(SOLUTION)

    if args.level == "span":
        extract_natural, extract_synth = extract_natural_spans, extract_synthetic_spans
    else:
        extract_natural, extract_synth = extract_natural_responses, extract_synthetic_responses

    natural = extract_natural(train_records)
    heuristic = extract_synth(aug_records, source="heuristic")
    llm = extract_synth(aug_records, source="llm")

    rng = np.random.RandomState(42)
    print(f"Permutation test ({args.level}-level, solution {SOLUTION}, N={n_perm} permutations)\n")
    print(f"{'category':14} {'n_heur':>7} {'n_llm':>7} {'obs AUC gap (heur-llm)':>24} {'null std':>10} {'p-value':>10}")
    for label in CATEGORIES_WITH_LLM:
        heur_entries = heuristic.get(label, [])
        llm_entries = llm.get(label, [])
        nat_entries = natural[label][:]
        if not heur_entries or not llm_entries:
            print(f"{label:14} skipped -- missing heuristic or llm entries")
            continue
        observed_diff, p_value, null_std = run_test(label, nat_entries, heur_entries, llm_entries, ocr_cache, rng, n_perm)
        print(f"{label:14} {len(heur_entries):>7} {len(llm_entries):>7} {observed_diff:>+24.3f} "
              f"{null_std:>10.3f} {p_value:>10.4f}")


if __name__ == "__main__":
    main()
