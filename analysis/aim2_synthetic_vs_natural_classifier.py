# -*- coding: utf-8 -*-
"""Aim 2 preliminary work: can a classifier distinguish synthetic from
natural hallucination spans of the same category? (Hypothesis H2)

Extends the paper's span-LENGTH-only comparison (Appendix, Table:
Synthetic vs Natural Span Length) to lexical form: character composition
(alpha/digit ratio, word count, alpha-only flag) plus character-trigram
Jaccard similarity to a held-out reference pool of natural spans of the
same category (never including the instance being scored, avoiding
self-match leakage for the natural class), plus one image-grounded
feature for OCR specifically (see `image_ocr_overlap` below).

Data:
  - Natural spans: data/train.jsonl's own gold `labels` (the "original"
    records already deduplicated -- upsampled duplicates in
    train_augmented.jsonl are the same underlying spans, so using the
    unaugmented file avoids double-counting).
  - Synthetic spans: solutions/<NN>_.../train_augmented.jsonl's synth_*
    -tagged records (--solution selects which of 03/04/05/06; the
    heuristic-injection logic itself is identical across all four, only
    upsampling/LLM differ -- see Aim2_SyntheticVsNatural_Findings.md's
    "Cross-solution check" section for why 03 and 05 give near-identical
    results).

Per category: logistic regression (5-fold CV, class_weight='balanced'),
report mean AUC -- a high AUC means synthetic and natural spans of that
category are lexically distinguishable by these specific features.

See Aim2_SyntheticVsNatural_Findings.md for the full writeup of results.
"""
import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, StratifiedKFold

_REPO_ROOT = Path(__file__).resolve().parent.parent
_TRAIN_FILE = _REPO_ROOT / "data" / "train.jsonl"
_SOLUTION_DIRS = {
    "03": "03_vlm_lora_synthetic_augmented",
    "04": "04_vlm_lora_synthetic_augmented_llm",
    "05": "05_vlm_lora_synthetic_tuned",
    "06": "06_vlm_lora_synthetic_tuned_llm",
}

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

FEATURE_NAMES = ["length", "alpha_ratio", "digit_ratio", "space_ratio", "word_count",
                  "is_alpha_only", "max_trigram_jaccard", "image_ocr_overlap"]


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_ocr_cache(solution_key):
    path = _REPO_ROOT / "solutions" / _SOLUTION_DIRS[solution_key] / "ocr_cache.json"
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def extract_natural_spans(records):
    """label -> list of {"text", "image_name"}, from gold `labels` on
    unaugmented records."""
    spans = defaultdict(list)
    for r in records:
        resp = r["response"]
        for lab in r.get("labels", []):
            text = resp[lab["start"]:lab["end"]]
            if text:
                spans[lab["label"]].append({"text": text, "image_name": r.get("image_name")})
    return spans


def extract_synthetic_spans(records, source="all"):
    """label -> list of {"text", "image_name"}, from synth_*-tagged records only.

    Two id shapes: heuristic injections are `synth_<tag>_<orig_id>` (tag maps
    to a label via TAG_TO_LABEL), LLM-assisted ones are
    `synth_llm_<label>_<orig_id>` (label is already spelled out verbatim as
    the third underscore-part, and IS one of LABELS directly -- an earlier
    version of this function only ever looked at the second part, so every
    LLM-tagged record silently fell through `label is None: continue` and
    was never included in the classifier's synthetic pool).

    `source`: 'all' (default), 'heuristic' (exclude synth_llm_*), or 'llm'
    (synth_llm_* only) -- lets a caller isolate LLM-quality specifically
    instead of it being diluted by the heuristic majority within the same
    category (LLM injection only ever produces ~400/category vs. heuristic's
    ~600-1200/category in solution 04).
    """
    spans = defaultdict(list)
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
        resp = r["response"]
        for lab in r.get("labels", []):
            text = resp[lab["start"]:lab["end"]]
            if text:
                spans[label].append({"text": text, "image_name": r.get("image_name")})
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


def image_ocr_overlap(text, image_name, ocr_cache):
    """Max trigram-Jaccard between `text` and the EasyOCR-detected strings
    for THIS SPECIFIC image (ocr_cache.json, already computed once per
    solution's generate_synthetic.py run) -- i.e. "does this span actually
    match what's really written in the image it's attached to", as opposed
    to `max_trigram_jaccard`'s comparison against a generic reference pool
    of natural spans from OTHER images. This is the one feature in this
    script that uses image-grounded signal rather than text alone (the
    thesis proposal's "image-question context" dimension for Aim 2, in its
    cheapest, no-model form). 0.0 if the image has no cached OCR text at
    all (silent majority of images -- most don't contain visible text).
    """
    if not image_name:
        return 0.0
    texts = ocr_cache.get(image_name)
    if not texts:
        return 0.0
    tri = char_trigrams(text)
    return max((jaccard(tri, char_trigrams(t)) for t in texts), default=0.0)


def lexical_features(text, reference_trigram_sets, image_name=None, ocr_cache=None):
    n = len(text)
    n_alpha = sum(c.isalpha() for c in text)
    n_digit = sum(c.isdigit() for c in text)
    n_space = sum(c.isspace() for c in text)
    word_count = len(text.split())
    # "alpha-only" span-rate diagnostic, made a direct feature instead of
    # only inferred indirectly via max_trigram_jaccard -- matches the exact
    # qualitative check in solutions/03_.../README.md (34% -> 55% vs a 74%
    # real-data target for OCR specifically).
    stripped = text.replace(" ", "")
    is_alpha_only = 1.0 if stripped and stripped.isalpha() else 0.0
    tri = char_trigrams(text)
    max_jac = max((jaccard(tri, ref) for ref in reference_trigram_sets), default=0.0)
    ocr_overlap = image_ocr_overlap(text, image_name, ocr_cache) if ocr_cache is not None else 0.0
    return [
        n,
        n_alpha / n if n else 0.0,
        n_digit / n if n else 0.0,
        n_space / n if n else 0.0,
        word_count,
        is_alpha_only,
        max_jac,
        ocr_overlap,
    ]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--solution", choices=sorted(_SOLUTION_DIRS), default="05",
                    help="Which solution's train_augmented.jsonl to draw synthetic spans from.")
    p.add_argument("--source", choices=["all", "heuristic", "llm"], default="all",
                    help="Restrict synthetic spans to heuristic-only, LLM-only, or both (default).")
    args = p.parse_args()

    augmented_file = _REPO_ROOT / "solutions" / _SOLUTION_DIRS[args.solution] / "train_augmented.jsonl"
    suffix = f"sol{args.solution}" + ("" if args.source == "all" else f"_{args.source}")
    output_file = Path(__file__).resolve().parent / f"aim2_lexical_classifier_results_{suffix}.json"

    train_records = load_jsonl(_TRAIN_FILE)
    aug_records = load_jsonl(augmented_file)
    ocr_cache = load_ocr_cache(args.solution)
    print(f"Using synthetic spans from solutions/{_SOLUTION_DIRS[args.solution]}/train_augmented.jsonl "
          f"(source={args.source}, ocr_cache entries={len(ocr_cache)})\n")

    natural = extract_natural_spans(train_records)
    synthetic = extract_synthetic_spans(aug_records, source=args.source)

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
        # Reference pool: held out, used only to compute similarity features,
        # never used as a classifier instance -- avoids self-match leakage.
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

        # Fit on everything to inspect which features matter (standardized coefficients)
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
