"""Aim 3, Phase 2 evaluation: given the shards produced by
extract_features.py, train per-(category, language) probes on each
uncertainty signal (Experiments 1-3) and report an AUC grid -- this is
what Hypothesis H3 ("at least a subset of (category, language) pairs will
show validation AUC significantly above chance") is actually tested
against, and what gates Experiment 4 (the ensemble).

Runs entirely on CPU -- no GPU needed, unlike extract_features.py. Safe to
run locally after copying the `features/` shards off nlp2, or directly on
nlp2 right after extraction.

Per (category, language) cell:
  - Positive tokens: label_id == that category, language == that language.
  - Negative tokens: label_id == CLEAN (-1), language == that language,
    subsampled to at most `--neg_cap` for speed/balance.
  - If positives < `--min_positive` (default 30): falls back to a pooled
    multilingual probe for that category (all languages' positives/
    negatives together) instead of skipping the cell -- this is the
    "shared multilingual fallback probe for low-count cells such as OCR"
    named in the thesis proposal's Phase 2 description. The grid marks
    which cells used the fallback.

Signals evaluated separately, then ensembled (concatenated features) only
for cells where >=2 of the 3 individual signals reach `--gate_auc` (default
0.55) -- mirrors Experiment 4's stated gate ("proceeds only if at least two
of the three signals individually show validation AUC clearly above 0.5").

Usage:
    python train_probes.py --features_dir features/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing import StandardScaler

import sys
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from solutions.common import data_io

LABEL_ORDER = data_io.LABEL_ORDER
LANGUAGES = sorted(data_io.VALID_LANGUAGES)
CLEAN_LABEL_ID = -1


def load_shards(features_dir):
    files = sorted(Path(features_dir).glob("*_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"No shard files found under {features_dir}")
    merged = {}
    hidden_keys = set()
    for f in files:
        data = np.load(f, allow_pickle=True)
        for key in data.files:
            if key == "lookback_columns":
                merged.setdefault(key, data[key])  # assume identical across shards
                continue
            merged.setdefault(key, []).append(data[key])
            if key.startswith("hidden_"):
                hidden_keys.add(key)
    out = {}
    for key, parts in merged.items():
        if key == "lookback_columns":
            out[key] = parts
        else:
            out[key] = np.concatenate(parts, axis=0)
    print(f"Loaded {len(files)} shard file(s), {out['entropy'].shape[0]} total response tokens, "
          f"hidden-state layers found: {sorted(hidden_keys)}")
    return out


def auc_5fold(X, y, seed=42):
    if len(np.unique(y)) < 2 or len(y) < 20:
        return None
    Xs = StandardScaler().fit_transform(X)
    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    try:
        scores = cross_val_score(clf, Xs, y, cv=cv, scoring="roc_auc")
    except ValueError:
        return None
    return float(scores.mean()), float(scores.std())


def select_indices(data, category_idx, language, neg_cap, min_positive, rng):
    """Returns (pos_idx, neg_idx, used_fallback). Tries the (category,
    language) cell first; falls back to pooling all languages for that
    category if positives are too scarce."""
    label_id = data["label_id"]
    lang = data["language"]

    def build(lang_mask):
        pos = np.where((label_id == category_idx) & lang_mask)[0]
        neg_pool = np.where((label_id == CLEAN_LABEL_ID) & lang_mask)[0]
        if len(neg_pool) > neg_cap:
            neg_pool = rng.choice(neg_pool, size=neg_cap, replace=False)
        return pos, neg_pool

    lang_mask = (lang == language)
    pos, neg = build(lang_mask)
    if len(pos) >= min_positive:
        return pos, neg, False
    all_mask = np.ones(len(label_id), dtype=bool)
    pos, neg = build(all_mask)
    return pos, neg, True


def features_for(data, idx, signal):
    if signal == "entropy_nll":
        return np.stack([data["entropy"][idx], data["nll"][idx]], axis=1)
    if signal == "lookback":
        return data["lookback"][idx]
    if signal.startswith("hidden_"):
        return data[signal][idx]
    raise ValueError(signal)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=str(_THIS_DIR / "features"))
    p.add_argument("--neg_cap", type=int, default=3000)
    p.add_argument("--min_positive", type=int, default=30)
    p.add_argument("--gate_auc", type=float, default=0.55)
    p.add_argument("--output", default=str(_THIS_DIR / "aim3_auc_grid.json"))
    args = p.parse_args()

    data = load_shards(args.features_dir)
    hidden_signals = sorted(k for k in data if k.startswith("hidden_") and data[k].shape[1] > 0)
    signals = ["entropy_nll", "lookback"] + hidden_signals
    rng = np.random.RandomState(42)

    grid = {}
    print(f"\n{'category':22} {'language':8} {'fallback':8} " +
          " ".join(f"{s:>14}" for s in signals) + f" {'ensemble':>14}")
    for cat_idx, category in enumerate(LABEL_ORDER):
        for language in LANGUAGES:
            pos, neg, fallback = select_indices(data, cat_idx, language, args.neg_cap, args.min_positive, rng)
            cell_key = f"{category}/{language}"
            if len(pos) < 5 or len(neg) < 5:
                print(f"{category:22} {language:8} {'--':8} insufficient data (pos={len(pos)}, neg={len(neg)})")
                grid[cell_key] = {"status": "insufficient_data", "n_positive": int(len(pos)), "n_negative": int(len(neg))}
                continue

            idx = np.concatenate([pos, neg])
            y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])

            cell_result = {"n_positive": int(len(pos)), "n_negative": int(len(neg)), "fallback_multilingual": fallback,
                           "signals": {}}
            passing = []
            feats_for_ensemble = []
            row_str = f"{category:22} {language:8} {'yes' if fallback else 'no':8}"
            for signal in signals:
                X = features_for(data, idx, signal)
                result = auc_5fold(X, y)
                if result is None:
                    row_str += f" {'n/a':>14}"
                    continue
                mean_auc, std_auc = result
                cell_result["signals"][signal] = {"auc_mean": mean_auc, "auc_std": std_auc}
                row_str += f" {mean_auc:.3f}+/-{std_auc:.2f}"
                if mean_auc >= args.gate_auc:
                    passing.append(signal)
                    feats_for_ensemble.append(X)

            if len(passing) >= 2:
                X_ens = np.concatenate(feats_for_ensemble, axis=1)
                ens_result = auc_5fold(X_ens, y)
                if ens_result is not None:
                    cell_result["ensemble"] = {"auc_mean": ens_result[0], "auc_std": ens_result[1],
                                                "signals_used": passing}
                    row_str += f" {ens_result[0]:.3f}+/-{ens_result[1]:.2f}"
            else:
                row_str += f" {'gated off':>14}"

            cell_result["status"] = "ok"
            grid[cell_key] = cell_result
            print(row_str)

    n_cells_tested = sum(1 for v in grid.values() if v["status"] == "ok")
    n_cells_any_signal = sum(1 for v in grid.values() if v["status"] == "ok"
                              and any(s["auc_mean"] >= args.gate_auc for s in v["signals"].values()))
    print(f"\n{n_cells_any_signal}/{n_cells_tested} testable (category, language) cells have "
          f">=1 signal with AUC >= {args.gate_auc} -- this is the evidence relevant to H3.")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)
    print(f"Saved -> {args.output}")


if __name__ == "__main__":
    main()
