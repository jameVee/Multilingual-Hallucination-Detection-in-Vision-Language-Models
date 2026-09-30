"""Aim 3, Phase 2 evaluation: given the shards produced by
extract_features.py, train per-(category, language) probes on each
uncertainty signal (Experiments 1-3) and report an AUC grid -- this is
what Hypothesis H3 ("at least a subset of (category, language) pairs will
show validation AUC significantly above chance") is actually tested
against, and what gates Experiment 4 (the ensemble).

Runs entirely on CPU -- no GPU needed, unlike extract_features.py. Safe to
run locally after copying the `features/` shards off nlp2, or directly on
nlp2 right after extraction.

**Train/val held-out split, not token-level CV.** An earlier version fit
AND evaluated each probe via 5-fold StratifiedKFold over every token
pooled from every shard file (train.jsonl's and val.jsonl's alike). That
double-counts information: adjacent tokens within the same response are
highly correlated (smooth entropy/hidden-state/attention-ratio across a
sequence), so a token-level CV split can land two tokens from the SAME
response on opposite sides of a fold, letting the classifier partly
"generalize" to a held-out token by recognizing its near-identical
neighbor sitting in the training fold -- inflating the reported AUC. This
version instead fits each probe ONLY on tokens from `train_shard*.npz`
files and evaluates ONLY on tokens from `val_shard*.npz` files -- disjoint
by construction (different source records entirely), mirroring exactly
how Aim 1's LoRA solutions were themselves evaluated. Variance is now
estimated by bootstrap-resampling the held-out val tokens (not CV folds).

Per (category, language) cell:
  - Positive tokens: label_id == that category, language == that language,
    drawn separately from the train-split and val-split token pools.
  - Negative tokens: label_id == CLEAN (-1), same split/language,
    subsampled to at most `--neg_cap` per side for speed/balance.
  - If a side's positives < `--min_train_positive` / `--min_val_positive`:
    falls back to pooling all languages together for that category, on
    that side only -- this is the "shared multilingual fallback probe for
    low-count cells such as OCR" named in the thesis proposal's Phase 2
    description, now applied independently to the train-fit and val-eval
    sides so it never mixes train and val data together.

Signals evaluated separately, then ensembled (concatenated features) only
for cells where >=2 of the 3 individual signals reach `--gate_auc` (default
0.55) on the held-out val AUC -- mirrors Experiment 4's stated gate
("proceeds only if at least two of the three signals individually show
validation AUC clearly above 0.5").

Usage:
    python train_probes.py --features_dir features/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
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


def _source_split_of(filename: str) -> str:
    """'train_shard0of3.npz' -> 'train', 'val_shard0of1.npz' -> 'val'.
    NOTE: deliberately NOT derived from each record's own `id` field --
    the dataset's id scheme embeds the split it was originally annotated
    under, which does not reliably match which of data/train.jsonl vs.
    data/val.jsonl a record currently lives in (observed directly: records
    loaded from data/val.jsonl carrying ids like "train-en-2667"). The
    shard's own filename (set by extract_features.py's --split flag) is
    the only reliable source of truth for this.
    """
    stem = Path(filename).stem
    if stem.startswith("train_shard"):
        return "train"
    if stem.startswith("val_shard"):
        return "val"
    raise ValueError(f"Can't infer split from shard filename: {filename}")


def load_shards(features_dir):
    files = sorted(Path(features_dir).glob("*_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"No shard files found under {features_dir}")
    merged = {}
    hidden_keys = set()
    for f in files:
        split = _source_split_of(f.name)
        data = np.load(f, allow_pickle=True)
        n = data["entropy"].shape[0]
        for key in data.files:
            if key == "lookback_columns":
                merged.setdefault(key, data[key])  # assume identical across shards
                continue
            merged.setdefault(key, []).append(data[key])
            if key.startswith("hidden_"):
                hidden_keys.add(key)
        merged.setdefault("source_split", []).append(np.array([split] * n, dtype=object))
    out = {}
    for key, parts in merged.items():
        if key == "lookback_columns":
            out[key] = parts
        else:
            out[key] = np.concatenate(parts, axis=0)
    n_train = int((out["source_split"] == "train").sum())
    n_val = int((out["source_split"] == "val").sum())
    print(f"Loaded {len(files)} shard file(s): {n_train} train-split tokens, {n_val} val-split tokens, "
          f"hidden-state layers found: {sorted(hidden_keys)}")
    if n_train == 0 or n_val == 0:
        raise RuntimeError(
            f"Need at least one train_shard*.npz AND one val_shard*.npz file under {features_dir} -- "
            f"found train_tokens={n_train}, val_tokens={n_val}. Probes are fit on train, evaluated on val."
        )
    return out


def select_split_indices(data, category_idx, language, split, neg_cap, min_positive, rng):
    """Same idea as the old select_indices, but restricted to one `split`
    ('train' or 'val') throughout -- both the primary language-matched
    attempt and the multilingual-pooling fallback stay within that split,
    so train and val tokens are never mixed."""
    label_id = data["label_id"]
    lang = data["language"]
    split_mask = (data["source_split"] == split)

    def build(lang_mask):
        mask = split_mask & lang_mask
        pos = np.where((label_id == category_idx) & mask)[0]
        neg_pool = np.where((label_id == CLEAN_LABEL_ID) & mask)[0]
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


def auc_holdout(X_train, y_train, X_val, y_val, n_bootstrap=200, seed=42):
    """Fit on train, evaluate on held-out val -- point estimate plus a
    bootstrap std (resampling val rows with replacement) instead of a CV
    std, since there's no CV here: train and val are already a genuine
    held-out split by construction."""
    if len(np.unique(y_train)) < 2 or len(np.unique(y_val)) < 2:
        return None
    scaler = StandardScaler().fit(X_train)
    Xs_train = scaler.transform(X_train)
    Xs_val = scaler.transform(X_val)

    clf = LogisticRegression(class_weight="balanced", max_iter=2000)
    clf.fit(Xs_train, y_train)
    val_probs = clf.predict_proba(Xs_val)[:, 1]
    point_auc = roc_auc_score(y_val, val_probs)

    rng = np.random.RandomState(seed)
    n = len(y_val)
    boot_aucs = []
    for _ in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        y_b = y_val[idx]
        if len(np.unique(y_b)) < 2:
            continue
        boot_aucs.append(roc_auc_score(y_b, val_probs[idx]))
    std = float(np.std(boot_aucs)) if boot_aucs else float("nan")
    return float(point_auc), std, clf, scaler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--features_dir", default=str(_THIS_DIR / "features"))
    p.add_argument("--neg_cap", type=int, default=3000)
    p.add_argument("--min_train_positive", type=int, default=30,
                    help="Below this many train-split positives for a (category, language) cell, "
                         "fall back to pooling all languages' train-split tokens for that category.")
    p.add_argument("--min_val_positive", type=int, default=10,
                    help="Below this many val-split positives for a (category, language) cell, "
                         "fall back to pooling all languages' val-split tokens for that category "
                         "(evaluation only -- never mixed with train).")
    p.add_argument("--gate_auc", type=float, default=0.55)
    p.add_argument("--n_bootstrap", type=int, default=200,
                    help="Bootstrap resamples of the held-out val set used to estimate each AUC's std.")
    p.add_argument("--output", default=str(_THIS_DIR / "aim3_auc_grid.json"))
    p.add_argument("--save_models_dir", default=None,
                    help="If set, persist the single best-performing fitted model per cell (whichever "
                         "of the individual signals or the ensemble has the highest held-out AUC) as a "
                         "joblib file, plus a models_manifest.json describing which signal/feature "
                         "columns each one expects -- needed by Aim 4's threshold-tuning and inference "
                         "steps. Not saved by default (these files are small, but there's no reason to "
                         "write them if this is just being run to check the AUC grid).")
    args = p.parse_args()

    if args.save_models_dir:
        import joblib
        save_dir = Path(args.save_models_dir).resolve()
        save_dir.mkdir(parents=True, exist_ok=True)
        manifest = {}

    data = load_shards(args.features_dir)
    hidden_signals = sorted(k for k in data if k.startswith("hidden_") and data[k].shape[1] > 0)
    signals = ["entropy_nll", "lookback"] + hidden_signals
    rng = np.random.RandomState(42)

    grid = {}
    print(f"\n{'category':22} {'language':8} {'fallback':10} " +
          " ".join(f"{s:>14}" for s in signals) + f" {'ensemble':>14}")
    for cat_idx, category in enumerate(LABEL_ORDER):
        for language in LANGUAGES:
            pos_tr, neg_tr, fb_tr = select_split_indices(
                data, cat_idx, language, "train", args.neg_cap, args.min_train_positive, rng)
            pos_va, neg_va, fb_va = select_split_indices(
                data, cat_idx, language, "val", args.neg_cap, args.min_val_positive, rng)
            cell_key = f"{category}/{language}"
            fallback_str = f"{'T' if fb_tr else '-'}{'V' if fb_va else '-'}"

            if len(pos_tr) < 5 or len(neg_tr) < 5 or len(pos_va) < 5 or len(neg_va) < 5:
                print(f"{category:22} {language:8} {fallback_str:10} insufficient data "
                      f"(train pos={len(pos_tr)} neg={len(neg_tr)}, val pos={len(pos_va)} neg={len(neg_va)})")
                grid[cell_key] = {
                    "status": "insufficient_data",
                    "n_train_positive": int(len(pos_tr)), "n_train_negative": int(len(neg_tr)),
                    "n_val_positive": int(len(pos_va)), "n_val_negative": int(len(neg_va)),
                }
                continue

            idx_tr = np.concatenate([pos_tr, neg_tr])
            y_tr = np.concatenate([np.ones(len(pos_tr)), np.zeros(len(neg_tr))])
            idx_va = np.concatenate([pos_va, neg_va])
            y_va = np.concatenate([np.ones(len(pos_va)), np.zeros(len(neg_va))])

            cell_result = {
                "n_train_positive": int(len(pos_tr)), "n_train_negative": int(len(neg_tr)),
                "n_val_positive": int(len(pos_va)), "n_val_negative": int(len(neg_va)),
                "fallback_train_multilingual": fb_tr, "fallback_val_multilingual": fb_va,
                "signals": {},
            }
            passing = []
            feats_tr_for_ensemble, feats_va_for_ensemble = [], []
            row_str = f"{category:22} {language:8} {fallback_str:10}"
            best = None  # (auc_mean, kind, signal_or_signals, clf, scaler)
            for signal in signals:
                X_tr = features_for(data, idx_tr, signal)
                X_va = features_for(data, idx_va, signal)
                result = auc_holdout(X_tr, y_tr, X_va, y_va, n_bootstrap=args.n_bootstrap)
                if result is None:
                    row_str += f" {'n/a':>14}"
                    continue
                mean_auc, std_auc, clf, scaler = result
                cell_result["signals"][signal] = {"auc_mean": mean_auc, "auc_std": std_auc}
                row_str += f" {mean_auc:.3f}+/-{std_auc:.2f}"
                if best is None or mean_auc > best[0]:
                    best = (mean_auc, "single", signal, clf, scaler)
                if mean_auc >= args.gate_auc:
                    passing.append(signal)
                    feats_tr_for_ensemble.append(X_tr)
                    feats_va_for_ensemble.append(X_va)

            if len(passing) >= 2:
                X_tr_ens = np.concatenate(feats_tr_for_ensemble, axis=1)
                X_va_ens = np.concatenate(feats_va_for_ensemble, axis=1)
                ens_result = auc_holdout(X_tr_ens, y_tr, X_va_ens, y_va, n_bootstrap=args.n_bootstrap)
                if ens_result is not None:
                    ens_auc, ens_std, ens_clf, ens_scaler = ens_result
                    cell_result["ensemble"] = {"auc_mean": ens_auc, "auc_std": ens_std,
                                                "signals_used": passing}
                    row_str += f" {ens_auc:.3f}+/-{ens_std:.2f}"
                    if best is None or ens_auc > best[0]:
                        best = (ens_auc, "ensemble", passing, ens_clf, ens_scaler)
            else:
                row_str += f" {'gated off':>14}"

            cell_result["status"] = "ok"
            grid[cell_key] = cell_result
            print(row_str)

            if args.save_models_dir and best is not None:
                best_auc, kind, signal_spec, clf, scaler = best
                model_path = save_dir / f"{category}_{language}.joblib"
                joblib.dump({"clf": clf, "scaler": scaler, "kind": kind, "signal": signal_spec}, model_path)
                manifest[cell_key] = {
                    "model_path": str(model_path.relative_to(_THIS_DIR)),
                    "kind": kind, "signal": signal_spec, "auc_mean": best_auc,
                }

    n_cells_tested = sum(1 for v in grid.values() if v["status"] == "ok")
    n_cells_any_signal = sum(1 for v in grid.values() if v["status"] == "ok"
                              and any(s["auc_mean"] >= args.gate_auc for s in v["signals"].values()))
    print(f"\n{n_cells_any_signal}/{n_cells_tested} testable (category, language) cells have "
          f">=1 signal with held-out AUC >= {args.gate_auc} -- this is the evidence relevant to H3.")
    print("(fallback column: T = train side pooled multilingual, V = val side pooled multilingual, "
          "- = language-specific data was sufficient)")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(grid, f, indent=2)
    print(f"Saved -> {args.output}")

    if args.save_models_dir:
        manifest_path = save_dir / "models_manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print(f"Saved {len(manifest)} fitted model(s) -> {save_dir} (manifest: {manifest_path})")


if __name__ == "__main__":
    main()
