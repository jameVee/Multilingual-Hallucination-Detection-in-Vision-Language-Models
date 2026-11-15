"""Per-record judge features for Part 3b, with full-coverage checks.

Sources (npz shards in the old extractor's format):
  OLD_FEATURES_DIR      labeled train/val records + all test records (old Aim 3 run)
  CLEAN_FEATURES_DIR    clean train/val records (extract_features_all.py, the leak fix)

Raw token features are [entropy, nll, lookback (1152), hidden_-1 (4096)]. Holding those
for ~2.4M train tokens would need ~50 GB, so the two wide blocks are reduced by PCA
(hidden -> 256, lookback -> 128) fitted on a 200k-token sample of TRAIN only, and
stored as float16 (~2 GB). The PCA plays the role of the probe's first linear layer.

build() writes one compact file per split to RUNS_DIR/part3/compact/{split}.npz:
  x (tokens x 386, float16), y (label index 0-4, -1 clean), off_start, off_end,
  rec_ptr (record boundaries), record_id, language, skipped_ids
and asserts: features + explicitly skipped ids == every record of the split.

Usage (nlp2, after extract_features_all has run):
    python -m thesis_v2.part3_uncert.features --build
    python -m thesis_v2.part3_uncert.features --build --source gemma3_12b   # R7 second judge

--source qwen (default) is the original probe judge. Any other source is a backbone whose
features came from extract_features_judge2.py (every record of every split, chunked);
it gets its own compact dir, and there every record must be covered.
"""
from __future__ import annotations

import argparse
from collections import defaultdict

import numpy as np

from thesis_v2.common.paths import OLD_FEATURES_DIR, RUNS_DIR
from thesis_v2.part3_uncert.extract_features_all import CLEAN_FEATURES_DIR
from thesis_v2.part3_uncert.extract_features_judge2 import features_dir
from solutions.common import data_io

COMPACT_DIR = RUNS_DIR / "part3" / "compact"
N_HIDDEN, N_LOOKBACK, PCA_SAMPLE, SEED = 256, 128, 200_000, 0
SPLIT_FILES = {"train": data_io.TRAIN_FILE, "val": data_io.VAL_FILE, "test": data_io.TEST_FILE}


def compact_dir(source="qwen"):
    return COMPACT_DIR if source == "qwen" else RUNS_DIR / "part3" / f"compact_{source}"


def shard_files(split, source="qwen"):
    if source != "qwen":
        files = sorted(features_dir(source).glob(f"{split}_chunk*.npz"))
        if not files:
            raise FileNotFoundError(f"no {split} chunks under {features_dir(source)}")
        return files
    files = sorted(OLD_FEATURES_DIR.glob(f"{split}_shard*.npz"))
    if split != "test":
        files += sorted(CLEAN_FEATURES_DIR.glob(f"{split}_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"no {split} shards under {OLD_FEATURES_DIR} / {CLEAN_FEATURES_DIR}")
    return files


def fit_pca(rng, source="qwen"):
    """PCA bases for hidden_-1 and lookback from a token sample of train shards."""
    files = shard_files("train", source)
    per_file = PCA_SAMPLE // len(files) + 1
    hs, ls = [], []
    for f in files:
        z = np.load(f, allow_pickle=True)
        n = z["entropy"].shape[0]
        idx = rng.choice(n, min(n, per_file), replace=False)
        hs.append(z["hidden_-1"][idx].astype(np.float32))
        ls.append(z["lookback"][idx].astype(np.float32))
    out = {}
    for name, block, k in [("hidden", np.concatenate(hs), N_HIDDEN), ("lookback", np.concatenate(ls), N_LOOKBACK)]:
        mean = block.mean(0)
        _, _, vt = np.linalg.svd(block - mean, full_matrices=False)
        out[name] = (mean, vt[:k].T.astype(np.float32))
    return out


def compact(split, pca, source="qwen"):
    hm, hp = pca["hidden"]
    lm, lp = pca["lookback"]
    per_rec = {}
    skipped = set()
    for f in shard_files(split, source):
        z = np.load(f, allow_pickle=True)
        if "skipped_ids" in z.files:
            skipped |= set(z["skipped_ids"].tolist())
        x = np.concatenate([z["entropy"][:, None], z["nll"][:, None],
                            (z["lookback"].astype(np.float32) - lm) @ lp,
                            (z["hidden_-1"].astype(np.float32) - hm) @ hp], axis=1).astype(np.float16)
        rid = z["record_id"]
        bounds = np.flatnonzero(np.r_[True, rid[1:] != rid[:-1], True])
        for a, b in zip(bounds[:-1], bounds[1:]):
            if rid[a] in per_rec:
                raise ValueError(f"record {rid[a]} appears in two shards")
            per_rec[rid[a]] = (x[a:b], z["label_id"][a:b], z["offset_start"][a:b], z["offset_end"][a:b],
                               z["language"][a])
    records = data_io.load_jsonl(SPLIT_FILES[split])
    ids = [r["id"] for r in records]
    missing = set(ids) - set(per_rec) - skipped
    if source != "qwen" and missing:
        raise RuntimeError(f"{source} {split}: {len(missing)} records have neither features nor an explicit "
                           f"skip (e.g. {sorted(missing)[:3]}) -- extraction incomplete")
    # labeled records the old run skipped are not recorded as skipped; they are listed
    # here and get empty predictions -- that is penalized (gold non-empty), not free
    labeled = {r["id"] for r in records if r.get("labels")}
    old_unrecorded = sorted(missing & labeled)
    clean_missing = sorted(set(missing) - set(old_unrecorded))
    if clean_missing:
        raise RuntimeError(f"{split}: {len(clean_missing)} clean records have neither features nor an "
                           f"explicit skip (e.g. {clean_missing[:3]}) -- run extract_features_all first")
    order = [i for i in ids if i in per_rec]
    rec_ptr = np.cumsum([0] + [len(per_rec[i][1]) for i in order])
    np.savez(compact_dir(source) / f"{split}.npz",
             x=np.concatenate([per_rec[i][0] for i in order]),
             y=np.concatenate([per_rec[i][1] for i in order]).astype(np.int8),
             off_start=np.concatenate([per_rec[i][2] for i in order]).astype(np.int32),
             off_end=np.concatenate([per_rec[i][3] for i in order]).astype(np.int32),
             rec_ptr=rec_ptr, record_id=np.array(order, dtype=object),
             language=np.array([per_rec[i][4] for i in order], dtype=object),
             skipped_ids=np.array(sorted(skipped | set(old_unrecorded)), dtype=object))
    print(f"{split}: {len(order)} records with features, {len(skipped)} skipped by the judge, "
          f"{len(old_unrecorded)} labeled records missing from the old run")


def load(split, source="qwen"):
    z = np.load(compact_dir(source) / f"{split}.npz", allow_pickle=True)
    return {k: z[k] for k in z.files}


def build(source="qwen"):
    compact_dir(source).mkdir(parents=True, exist_ok=True)
    pca = fit_pca(np.random.default_rng(SEED), source)
    np.savez(compact_dir(source) / "pca.npz", **{f"{n}_{p}": v for n, (m, b) in pca.items() for p, v in
                                         (("mean", m), ("basis", b))})
    for split in ("train", "val", "test"):
        compact(split, pca, source)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    p.add_argument("--source", default="qwen")
    a = p.parse_args()
    if a.build:
        build(a.source)
