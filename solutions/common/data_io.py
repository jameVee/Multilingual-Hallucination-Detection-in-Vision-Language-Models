"""Shared data loading and image-path utilities for solutions/.

All solution pipelines import from here instead of re-implementing jsonl I/O
or image-path resolution, so behavior (especially the image-filename fallback
logic) stays identical across solutions.
"""
from __future__ import annotations

import difflib
import json
import re
from collections import Counter
from functools import lru_cache
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
IMAGE_DIR = REPO_ROOT / "shroom-vis-images"

TRAIN_FILE = DATA_DIR / "train.jsonl"
VAL_FILE = DATA_DIR / "val.jsonl"
TEST_FILE = DATA_DIR / "test.jsonl"

LABEL_ORDER = ["invention", "mischaracterization", "OCR", "miscounting", "other"]
VALID_LABELS = set(LABEL_ORDER)
VALID_LANGUAGES = {"en", "fr", "it", "zh"}

_ILLEGAL_CHARS_RE = re.compile(r'[:*?"<>|]')


def load_jsonl(path) -> list:
    path = Path(path)
    records = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def write_jsonl(path, records) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


@lru_cache(maxsize=4)
def _image_dir_listing(image_dir: str) -> tuple:
    return tuple(p.name for p in Path(image_dir).iterdir() if p.is_file())


def resolve_image_path(image_name: str, image_dir=IMAGE_DIR):
    """Resolve a dataset `image_name` to an actual file in `image_dir`.

    A handful of filenames in the dataset JSON diverge from the files actually
    saved to disk (illegal filename characters like ':' and '*' were sanitized
    to '_' on write, and a few names picked up mojibake around smart quotes
    during download). Tries an exact match, then a normalized-character match,
    then a fuzzy prefix match against the directory listing, in that order.
    Returns None if nothing close enough is found.
    """
    image_dir = Path(image_dir)
    direct = image_dir / image_name
    if direct.exists():
        return direct

    normalized = _ILLEGAL_CHARS_RE.sub("_", image_name)
    if normalized != image_name:
        candidate = image_dir / normalized
        if candidate.exists():
            return candidate

    listing = _image_dir_listing(str(image_dir))
    matches = difflib.get_close_matches(image_name, listing, n=1, cutoff=0.6)
    if matches:
        return image_dir / matches[0]
    return None


def char_labels_to_vector(labels, text_len: int, label=None):
    """Gold-style accumulation: overlapping spans of the same/target label ADD probabilities.

    Mirrors participant_kit/scorer.py's score_cor reference-side accumulation
    exactly, so synthetic-data sanity checks and training targets stay
    consistent with how the official metric reads gold spans.
    """
    vec = [0.0] * text_len
    for span in labels:
        if label is not None and span["label"] != label:
            continue
        start, end = span["start"], span["end"]
        for idx in range(start, end):
            if 0 <= idx < text_len:
                vec[idx] += span["prob"]
    return vec


def class_counts(records) -> dict:
    span_counts = Counter()
    example_counts = Counter()
    for rec in records:
        seen = set()
        for span in rec.get("labels", []) or []:
            span_counts[span["label"]] += 1
            seen.add(span["label"])
        for lbl in seen:
            example_counts[lbl] += 1
    return {
        "span_counts": dict(span_counts),
        "example_counts": dict(example_counts),
        "n_examples": len(records),
        "n_clean": sum(1 for r in records if not r.get("labels")),
    }


if __name__ == "__main__":
    # Quick manual sanity check: python -m solutions.common.data_io
    for name, path in [("train", TRAIN_FILE), ("val", VAL_FILE), ("test", TEST_FILE)]:
        if not path.exists():
            print(f"{name}: MISSING at {path}")
            continue
        recs = load_jsonl(path)
        print(f"{name}: {len(recs)} rows @ {path}")
        if recs and "labels" in recs[0]:
            print(" ", class_counts(recs))
        missing_images = 0
        for r in recs[:200]:
            if resolve_image_path(r["image_name"]) is None:
                missing_images += 1
        print(f"  unresolved images in first 200 rows: {missing_images}")
