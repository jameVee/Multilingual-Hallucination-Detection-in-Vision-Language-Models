"""Shared paths for thesis_v2. Importing this module also puts the repo root on
sys.path, so `from solutions.common import data_io, scorer_local` works from any
script in thesis_v2/ regardless of the current working directory.

Old experiment code under solutions/ is imported read-only and never edited.
"""
from __future__ import annotations

import sys
from pathlib import Path

THESIS_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = THESIS_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SPLITS_DIR = THESIS_DIR / "splits"
RESULTS_DIR = THESIS_DIR / "results"
RUNS_DIR = THESIS_DIR / "runs"          # gitignored: checkpoints, predictions, shards, caches
PART2_DATA_DIR = THESIS_DIR / "part2_synth" / "data"   # committed: synthetic additions per condition

# Old-experiment assets reused read-only
OLD_SOLUTIONS = REPO_ROOT / "solutions"
OCR_CACHE = OLD_SOLUTIONS / "05_vlm_lora_synthetic_tuned" / "ocr_cache.json"
OLD_FEATURES_DIR = OLD_SOLUTIONS / "07_uncertainty_quantification" / "features"

LANGUAGES = ["en", "fr", "it", "zh"]
SEEDS = [42, 123, 2024]
