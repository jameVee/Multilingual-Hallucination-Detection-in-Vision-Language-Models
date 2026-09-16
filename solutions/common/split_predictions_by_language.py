"""Splits a combined-language predictions JSONL file into one file per language.

The SHROOM submission format requires exactly one file per language (see
participant_kit/format_checker.py: "Each file must contain exactly one
language"), but data/test.jsonl mixes all 4 languages together in one file,
so a naive predict.py run over the whole thing produces one combined
test_predictions.jsonl that the checker (and the submission site) rejects.
This splits that combined file using the language already embedded in each
record's id (<split>-<language>-<item>, e.g. "test-en-428").

Usage:
    python split_predictions_by_language.py test_predictions.jsonl
    python split_predictions_by_language.py test_predictions.jsonl --output_dir ./split
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from solutions.common import data_io


def split_by_language(input_path, output_dir=None, prefix=None):
    input_path = Path(input_path)
    output_dir = Path(output_dir) if output_dir else input_path.parent
    prefix = prefix or input_path.stem

    records = data_io.load_jsonl(input_path)
    by_lang = defaultdict(list)
    skipped = 0
    for record in records:
        parts = str(record.get("id", "")).split("-")
        if len(parts) != 3:
            skipped += 1
            continue
        lang = parts[1].lower()
        by_lang[lang].append(record)

    written = []
    for lang, recs in sorted(by_lang.items()):
        out_path = output_dir / f"{prefix}.{lang}.jsonl"
        data_io.write_jsonl(out_path, recs)
        written.append(out_path)
        print(f"  {lang}: {len(recs)} predictions -> {out_path}")
    if skipped:
        print(f"  WARNING: skipped {skipped} record(s) with an unparseable id (expected <split>-<language>-<item>)")
    return written


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("input", help="Combined predictions JSONL file (all languages mixed together)")
    p.add_argument("--output_dir", default=None, help="Defaults to the input file's own directory")
    p.add_argument("--prefix", default=None, help="Output filename prefix; defaults to the input filename stem")
    args = p.parse_args()
    split_by_language(args.input, args.output_dir, args.prefix)


if __name__ == "__main__":
    main()
