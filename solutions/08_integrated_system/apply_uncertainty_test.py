"""Aim 3/4: produce a pure UNCERTAINTY-ONLY test submission (no fine-tuned side).

Fills a real gap in the project's test-set comparison: every submission so
far is either a fine-tuned solution alone, or a combination that includes a
fine-tuned side (hard selection, the alpha blend). There is no standalone
test-set number for "what does the Aim 3 probe alone score, with the official
scorer, on real test data" -- only its validation-side AUC and the per-cell
"uncertainty_only" Cor+Lbl figure inside blend_selection_table*.json.

This reuses that already-tuned per-cell config (alpha=0.0's own
start_threshold / continue_threshold / min_span_tokens, chosen by
tune_blend.py's stage-2 search alongside everything else) rather than
re-tuning anything here -- nothing in this script searches hyperparameters.

Usage:
    python apply_uncertainty_test.py \
        --test_features_dir ../07_uncertainty_quantification/features \
        --models_dir ../07_uncertainty_quantification/models_clean_and_other_gbm_w1 \
        --selection_table blend_selection_table_models_clean_and_other_gbm_w1.json \
        --output_dir test_predictions_uncertainty_only
    python ../../participant_kit/format_checker.py \
        test_predictions_uncertainty_only/test_predictions_uncertainty_only.*.jsonl \
        --reference-file ../../data/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_UQ_DIR = _REPO_ROOT / "solutions" / "07_uncertainty_quantification"
if str(_UQ_DIR) not in sys.path:
    sys.path.insert(0, str(_UQ_DIR))

from solutions.common import data_io
from tune_blend import whitebox_predictions_by_record, char_probs_from_tokens
from apply_blend_test import load_test_shards
from span_assembly import merge_positive_tokens_to_spans_hysteresis


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_features_dir", default=str(_UQ_DIR / "features"))
    p.add_argument("--models_dir", required=True,
                    help="Folder with models_manifest.json and {category}_{language}.joblib, "
                         "e.g. ../07_uncertainty_quantification/models_clean_and_other_gbm_w1")
    p.add_argument("--selection_table", required=True,
                    help="A blend_selection_table*.json -- only its 'uncertainty_only' "
                         "sub-block per cell is used; the fine-tuned side is ignored entirely.")
    p.add_argument("--output_dir", default=str(_THIS_DIR / "test_predictions_uncertainty_only"))
    args = p.parse_args()

    with open(args.selection_table, encoding="utf-8") as f:
        table = json.load(f)
    with open(Path(args.models_dir) / "models_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    test_records = data_io.load_jsonl(data_io.TEST_FILE)
    data = load_test_shards(args.test_features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for category in data_io.LABEL_ORDER:
        for language in sorted(data_io.VALID_LANGUAGES):
            cell_key = f"{category}/{language}"
            if cell_key not in table or not table[cell_key].get("uncertainty_only"):
                raise RuntimeError(f"{cell_key}: no tuned uncertainty_only config in {args.selection_table} "
                                   f"-- refusing to write a partial submission.")
            if cell_key not in manifest:
                raise RuntimeError(f"{cell_key}: missing from {args.models_dir}/models_manifest.json")

    for language in sorted(data_io.VALID_LANGUAGES):
        lang_records = [r for r in test_records if r["language"] == language]
        lang_ids = {r["id"] for r in lang_records}
        text_lens = {r["id"]: len(r["response"]) for r in lang_records}
        char_offsets = {rid: [(i, i + 1) for i in range(n)] for rid, n in text_lens.items()}
        out_spans = {rid: [] for rid in text_lens}
        n_no_features = None

        for category in data_io.LABEL_ORDER:
            cell_key = f"{category}/{language}"
            cfg = table[cell_key]["uncertainty_only"]
            model_entry = joblib.load(Path(args.models_dir) / f"{category}_{language}.joblib")
            wb_by_record = whitebox_predictions_by_record(data, language, model_entry)
            n_no_features = len(lang_ids - set(wb_by_record))

            for rid in lang_ids:
                if rid not in wb_by_record:
                    continue  # no features -> clean default, same convention as apply_blend_test.py
                offs, token_probs = wb_by_record[rid]
                # Char-level assembly, matching exactly how tune_blend.py validated this
                # config (alpha=0.0 case): char_probs_from_tokens expands the per-token
                # probabilities to one value per character, then hysteresis merges over
                # [(i, i+1), ...] char offsets. min_span_tokens therefore means "minimum
                # span length in CHARACTERS" here, not sub-word tokens -- using the raw
                # token-level offsets instead would silently apply a different, unvalidated
                # threshold under the same numbers.
                n = text_lens[rid]
                char_probs = char_probs_from_tokens(offs, token_probs, n)
                spans = merge_positive_tokens_to_spans_hysteresis(
                    char_offsets[rid], char_probs.tolist(), category,
                    cfg["start_threshold"], cfg["continue_threshold"],
                    min_span_tokens=cfg["min_span_tokens"])
                out_spans[rid].extend(spans)

        records = []
        for rec in lang_records:
            labels = sorted(out_spans[rec["id"]], key=lambda s: s["start"])
            records.append({"id": rec["id"], "labels": labels})
        out_path = output_dir / f"test_predictions_uncertainty_only.{language}.jsonl"
        data_io.write_jsonl(out_path, records)
        n_spans = sum(len(r["labels"]) for r in records)
        print(f"  {language}: {len(records)} predictions, {n_spans} spans -> {out_path}")
        print(f"    test records with NO features for this language (clean default): {n_no_features}")

    with open(output_dir / "selection_table_used.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"Saved the selection table used -> {output_dir / 'selection_table_used.json'}")


if __name__ == "__main__":
    main()
