"""Aim 4, step 5: produce the blended TEST predictions, one file per language.

For each (category, language) cell:

    p_final_char = alpha * p_blackbox_char + (1 - alpha) * p_whitebox_char

p_blackbox comes from the cell's fine-tuned winner's test_predictions.{lang}.jsonl.
p_whitebox comes from the cell's Aim 3 probe applied to the TEST shards.
Span assembly uses the cell's (start, continue, min_span) from blend_selection_table.json.

Every hyperparameter here is frozen from the val-side search in tune_blend.py.
Nothing is tuned on test. Scoring is done by the platform, so this script only
writes files and validates their format.

Needs test_shard*.npz from `extract_features.py --split test` (nlp2 only).

Usage:
    python apply_blend_test.py \
        --test_features_dir ../07_uncertainty_quantification/features \
        --models_dir ../07_uncertainty_quantification/models_gbm \
        --blend_table blend_selection_table.json \
        --output_dir test_predictions_blend
    python ../../participant_kit/format_checker.py test_predictions_blend/test_predictions_blend.*.jsonl \
        --reference-file ../../data/test.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
_UQ_DIR = _REPO_ROOT / "solutions" / "07_uncertainty_quantification"
if str(_UQ_DIR) not in sys.path:
    sys.path.insert(0, str(_UQ_DIR))

from solutions.common import data_io
from tune_blend import SOLUTIONS, whitebox_predictions_by_record, char_probs_from_tokens
from span_assembly import merge_positive_tokens_to_spans_hysteresis


def load_test_shards(features_dir):
    """Concatenates test_shard*.npz. train_probes.load_shards deliberately skips
    these (it expects train and val shards to exist), so the test path has its own
    loader with the same concatenation rules."""
    files = sorted(Path(features_dir).glob("test_shard*.npz"))
    if not files:
        raise FileNotFoundError(f"No test_shard*.npz under {features_dir} -- run extract_features.py --split test")
    parts = {}
    for f in files:
        data = np.load(f, allow_pickle=True)
        for key in data.files:
            if key == "lookback_columns":
                parts.setdefault(key, data[key])  # identical across shards
                continue
            parts.setdefault(key, []).append(data[key])
    out = {k: (v if k == "lookback_columns" else np.concatenate(v, axis=0)) for k, v in parts.items()}
    if "offset_start" not in out:
        raise RuntimeError("Test shards have no offsets -- rerun extract_features.py --split test "
                           "with the current version (it saves offsets for every split).")
    print(f"Loaded {len(files)} test shard file(s): {len(out['record_id'])} response tokens, "
          f"{len(set(out['record_id'].tolist()))} records with features")
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--test_features_dir", default=str(_UQ_DIR / "features"))
    p.add_argument("--models_dir", default=str(_UQ_DIR / "models_gbm"))
    p.add_argument("--blend_table", default=str(_THIS_DIR / "blend_selection_table.json"))
    p.add_argument("--output_dir", default=str(_THIS_DIR / "test_predictions_blend"))
    args = p.parse_args()

    with open(args.blend_table, encoding="utf-8") as f:
        table = json.load(f)
    with open(Path(args.models_dir) / "models_manifest.json", encoding="utf-8") as f:
        manifest = json.load(f)

    test_records = data_io.load_jsonl(data_io.TEST_FILE)
    data = load_test_shards(args.test_features_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Every cell must be present -- a partial blend would silently fall back to
    # something else for the missing cells, which is not what was validated.
    for category in data_io.LABEL_ORDER:
        for language in sorted(data_io.VALID_LANGUAGES):
            cell_key = f"{category}/{language}"
            if cell_key not in table or cell_key not in manifest:
                raise RuntimeError(f"{cell_key}: missing from blend table or models manifest -- refusing to write a partial submission.")

    bb_cache = {}

    def load_bb_test(solution_key, language):
        cache_key = (solution_key, language)
        if cache_key not in bb_cache:
            path = _REPO_ROOT / "solutions" / SOLUTIONS[solution_key] / f"test_predictions.{language}.jsonl"
            bb_cache[cache_key] = {r["id"]: r for r in data_io.load_jsonl(path)}
        return bb_cache[cache_key]

    for language in sorted(data_io.VALID_LANGUAGES):
        lang_records = [r for r in test_records if r["language"] == language]
        lang_ids = {r["id"] for r in lang_records}
        text_lens = {r["id"]: len(r["response"]) for r in lang_records}
        char_offsets = {rid: [(i, i + 1) for i in range(n)] for rid, n in text_lens.items()}
        out_spans = {rid: [] for rid in text_lens}
        n_no_features_any = None

        for category in data_io.LABEL_ORDER:
            cell_key = f"{category}/{language}"
            cell = table[cell_key]
            bb_by_id = load_bb_test(cell["finetuned_solution"], language)
            missing_bb = lang_ids - set(bb_by_id)
            if missing_bb:
                raise RuntimeError(f"{cell['finetuned_solution']} test predictions for {language} "
                                   f"are missing {len(missing_bb)} test ids -- refusing to blend.")

            model_entry = joblib.load(Path(args.models_dir) / f"{category}_{language}.joblib")
            wb_by_record = whitebox_predictions_by_record(data, language, model_entry)
            n_no_features = len(lang_ids - set(wb_by_record))
            n_no_features_any = n_no_features
            # Offsets past the end of a response mean the token->char alignment is wrong
            # for that record; char_probs_from_tokens would silently truncate them.
            n_misaligned = sum(1 for rid, (offs, _) in wb_by_record.items()
                               if rid in text_lens and offs and max(e for _, e in offs) > text_lens[rid])
            if n_misaligned:
                print(f"  WARNING {cell_key}: {n_misaligned} records have token offsets past the response end")

            alpha = cell["alpha"]
            for rid in lang_ids:
                text_len = text_lens[rid]
                bb_vec = np.array(data_io.char_labels_to_vector(
                    bb_by_id[rid].get("labels", []), text_len, label=category))
                if rid in wb_by_record:
                    offs, token_probs = wb_by_record[rid]
                    wb_vec = char_probs_from_tokens(offs, token_probs, text_len)
                else:
                    wb_vec = np.zeros(text_len)  # no features -> clean record, same convention as tune_blend
                blended = alpha * bb_vec + (1 - alpha) * wb_vec
                spans = merge_positive_tokens_to_spans_hysteresis(
                    char_offsets[rid], blended.tolist(), category,
                    cell["start_threshold"], cell["continue_threshold"],
                    min_span_tokens=cell["min_span_tokens"])
                out_spans[rid].extend(spans)

        records = []
        for rec in lang_records:
            labels = sorted(out_spans[rec["id"]], key=lambda s: s["start"])
            records.append({"id": rec["id"], "labels": labels})
        out_path = output_dir / f"test_predictions_blend.{language}.jsonl"
        data_io.write_jsonl(out_path, records)
        n_spans = sum(len(r["labels"]) for r in records)
        print(f"  {language}: {len(records)} predictions, {n_spans} spans -> {out_path}")
        print(f"    test records with NO features for this language (clean default): {n_no_features_any}")

    with open(output_dir / "blend_config_used.json", "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)
    print(f"Saved the blend config used -> {output_dir / 'blend_config_used.json'}")


if __name__ == "__main__":
    main()
