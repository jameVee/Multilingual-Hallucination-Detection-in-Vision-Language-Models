"""Decode the model's span JSON into char-offset spans, keeping dual-labeled text.

The shared decoder (solutions/common/vlm_io.map_spans_to_positions) drops any span that
overlaps an already-placed span by more than half, whatever its label. Gold often gives
the same text two labels (annotators disagree: e.g. invention and mischaracterization on
the same words), training targets keep both, and the decoder then throws the second one
away. Fed the gold targets of the labeled train records, it fails to reproduce 3,939 of
8,646 of them exactly.

Here a span only clashes with spans of its OWN label (gold never overlaps two spans of the
same label). `variant` selects how a span is placed; the choice was made on the train
round trip only (`python -m thesis_v2.common.span_io --roundtrip`), never on val:
  "per_label"  per-label clash check, cursor = previous span's end, then the whole text
  "reuse"      + identical text with a different label than the previous span takes the
               previous span's position
  "near_prev"  + a span with a different label is searched from the previous span's
               START first (partially overlapping dual labels)

Raw generations are saved by predict.py, so any decoder can be re-applied without the GPU.
"""
from __future__ import annotations

import argparse
import collections
import json

from solutions.common import vlm_io

VARIANTS = ("per_label", "reuse", "near_prev")
DEFAULT_VARIANT = "near_prev"


def _clash(pos, used):
    return any(i in used for i in range(*pos))


def map_spans_to_positions(response, parsed_spans, variant=DEFAULT_VARIANT):
    result, used = [], collections.defaultdict(set)
    prev = None   # (text, label, (start, end)) of the last placed span
    for sp in parsed_spans:
        text, lab = sp["text"], sp["label"]
        own = used[lab]
        cands = []
        if prev and variant in ("reuse", "near_prev") and prev[0] == text and prev[1] != lab:
            cands.append(prev[2])
        froms = [prev[2][1] if prev else 0]
        if prev and variant == "near_prev" and prev[1] != lab:
            froms.insert(0, prev[2][0])
        cands += [vlm_io.find_span_pos(response, text, search_from=f) for f in froms + [0]]
        pos = next((c for c in cands if c is not None and not _clash(c, own)), None)
        if pos is None:
            continue
        result.append({"start": pos[0], "end": pos[1], "prob": sp["prob"], "label": lab})
        own.update(range(*pos))
        prev = (text, lab, pos)
    return sorted(result, key=lambda x: (x["start"], x["label"]))


def spans_from_raw(response, raw, variant=DEFAULT_VARIANT):
    return map_spans_to_positions(response, vlm_io.parse_model_output(raw), variant)


def roundtrip(records, decode):
    """Feed each record's gold training target through `decode`; count what comes back."""
    lost, total, bad = collections.Counter(), collections.Counter(), 0
    for r in records:
        target = json.dumps(vlm_io.build_target_spans(r), ensure_ascii=False)
        got = {(s["start"], s["end"], s["label"]) for s in decode(r["response"], target)}
        miss = False
        for s in r["labels"]:
            if not r["response"][s["start"]:s["end"]].strip():
                continue
            total[s["label"]] += 1
            if (s["start"], s["end"], s["label"]) not in got:
                lost[s["label"]] += 1
                miss = True
        bad += miss
    return bad, lost, total


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--roundtrip", action="store_true", help="compare decoders on train gold targets")
    args = p.parse_args()
    if not args.roundtrip:
        p.error("pass --roundtrip")
    from solutions.common import data_io
    train = [r for r in data_io.load_jsonl(data_io.TRAIN_FILE) if r["labels"]]
    decoders = {"old (vlm_io)": vlm_io.predict_spans_from_raw_output}
    decoders |= {v: (lambda resp, raw, v=v: spans_from_raw(resp, raw, v)) for v in VARIANTS}
    for name, dec in decoders.items():
        bad, lost, total = roundtrip(train, dec)
        per = "  ".join(f"{l} {lost[l] / total[l]:.1%}" for l in sorted(total))
        print(f"{name:14} records not reproduced {bad}/{len(train)} | spans lost: {per}")


if __name__ == "__main__":
    main()
