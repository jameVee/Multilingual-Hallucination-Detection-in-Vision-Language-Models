"""Re-apply a span decoder to saved raw generations (predict.py writes them as "raw").

    python -m thesis_v2.common.redecode --pred runs/.../val_predictions.jsonl --decoder old \
        --out runs/.../val_predictions_olddecoder.jsonl
    python -m thesis_v2.common.redecode --pred ... --decoder old --compare runs/.../old_file.jsonl

--compare checks the re-decoded labels against another prediction file record by record
(e.g. that a new greedy run decoded the old way reproduces the original predictions).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from solutions.common import data_io, vlm_io
from thesis_v2.common import span_io

DECODERS = ("old",) + span_io.VARIANTS


def decode(response, raw, decoder):
    if decoder == "old":
        return vlm_io.predict_spans_from_raw_output(response, raw)
    return span_io.spans_from_raw(response, raw, decoder)


def redecode(rows, decoder, split):
    src = data_io.VAL_FILE if split == "val" else data_io.TEST_FILE
    resp = {r["id"]: r["response"] for r in data_io.load_jsonl(src)}
    out = []
    for r in rows:
        if "raw" not in r:
            raise SystemExit(f"{r['id']}: no raw generation saved -- cannot re-decode")
        if isinstance(r["raw"], list):
            out.append({"id": r["id"], "labels": [],
                        "samples": [decode(resp[r["id"]], t, decoder) for t in r["raw"]]})
        else:
            out.append({"id": r["id"], "labels": decode(resp[r["id"]], r["raw"], decoder)})
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pred", required=True)
    p.add_argument("--decoder", choices=DECODERS, required=True)
    p.add_argument("--split", choices=["val", "test"], default=None, help="default: from the file name")
    p.add_argument("--out", default=None)
    p.add_argument("--compare", default=None)
    args = p.parse_args()
    split = args.split or ("test" if Path(args.pred).name.startswith("test") else "val")
    rows = redecode(data_io.load_jsonl(args.pred), args.decoder, split)
    if args.compare:
        other = {r["id"]: r for r in data_io.load_jsonl(args.compare)}
        key = "samples" if "samples" in rows[0] else "labels"
        diff = [r["id"] for r in rows if r[key] != other[r["id"]][key]]
        print(f"{len(rows) - len(diff)}/{len(rows)} records identical to {args.compare}; first differing: {diff[:5]}")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote -> {args.out}")


if __name__ == "__main__":
    main()
