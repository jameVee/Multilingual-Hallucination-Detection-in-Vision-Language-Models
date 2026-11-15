"""How much does the image file name tell about the hallucinations? (thesis discussion;
descriptive only, nothing is selected or trained on it)

Image names come in three forms:
  descriptive     <object>_<Variety_name>_<original file stem>, e.g. lion_Lion_cub_Lion-cub-38326918.jpg,
                  otter_albino_otter_d221e745.jpg: the dataset builders chose an object with an
                  unusual property and the name states it
  count-prefixed  <n>_<flickr id>_..._o.jpg
  flickr          <flickr id>_<secret>_o.jpg
For descriptive names the variety words are the fields between the object word and the
last field (the original file stem), minus the object word itself.

Writes thesis_v2/results/filename_hint.json.
Usage:  PYTHONPATH=. python thesis_v2/results/filename_hint.py
"""
from __future__ import annotations

import json
import re

from thesis_v2.common.paths import RESULTS_DIR
from solutions.common import data_io


def kind(name):
    parts = name.rsplit(".", 1)[0].split("_")
    if re.fullmatch(r"[A-Za-z][A-Za-z-]*", parts[0]) and len(parts) >= 3:
        return "descriptive"
    if parts[-1] == "o" and len(parts) == 4:
        return "count-prefixed"
    return "flickr"


def variety_words(name):
    parts = name.rsplit(".", 1)[0].split("_")
    obj = parts[0].lower()
    return {w.lower() for p in parts[1:-1] for w in re.split(r"[-\s]", p)
            if re.fullmatch(r"[A-Za-z]{3,}", w) and w.lower() not in (obj, obj + "s")}


def has_label(r, label):
    return any(l["label"] == label for l in r.get("labels") or [])


def main():
    tv = data_io.load_jsonl(data_io.TRAIN_FILE) + data_io.load_jsonl(data_io.VAL_FILE)
    te = data_io.load_jsonl(data_io.TEST_FILE)
    out = {"kinds": {}}
    for k in ("descriptive", "count-prefixed", "flickr"):
        sub = [r for r in tv if kind(r["image_name"]) == k]
        out["kinds"][k] = {
            "share_train_val": len(sub) / len(tv),
            "share_test": sum(kind(r["image_name"]) == k for r in te) / len(te),
            "hallucinated": sum(bool(r.get("labels")) for r in sub) / len(sub),
            **{lab: sum(has_label(r, lab) for r in sub) / len(sub) for lab in data_io.LABEL_ORDER},
        }
    en = [r for r in tv if r["language"] == "en" and kind(r["image_name"]) == "descriptive"
          and variety_words(r["image_name"])]
    mention = [r for r in en if any(re.search(rf"\b{w}", r["response"].lower()) for w in variety_words(r["image_name"]))]
    lab = [r for r in en if r.get("labels")]
    in_span = [r for r in lab if any(re.search(rf"\b{w}", " ".join(r["response"][l["start"]:l["end"]].lower()
                                                               for l in r["labels"]))
                                     for w in variety_words(r["image_name"]))]
    out["english_descriptive"] = {"n": len(en), "answer_mentions_variety_word": len(mention) / len(en),
                                  "n_hallucinated": len(lab), "variety_word_inside_gold_span": len(in_span) / len(lab)}
    (RESULTS_DIR / "filename_hint.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for k, v in out["kinds"].items():
        print(f"{k:15s} train+val {v['share_train_val']:.3f} test {v['share_test']:.3f} "
              f"halluc {v['hallucinated']:.3f} mischar {v['mischaracterization']:.3f}")
    print(out["english_descriptive"])


if __name__ == "__main__":
    main()
