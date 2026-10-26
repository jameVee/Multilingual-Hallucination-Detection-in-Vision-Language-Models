"""Frozen split of data/val.jsonl into val_select and val_confirm.

- val_select: the only data any candidate comparison is made on.
- val_confirm: re-checks a choice already made on val_select. Never used to choose.

Grouped by image_name (val has 3,020 records over only 1,473 images, so a
record-level split would put responses about the same image on both sides),
and balanced per language with a greedy assignment. Built once, committed to
git, then only ever loaded.

Usage:
    python -m thesis_v2.common.splits --build   # one-time; refuses to overwrite
    python -m thesis_v2.common.splits           # print a summary of the frozen split
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict

from thesis_v2.common.paths import LANGUAGES, SPLITS_DIR  # noqa: F401  (also sets sys.path)
from solutions.common import data_io

SELECT_FILE = SPLITS_DIR / "val_select_ids.json"
CONFIRM_FILE = SPLITS_DIR / "val_confirm_ids.json"
SPLIT_SEED = 20261026


def build_split(records, seed=SPLIT_SEED):
    groups = defaultdict(list)
    for r in records:
        groups[r["image_name"]].append(r)
    names = sorted(groups)
    random.Random(seed).shuffle(names)

    halves = {"select": Counter(), "confirm": Counter()}
    assign = {}
    for name in names:
        g = Counter(r["language"] for r in groups[name])
        # put the group in whichever half is currently further below 50% for its languages
        def deficit(half):
            other = "confirm" if half == "select" else "select"
            return sum(n * (halves[other][lg] - halves[half][lg]) for lg, n in g.items())
        half = "select" if deficit("select") >= deficit("confirm") else "confirm"
        assign[name] = half
        halves[half].update(g)

    select = sorted(r["id"] for r in records if assign[r["image_name"]] == "select")
    confirm = sorted(r["id"] for r in records if assign[r["image_name"]] == "confirm")
    return select, confirm


def load_split(name):
    """name in {'val_select', 'val_confirm', 'val_full'} -> set of record ids."""
    if name == "val_full":
        return {r["id"] for r in data_io.load_jsonl(data_io.VAL_FILE)}
    path = {"val_select": SELECT_FILE, "val_confirm": CONFIRM_FILE}[name]
    if not path.exists():
        raise FileNotFoundError(f"{path} missing -- build the split once with --build")
    return set(json.loads(path.read_text(encoding="utf-8")))


def summarize():
    records = {r["id"]: r for r in data_io.load_jsonl(data_io.VAL_FILE)}
    sel, conf = load_split("val_select"), load_split("val_confirm")
    assert not (sel & conf), "val_select and val_confirm overlap"
    assert sel | conf == set(records), "split does not cover all of val"
    img_sel = {records[i]["image_name"] for i in sel}
    img_conf = {records[i]["image_name"] for i in conf}
    shared = img_sel & img_conf
    print(f"val_select={len(sel)}  val_confirm={len(conf)}  shared images={len(shared)}")
    for lg in LANGUAGES:
        a = sum(1 for i in sel if records[i]["language"] == lg)
        b = sum(1 for i in conf if records[i]["language"] == lg)
        ca = sum(1 for i in sel if records[i]["language"] == lg and not records[i].get("labels"))
        cb = sum(1 for i in conf if records[i]["language"] == lg and not records[i].get("labels"))
        print(f"  {lg}: select={a} ({a / (a + b):.1%})  confirm={b}   clean: {ca} / {cb}")
    return shared


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--build", action="store_true")
    args = p.parse_args()
    if args.build:
        if SELECT_FILE.exists() or CONFIRM_FILE.exists():
            raise SystemExit("Split already frozen -- refusing to overwrite. Delete deliberately if you mean it.")
        SPLITS_DIR.mkdir(parents=True, exist_ok=True)
        select, confirm = build_split(data_io.load_jsonl(data_io.VAL_FILE))
        SELECT_FILE.write_text(json.dumps(select, indent=0), encoding="utf-8")
        CONFIRM_FILE.write_text(json.dumps(confirm, indent=0), encoding="utf-8")
        print(f"wrote {SELECT_FILE.name} ({len(select)}) and {CONFIRM_FILE.name} ({len(confirm)})")
    summarize()


if __name__ == "__main__":
    main()
