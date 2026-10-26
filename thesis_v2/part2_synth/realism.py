"""How closely does a synthetic set resemble real training data? (Part 2 realism gate)

Measured on train data only -- never on validation. Per category (pooled over
languages for stable statistics): span length (chars), relative span position
(start / response length), prob distribution, OCR in-quote rate, and the
miscounting bare-number rate. Real spans are reweighted to the synthetic set's
language mix: per-cell volumes are fixed by design (e.g. miscounting: 31% of real
spans are zh vs ~19% of synthetic), and zh spans are much shorter (a count is one
character), so an unweighted pooled KS would partly measure language composition.
The gate thresholds are fixed in advance:

    KS(span length) <= 0.15, KS(relative position) <= 0.15,
    TV(prob over {1/3, 2/3, 1}) <= 0.10,
    OCR in-quote rate and miscounting bare-number rate within +-0.10 of real.

Usage:
    python -m thesis_v2.part2_synth.realism C2=path C3=path C4=path
"""
from __future__ import annotations

import json
import re
import sys
from collections import Counter

import numpy as np

from thesis_v2.common.paths import RESULTS_DIR
from solutions.common import data_io

CATEGORIES = ["OCR", "miscounting", "other"]
QUOTE_BEFORE = re.compile(r'["“«「\'*]')
NUMBER_ONLY = re.compile(
    r"^\W*(\d+|zero|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|zéro|un|une|deux|trois|"
    r"quatre|cinq|six|sept|huit|neuf|dix|onze|douze|uno|una|due|tre|quattro|cinque|sei|sette|otto|nove|"
    r"dieci|undici|dodici|[零一二两三四五六七八九十百]+)\W*$", re.I)
THRESH = {"ks_len": 0.15, "ks_pos": 0.15, "tv_prob": 0.10, "rate_delta": 0.10}


def prob_bucket(p):
    return min((1 / 3, 2 / 3, 1.0), key=lambda b: abs(b - float(p)))


def span_stats(records, category, only_synthetic_spans=False):
    """Stats over spans of `category`. For synthetic files every span counts (C4 can
    carry co-occurring real spans of other categories; those are excluded by category)."""
    L, P, probs, inq, num, langs = [], [], Counter(), 0, 0, []
    for r in records:
        t = r["response"]
        for s in r.get("labels", []) or []:
            if s["label"] != category:
                continue
            txt = t[s["start"]:s["end"]]
            L.append(len(txt))
            P.append(s["start"] / max(1, len(t)))
            langs.append(r["language"])
            probs[round(prob_bucket(s["prob"]), 4)] += 1
            inq += bool(QUOTE_BEFORE.search(t[max(0, s["start"] - 15):s["start"]]))
            num += bool(NUMBER_ONLY.match(txt.strip()))
    n = len(L)
    return {"n": n, "len": L, "pos": P, "prob": probs, "lang": langs,
            "in_quote": inq / n if n else None, "number_only": num / n if n else None}


def ks(a, b, wa=None):
    """KS distance; `wa` optionally weights the samples of `a`."""
    a, b = np.asarray(a, float), np.sort(np.asarray(b, float))
    if not len(a) or not len(b):
        return 1.0
    wa = np.ones(len(a)) if wa is None else np.asarray(wa, float)
    order = np.argsort(a, kind="stable")
    a, cw = a[order], np.cumsum(wa[order]) / wa.sum()
    grid = np.concatenate([a, b])
    ia = np.searchsorted(a, grid, side="right")
    fa = np.where(ia > 0, cw[np.maximum(ia - 1, 0)], 0.0)
    fb = np.searchsorted(b, grid, side="right") / len(b)
    return float(np.max(np.abs(fa - fb)))


def lang_weights(real_langs, synth_langs):
    pr, ps = Counter(real_langs), Counter(synth_langs)
    nr, ns = len(real_langs), max(1, len(synth_langs))
    return [(ps[l] / ns) / (pr[l] / nr) for l in real_langs]


def tv(ca, cb):
    keys = set(ca) | set(cb)
    na, nb = sum(ca.values()) or 1, sum(cb.values()) or 1
    return 0.5 * sum(abs(ca.get(k, 0) / na - cb.get(k, 0) / nb) for k in keys)


def compare(real_records, synth_records, category):
    r, s = span_stats(real_records, category), span_stats(synth_records, category)
    w = lang_weights(r["lang"], s["lang"])
    out = {
        "n_real": r["n"], "n_synth": s["n"],
        "ks_len": ks(r["len"], s["len"], w), "ks_pos": ks(r["pos"], s["pos"], w),
        "ks_len_unweighted": ks(r["len"], s["len"]), "ks_pos_unweighted": ks(r["pos"], s["pos"]),
        "tv_prob": tv(r["prob"], s["prob"]),
        "len_median": [float(np.median(r["len"])), float(np.median(s["len"])) if s["n"] else None],
        "pos_median": [float(np.median(r["pos"])), float(np.median(s["pos"])) if s["n"] else None],
    }
    checks = {"ks_len": out["ks_len"] <= THRESH["ks_len"], "ks_pos": out["ks_pos"] <= THRESH["ks_pos"],
              "tv_prob": out["tv_prob"] <= THRESH["tv_prob"]}
    if category == "OCR":
        out["in_quote"] = [r["in_quote"], s["in_quote"]]
        checks["in_quote"] = s["n"] > 0 and abs(r["in_quote"] - s["in_quote"]) <= THRESH["rate_delta"]
    if category == "miscounting":
        out["number_only"] = [r["number_only"], s["number_only"]]
        checks["number_only"] = s["n"] > 0 and abs(r["number_only"] - s["number_only"]) <= THRESH["rate_delta"]
    out["checks"] = checks
    out["pass"] = all(checks.values())
    return out


def matched_order(items, values, real_values, seed, n_bins=5):
    """Order `items` so that any prefix approximates the real distribution of `values`:
    weighted sampling without replacement (Efraimidis-Spirakis), weight = real share /
    candidate share of the item's bin (bins = real quintiles). Train data only."""
    import random as _random
    edges = np.unique(np.quantile(real_values, np.linspace(0, 1, n_bins + 1)[1:-1]))
    real_bins = Counter(int(b) for b in np.searchsorted(edges, real_values, side="right"))
    bins = [int(np.searchsorted(edges, v, side="right")) for v in values]
    share = Counter(bins)
    rng = _random.Random(seed)
    keyed = []
    for it, b in zip(items, bins):
        w = (real_bins[b] / len(real_values)) / (share[b] / len(items))
        keyed.append((rng.random() ** (1 / max(w, 1e-6)), it))
    keyed.sort(key=lambda x: -x[0])
    return [it for _, it in keyed]


def real_lengths(records, category, lang):
    st = span_stats([r for r in records if r["language"] == lang], category)
    return st["len"]


def gate(named_files, save=True):
    real = data_io.load_jsonl(data_io.TRAIN_FILE)
    report = {}
    for name, path in named_files.items():
        synth = data_io.load_jsonl(path)
        report[name] = {c: compare(real, synth, c) for c in CATEGORIES}
        report[name]["pass"] = all(report[name][c]["pass"] for c in CATEGORIES)
    if save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "part2_realism.json").write_text(json.dumps(report, indent=2))
    return report


def print_report(report):
    for name, rep in report.items():
        print(f"== {name}: {'PASS' if rep['pass'] else 'FAIL'}")
        for c in CATEGORIES:
            x = rep[c]
            extra = (f" in_quote real/synth={x['in_quote'][0]:.2f}/{x['in_quote'][1]:.2f}" if "in_quote" in x else "")
            extra += (f" number_only={x['number_only'][0]:.2f}/{x['number_only'][1]:.2f}" if "number_only" in x else "")
            print(f"   {c:12} n={x['n_synth']:4}  KS(len)={x['ks_len']:.3f}  KS(pos)={x['ks_pos']:.3f}  "
                  f"TV(prob)={x['tv_prob']:.3f}  len med {x['len_median'][0]:.0f}/{x['len_median'][1]}"
                  f"  pos med {x['pos_median'][0]:.2f}/{x['pos_median'][1]}{extra}  -> "
                  f"{'ok' if x['pass'] else 'FAIL ' + str([k for k, v in x['checks'].items() if not v])}")


if __name__ == "__main__":
    files = dict(a.split("=", 1) for a in sys.argv[1:])
    print_report(gate(files))
