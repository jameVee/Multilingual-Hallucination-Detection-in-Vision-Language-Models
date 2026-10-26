"""Character-level voting over several span predictions of the same response.

Used by Part 3a (N stochastic decodes of one adapter), integration F3 (3a + the CRF
probe as one extra voter) and F4 (one greedy prediction per Part-1 backbone).

For each character c of the response, with voters i = 1..N (weight w_i, default 1):
    votes(c)    = sum_i w_i * [c inside a span of voter i]
    prob(c)     = (1 / sum_i w_i) * sum_i w_i * flag_i(c) * prob_i(c)
    label(c)    = argmax_label sum_i w_i * flag_i(c, label) * prob_i(c)
Characters with votes >= k are kept; consecutive kept characters with the same label
form one output span, whose prob is the mean prob(c) over its characters.
"""
from __future__ import annotations

import numpy as np

from solutions.common import data_io

LABELS = data_io.LABEL_ORDER


def vote(span_lists, length, k, weights=None):
    n = len(span_lists)
    if n == 0 or length == 0:
        return []
    w = np.ones(n) if weights is None else np.asarray(weights, float)
    votes = np.zeros(length)
    prob_sum = np.zeros(length)
    conf = np.zeros((len(LABELS), length))
    for wi, spans in zip(w, span_lists):
        flag = np.zeros(length, bool)
        p_voter = np.zeros(length)                    # overlapping spans of one voter: max, not sum
        c_voter = np.zeros((len(LABELS), length))
        for s in spans:
            a, b = max(0, int(s["start"])), min(length, int(s["end"]))
            if a >= b or s["label"] not in LABELS:
                continue
            p = float(s.get("prob", 1.0))
            flag[a:b] = True
            p_voter[a:b] = np.maximum(p_voter[a:b], p)
            li = LABELS.index(s["label"])
            c_voter[li, a:b] = np.maximum(c_voter[li, a:b], p)
        votes += wi * flag
        prob_sum += wi * p_voter
        conf += wi * c_voter
    prob = np.clip(prob_sum / w.sum(), 0.0, 1.0)
    keep = votes >= k
    label = conf.argmax(axis=0)
    out, i = [], 0
    while i < length:
        if not keep[i]:
            i += 1
            continue
        j = i
        while j < length and keep[j] and label[j] == label[i]:
            j += 1
        p = float(prob[i:j].mean())
        if p > 0:
            out.append({"start": i, "end": j, "prob": round(p, 4), "label": LABELS[label[i]]})
        i = j
    return out
