"""Aim 4: turn per-token positive/negative probe predictions back into
char-offset spans the official scorer can score.

Aim 3's probes predict a probability per response TOKEN, per category.
Nothing in Aim 1-3 turns that into the {"start","end","label","prob"}
span shape every solution 02-06 (and the official scorer) uses. This is
that missing piece -- deliberately small and dependency-free (no sklearn/
numpy needed beyond what's already in the caller) so it can be unit tested
without any model, GPU, or real data.

Usage (as a library, not a script):
    from span_assembly import merge_positive_tokens_to_spans
    spans = merge_positive_tokens_to_spans(offsets, probs, "OCR", threshold=0.6)
"""
from __future__ import annotations


def merge_positive_tokens_to_spans(offsets, probs, label, threshold, prob_agg="mean"):
    """offsets: list of (start, end) char tuples, one per response token, in
    token order. probs: parallel list of P(this token belongs to `label`).
    threshold: a token counts as "positive" iff probs[i] >= threshold.

    Contiguous runs of positive tokens (by TOKEN index, not char adjacency
    -- a run breaks the moment a token's probability drops below threshold,
    even if the next positive token's char offset happens to sit right
    next to it) are merged into one span: start = first token's start, end
    = last token's end, prob = the run's aggregated probability (mean or
    max across its tokens).

    Returns a list of {"start", "end", "label", "prob"} dicts, in order.
    """
    if len(offsets) != len(probs):
        raise ValueError(f"offsets ({len(offsets)}) and probs ({len(probs)}) length mismatch")
    if prob_agg not in ("mean", "max"):
        raise ValueError(f"prob_agg must be 'mean' or 'max', got {prob_agg!r}")

    spans = []
    run_start = None
    for i, p in enumerate(probs):
        is_positive = p >= threshold
        if is_positive and run_start is None:
            run_start = i
        elif not is_positive and run_start is not None:
            spans.append(_finalize(offsets, probs, run_start, i - 1, label, prob_agg))
            run_start = None
    if run_start is not None:
        spans.append(_finalize(offsets, probs, run_start, len(probs) - 1, label, prob_agg))
    return spans


def _finalize(offsets, probs, i0, i1, label, prob_agg):
    run = probs[i0:i1 + 1]
    prob = sum(run) / len(run) if prob_agg == "mean" else max(run)
    return {"start": offsets[i0][0], "end": offsets[i1][1], "label": label, "prob": float(prob)}


def merge_positive_tokens_to_spans_hysteresis(offsets, probs, label, start_threshold,
                                               continue_threshold=None, min_span_tokens=1,
                                               prob_agg="mean"):
    """Two-threshold ("hysteresis") version of merge_positive_tokens_to_spans,
    built to address a specific failure mode found in Aim 4's first real
    evaluation: a single flat threshold makes every token an independent
    yes/no decision, which scatters noise across many short spurious spans
    once the probe is actually exercised against realistic data (not just
    the clean-vs-category tokens it was scored on in Aim 3) -- exactly the
    kind of mistake Cor+Lbl specifically penalizes.

    A span can only START at a token with probability >= `start_threshold`
    (a high bar -- only confident tokens trigger a new span, cutting down
    on spurious starts), but once started, it keeps EXTENDING through any
    token with probability >= `continue_threshold` (a lower bar, so a
    genuine multi-token hallucination doesn't fragment into pieces just
    because one token's probability dips slightly). Same idea as Canny
    edge detection's hysteresis thresholding, applied to 1-D token runs
    instead of 2-D image edges.

    `continue_threshold` defaults to `start_threshold` (recovering the
    original flat-threshold behavior) if not given, and must not be
    higher than `start_threshold` -- continuing has to be at least as easy
    as starting, or the hysteresis logic is backwards.

    `min_span_tokens`: after assembly, spans covering fewer than this many
    tokens are dropped entirely -- a single isolated flipped token is far
    more likely to be noise than a genuine hallucination span.
    """
    if len(offsets) != len(probs):
        raise ValueError(f"offsets ({len(offsets)}) and probs ({len(probs)}) length mismatch")
    if prob_agg not in ("mean", "max"):
        raise ValueError(f"prob_agg must be 'mean' or 'max', got {prob_agg!r}")
    if continue_threshold is None:
        continue_threshold = start_threshold
    if continue_threshold > start_threshold:
        raise ValueError(
            f"continue_threshold ({continue_threshold}) must be <= start_threshold "
            f"({start_threshold}) -- continuing a span must be at least as easy as starting one"
        )

    runs = []  # (i0, i1) token-index ranges, inclusive -- token count is i1-i0+1, known directly
    run_start = None
    for i, p in enumerate(probs):
        if run_start is None:
            if p >= start_threshold:
                run_start = i
        else:
            if p < continue_threshold:
                runs.append((run_start, i - 1))
                run_start = None
    if run_start is not None:
        runs.append((run_start, len(probs) - 1))

    if min_span_tokens > 1:
        runs = [(i0, i1) for (i0, i1) in runs if (i1 - i0 + 1) >= min_span_tokens]
    return [_finalize(offsets, probs, i0, i1, label, prob_agg) for i0, i1 in runs]


def assemble_response_spans(offsets, per_category_probs, thresholds, prob_agg="mean"):
    """Multi-category convenience wrapper: per_category_probs is a dict
    {label: [prob_per_token]}, thresholds is a dict {label: threshold}
    (only categories present in both are assembled). Returns one combined,
    start-sorted span list for the whole response -- the shape a
    prediction record's "labels" field expects."""
    all_spans = []
    for label, probs in per_category_probs.items():
        if label not in thresholds:
            continue
        all_spans.extend(merge_positive_tokens_to_spans(offsets, probs, label, thresholds[label], prob_agg))
    return sorted(all_spans, key=lambda s: s["start"])
