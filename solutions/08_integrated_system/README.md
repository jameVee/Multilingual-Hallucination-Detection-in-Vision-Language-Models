# Solution 08 — Integration and benchmarking (Aim 4)

Combines the best-performing fine-tuned (solutions 02-06) and frozen-model
uncertainty (solution 07) signals into one span-level system, per
`(category, language)` cell, and benchmarks it against the baseline,
submitted systems, and official metrics — per the thesis proposal's Aim 4.

## Status

| Piece | Status |
|---|---|
| Category x language scorer breakdown | Done (`solutions/common/scorer_local.py`'s `per_label_language`) |
| Best-fine-tuned-solution-per-cell table | Done (`compare_cells.py` -> `finetuned_cell_winners.json`) |
| Span assembly (per-token probs -> spans) | Done, unit-tested (`span_assembly.py`) |
| Per-token character offsets | Done — ran on nlp2 (`compute_offsets.py`, all 4 shards) |
| Persist per-cell fitted probe models | Done — ran on nlp2, 20 `.joblib` files + manifest pushed |
| Per-cell threshold tuning + uncertainty-vs-finetuned selection | Code done, unit-tested (`tune_thresholds.py` -> `selection_table.json`) — **needs to run on nlp2** (CPU-only, needs the real feature shards) |
| Uncertainty-only inference script (for new/val/test data) | Not yet started — needs the frozen model + GPU, runs on nlp2 |
| Final merge (fine-tuned + uncertainty per selection table) | Not yet started |

## `tune_thresholds.py` scope caveat

Aim 3's feature shards only cover **labeled** records (`extract_features.py`
filters to `r.get("labels")`) — clean (zero-hallucination) val records were
never extracted. The per-cell comparison this script produces is therefore
restricted to records that have >=1 gold span of the category in question
(matching how `per_label_language` itself is scoped). It does **not**
measure how often the uncertainty system would false-positive on a
genuinely clean response, since none were ever run through it — worth
closing before trusting `selection_table.json` for anything beyond the
cell-vs-cell comparison it's built for.

## Finding so far: fine-tuned-only per-cell selection already beats any single solution

`compare_cells.py` scores all five existing `val_predictions.jsonl` files
against `data/val.jsonl` with the new category x language breakdown, then
picks whichever solution wins each of the 20 cells on the official
per-label Cor+Lbl term. Results in `finetuned_cell_winners.json`.

| | avg Cor+Lbl across 20 cells |
|---|---|
| Solution 02 alone | 0.2200 |
| Solution 03 alone | 0.2238 |
| Solution 04 alone | 0.2224 |
| Solution 05 alone | 0.2312 |
| Solution 06 alone | 0.2314 |
| **Best-per-cell selection** | **0.2589** |

No single solution wins outright: 06 wins 6/20 cells, 05 wins 6/20, 04
wins 4/20, 02 wins 4/20, 03 wins 0/20 (consistent with 03 being the
weakest fine-tuned config on the official test set too — see
`../RESULTS_SUMMARY.md`'s "Test set" section). Per-cell selection beats
even the best single solution (06) by +0.0275 (11.9% relative).

**Caveat, stated plainly**: this number is optimistic by construction —
the winner for each cell was *selected* using val, then *evaluated* on
that same val set. This is exactly the selection-bias pattern that
overturned Aim 1's original "05 beats baseline" claim once multi-seed
variance was checked. Before treating +0.0275 as a real, generalizing
gain, it should be validated against data the selection wasn't fit on
(the closest thing available: the actual test-set submission numbers, but
those only exist per-language, not per-category, so a true apples-to-apples
check isn't currently possible without either (a) a nested/cross-validated
selection on val, splitting it into a selection half and an eval half, or
(b) getting per-category test scores some other way).

## Next step (nlp2, CPU-only, no GPU/model needed)

```bash
cd solutions/08_integrated_system
python tune_thresholds.py \
  --features_dir ../07_uncertainty_quantification/features \
  --models_dir ../07_uncertainty_quantification/models
```

For each cell: applies that cell's persisted model to every held-out val
token in that language, sweeps 9 candidate thresholds (0.1-0.9), assembles
spans at each via `span_assembly.py`, scores each with the real official
scorer, and keeps whichever threshold gets the best Cor+Lbl. Then compares
that best achievable uncertainty-side score against the fine-tuned side's
score for the same cell (`finetuned_cell_winners.json`) and records the
winner, writing `selection_table.json`. Verified against synthetic
data (both the single-signal and ensemble feature-concatenation paths,
and that records with no gold span of the target category are correctly
excluded from a cell's scoring) -- not yet run against the real shards.
