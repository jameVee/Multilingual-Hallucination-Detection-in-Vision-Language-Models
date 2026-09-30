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
| Persist per-cell fitted probe models | Code done (`train_probes.py --save_models_dir`) — **needs to run on nlp2** against the real shards |
| Per-cell threshold tuning | Not yet started |
| Uncertainty-only inference script (for new/val/test data) | Not yet started — needs the frozen model + GPU, runs on nlp2 |
| Final merge (fine-tuned + uncertainty per selection table) | Not yet started |

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
cd solutions/07_uncertainty_quantification
python train_probes.py --features_dir features/ --save_models_dir models/
```

Same command as before, plus `--save_models_dir` — fits and persists the
single best-performing model (an individual signal, or the ensemble if it
wins) for each of the 20 cells as a small `.joblib` file under `models/`,
plus `models/models_manifest.json` summarizing which signal each cell
uses. These are small (a few KB each) and meant to be committed to git,
unlike the raw feature shards.
