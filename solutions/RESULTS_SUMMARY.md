# Results Summary

Fill in from each solution's `val_scores.json` after running on the VM (see
`VM_RUNBOOK.md`). All numbers come from `solutions/common/scorer_local.py`,
which calls the *official* `participant_kit/scorer.py` functions directly —
directly comparable across all solutions here, but **not** directly
comparable to the numbers in `last_experiment/*/results.csv`, which used
different self-rolled metrics on a different (old, per-language, no
persisted val) data split.

Solutions 03-06 are the four cells of the synthetic-augmentation ablation:
r_max=8 (03 without LLM-assisted injection, 04 with) and the tuned r_max=3
(05 without, 06 with) — see each solution's own README/docstrings for the
full rationale.

## Overall (val)

| Solution | Span IoU | Cor (overall) | Cor (per-label avg) |
|---|---|---|---|
| 01 — Lightweight tagger (real data) | | | |
| 02 — VLM LoRA (real data only) | | | |
| 03 — VLM LoRA (+ synthetic augmented, r_max=8, no LLM) | | | |
| 04 — VLM LoRA (+ synthetic augmented, r_max=8, + LLM) | | | |
| 05 — VLM LoRA (+ synthetic augmented, tuned r_max=3, no LLM) | | | |
| 06 — VLM LoRA (+ synthetic augmented, tuned r_max=3, + LLM) | | | |

## Per-label Cor (diagnostic — is the rare-class push actually working?)

| Solution | invention | mischaracterization | OCR | miscounting | other |
|---|---|---|---|---|---|
| 01 | | | | | |
| 02 | | | | | |
| 03 | | | | | |
| 04 | | | | | |
| 05 | | | | | |
| 06 | | | | | |

## Per-language (overall IoU / Cor)

| Solution | en | fr | it | zh |
|---|---|---|---|---|
| 01 | | | | |
| 02 | | | | |
| 03 | | | | |
| 04 | | | | |
| 05 | | | | |
| 06 | | | | |

## Recommendation

_Fill in once the above is populated._ Solutions 03-06 vs. 02 isolate
whether the synthetic augmentation actually helps (same model/recipe, only
the training data differs) — if one of them wins clearly, especially on the
OCR/miscounting/other columns above, prefer it for the final test-set
submission. If not, 02 (or 01, if it's competitive and much cheaper to
retrain/iterate on) is the safer choice. An ensemble of 01 + one of
03-06 (e.g. averaging per-character probabilities, or using 01's calibrated
probability as a prior/tiebreaker on top of the VLM's spans) is a
reasonable next step if both show real but different signal — not built
here, since the point of this round was to get clean, comparable
single-model numbers first.
