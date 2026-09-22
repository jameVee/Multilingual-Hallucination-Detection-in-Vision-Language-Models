# Results Summary

Fill in from each solution's `val_scores.json` after running on the server
(see `NLP2_RUNBOOK.md`). All numbers come from `solutions/common/scorer_local.py`,
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

`Cor (per-label avg)` is the official `cor_lbl` (Cor+Lbl) metric from each
solution's `val_scores.json` — the paper's primary metric, since it
penalizes spurious extra labels that raw `Cor` doesn't.

| Solution | Span IoU | Cor (overall) | Cor (per-label avg) |
|---|---|---|---|
| 01 — Lightweight tagger (real data) | | | |
| 02 — VLM LoRA (real data only) | 0.4380 | 0.4712 | 0.4172 |
| 03 — VLM LoRA (+ synthetic augmented, r_max=8, no LLM) | 0.4543 | 0.4852 | 0.4117 |
| 04 — VLM LoRA (+ synthetic augmented, r_max=8, + LLM) | 0.4532 | 0.4880 | 0.4113 |
| **05 — VLM LoRA (+ synthetic augmented, tuned r_max=3, no LLM)** | **0.4582** | 0.4881 | **0.4215** |
| 06 — VLM LoRA (+ synthetic augmented, tuned r_max=3, + LLM) | 0.4556 | 0.4872 | 0.4154 |

## Per-label Cor (diagnostic — is the rare-class push actually working?)

| Solution | invention | mischaracterization | OCR | miscounting | other |
|---|---|---|---|---|---|
| 01 | | | | | |
| 02 | 0.2037 | 0.1184 | 0.3579 | 0.3920 | 0.0274 |
| 03 | 0.2031 | 0.1292 | 0.3200 | 0.4017 | 0.0674 |
| 04 | 0.2051 | 0.1456 | 0.2738 | 0.3980 | 0.0920 |
| 05 | 0.2211 | 0.1379 | 0.3083 | 0.4215 | 0.0680 |
| 06 | 0.2137 | 0.1338 | 0.2929 | 0.4046 | 0.1141 |

## Per-language (overall IoU / Cor)

| Solution | en | fr | it | zh |
|---|---|---|---|---|
| 01 | | | | |
| 02 | 0.3703 / 0.4140 | 0.4281 / 0.4556 | 0.4417 / 0.4768 | 0.5073 / 0.5340 |
| 03 | 0.3876 / 0.4241 | 0.4439 / 0.4670 | 0.4549 / 0.4877 | 0.5260 / 0.5572 |
| 04 | 0.3956 / 0.4432 | 0.4441 / 0.4651 | 0.4549 / 0.4963 | 0.5141 / 0.5439 |
| 05 | 0.3961 / 0.4365 | 0.4584 / 0.4773 | 0.4520 / 0.4873 | 0.5221 / 0.5473 |
| 06 | 0.4136 / 0.4512 | 0.4484 / 0.4742 | 0.4402 / 0.4777 | 0.5169 / 0.5425 |

## Recommendation

**Solution 05 (tuned augmentation, r_max=3, no LLM) is the standout**: best
IoU, best Cor+Lbl, and essentially tied for best raw Cor, all
simultaneously — the only config that clearly beats the real-data baseline
(02) on Cor+Lbl (0.4215 vs 0.4172) rather than trading it off for a raw-Cor
gain. This matches the paper's own reported numbers for this exact
checkpoint (IoU=0.458, Cor+Lbl=0.422) almost exactly, so treat 05 as the
config to carry forward for any downstream use (e.g. a final test-set
submission) unless the multi-seed reruns (Aim 1) show this margin isn't
stable across seeds.

**Solution 03 (the newly-trained missing ablation cell) behaves exactly as
expected**: nearly identical overall numbers to solution 04 (same r_max=8,
differing only by the ~1,200 LLM-assisted examples 04 has and 03 doesn't),
with the per-label differences landing in sensible directions — `other`
drops without the LLM-targeted injection (0.067 vs 0.092), `OCR` recovers
somewhat (0.320 vs 0.274), consistent with the paper's finding that OCR
consistently degrades under heavier augmentation. This is a coherent
result, not just a plausible-looking one.

**`other` is the weak point across every augmented config** (0.061-0.114,
vs. 0.027 for the unaugmented baseline) — augmentation clearly helps this
class relative to no augmentation at all, but it's still the lowest-scoring
category everywhere, matching the paper's characterization of `other` as
the least-grounded, hardest-to-synthesize class of the five.

Solution 01 has never been run, so no ensemble comparison (01 + one of
03-06) is possible yet — still a reasonable next step once it has real
numbers to compare against.
