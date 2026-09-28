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

## Multi-seed variance (Aim 1 Phase 3 — baseline vs. tuned, 3 seeds each)

The single-seed numbers above understate how much these configs move
seed-to-seed. Aggregating seeds 42 (original), 123, and 2024 for solutions
02 and 05 (`solutions/common/run_seed_variant.py` outputs):

| Metric | 02 mean ± std | 05 mean ± std | gap (05-02) | gap / avg seed-std |
|---|---|---|---|---|
| Span IoU | 0.4446 ± 0.0072 | 0.4566 ± 0.0031 | +0.0120 | **2.3x** |
| Cor (overall) | 0.4785 ± 0.0083 | 0.4880 ± 0.0024 | +0.0094 | **1.8x** |
| Cor+Lbl | 0.4219 ± 0.0056 | 0.4206 ± 0.0011 | **-0.0012** | **-0.4x** |

**This overturns the single-seed conclusion below it, and confirms exactly
the caveat the paper itself flagged** ("tuned's gain over the baseline is
not significant... this bootstrap doesn't capture training-seed variance").
The apparent "05 beats baseline on Cor+Lbl" result was an artifact of
comparing single, arbitrary seed-42 runs — baseline's seed-42 Cor+Lbl
(0.4172) happened to be its *worst* of the three seeds (123 and 2024 both
exceeded it), while solution 05 was unusually stable seed-to-seed
(std=0.0011 vs. baseline's 0.0056). Averaged properly, baseline and tuned
are statistically indistinguishable on Cor+Lbl (gap smaller than the
noise), even though **IoU and raw Cor gains for solution 05 are real and
robust** (gap clearly exceeds seed noise on both).

Per-label Cor across the same 3 seeds tells the more decision-relevant
story — which categories does tuned augmentation actually, robustly help
or hurt, once seed noise is accounted for:

| Category | 02 mean ± std | 05 mean ± std | Robust? |
|---|---|---|---|
| OCR | 0.3635 ± 0.0085 | 0.3191 ± 0.0183 | **Robustly hurt** — ranges barely overlap, baseline wins all 3 seeds |
| mischaracterization | 0.1227 ± 0.0044 | 0.1414 ± 0.0061 | **Robustly helped** — tuned wins all 3 seeds, non-overlapping |
| other | 0.0414 ± 0.0122 | 0.0746 ± 0.0083 | **Robustly helped** — tuned wins all 3 seeds |
| miscounting | 0.4038 ± 0.0191 | 0.4175 ± 0.0035 | Likely helped, and far more *stable* under tuning |
| invention | 0.2093 ± 0.0080 | 0.2191 ± 0.0051 | Modest edge, ranges overlap |

**Takeaway for Aim 2**: the paper's OCR-degradation finding isn't a
single-seed fluke — it's the most seed-robust pattern in this whole
dataset (baseline beats tuned on OCR in all 3 seeds, non-overlapping
ranges). That's the strongest possible motivation for Aim 2's synthetic-
vs-natural-span classifier: if synthetic OCR spans are lexically/
structurally distinguishable from natural ones in a way the model
actually picks up on and fails to generalize from, this is where it
should show up most clearly.

## Test set (official SHROOM-Visions submission — Aim 1 conclusion)

Submitted single-seed (seed 42) test predictions for solutions 02-06 to the
official platform (raw output: `solutions/test_results_official_submission.txt`).
Solution 01 was never run, so it has no test submission either.

| Solution | EN Cor+Lbl | FR Cor+Lbl | IT Cor+Lbl | ZH Cor+Lbl | avg Cor+Lbl |
|---|---|---|---|---|---|
| 02 — real data (baseline) | 0.3516 | 0.3968 | 0.3926 | **0.4495** | **0.3976** |
| 03 — r_max=8, no LLM (the missing cell) | 0.3337 | 0.3786 | 0.3863 | 0.4426 | 0.3853 |
| 04 — r_max=8, + LLM | 0.3364 | 0.3758 | 0.3692 | 0.4175 | 0.3747 |
| 05 — tuned r_max=3, no LLM | 0.3373 | 0.3778 | 0.3893 | 0.4564 | 0.3902 |
| 06 — tuned r_max=3, + LLM | 0.3350 | 0.3493 | 0.3857 | 0.4377 | 0.3769 |

**H1 is not supported.** H1 predicted that training the missing ablation
cell (solution 03) would show the rarity-cap and LLM-assisted-injection
effects don't act uniformly, yielding *some* configuration that beats the
real-data baseline on Cor+Lbl "in more than the single language (Chinese)
achieved so far." Instead, across all 16 (augmented config × language)
combinations on the actual test set, **only one beats the baseline at
all**: solution 05 on Chinese, by +0.0069 — smaller than baseline's own
seed-to-seed std (0.0056) measured in the val multi-seed analysis above,
so not distinguishable from noise even before running a real significance
test. Every other augmented config loses to baseline in every language,
including **solution 03 itself — the specific cell H1 was staked on —
which loses to baseline in all 4 languages**, not just fails to extend
beyond Chinese. The shared task's original one-language pattern isn't
exceeded; it's barely matched, and by a different, already-known config
(05), not by the newly-completed missing cell.

**This is not a contradiction of the val-set multi-seed finding above — it's
independent confirmation of the same split.** Checking IoU (the label-
agnostic metric) on this same test data: augmented configs beat the
baseline in **15 of the same 16 comparisons** (only solution 06 on French
misses). This is exactly the pattern the val multi-seed analysis already
established: **augmentation's IoU/raw-Cor gains are real and robust; its
Cor+Lbl gains are not** — now verified on genuinely held-out test data the
model never touched during any tuning decision, not just on val.

**Revised takeaway for the thesis**: H1 as literally stated is rejected —
completing the ablation did not find a configuration that generalizes
augmentation's benefit past Chinese on the metric that matters most. The
more accurate characterization, consistent across both val (multi-seed)
and test (single-seed, this section): synthetic augmentation in this
pipeline reliably improves span *localization* (IoU) and *label-agnostic*
correlation (Cor), but does not reliably improve *label-aware* correlation
(Cor+Lbl) over real-data-only training, on any language tested. That
distinction — not "does augmentation help" but "which metric does it
help" — is the real, evidence-backed finding Aim 1 produced.

## Recommendation

**Solution 05's headline "beats baseline" claim does not survive seed
variance on val, and does not survive contact with the real test set
either** — see both sections above. It remains the best config on IoU and
raw Cor, robustly so on both val and test, and per-label (val) it
robustly helps `mischaracterization`/`other`/`miscounting` — just don't
claim it beats the baseline overall without immediately caveating Cor+Lbl,
and don't claim it beats baseline on Chinese Cor+Lbl either without noting
that +0.0069 margin is smaller than known seed noise. Treat 05 as the
strongest *available* config for downstream use on the strength of its
label-agnostic and per-label gains, not on any Cor+Lbl win.

**Solution 03 (the newly-trained missing ablation cell) behaves exactly as
expected internally, but did not close the gap H1 was betting on.** On
val, nearly identical overall numbers to solution 04 (same r_max=8,
differing only by the ~1,200 LLM-assisted examples 04 has and 03 doesn't),
with per-label differences landing in sensible directions — `other` drops
without the LLM-targeted injection (0.067 vs 0.092), `OCR` recovers
somewhat (0.320 vs 0.274), consistent with the paper's finding that OCR
consistently degrades under heavier augmentation. That's a coherent
result, not just a plausible-looking one — but on the real test set, this
same solution 03 loses to the real-data baseline on Cor+Lbl in all 4
languages (see "Test set" section above). Completing the missing ablation
cell answered the *engineering* question (is this cell's behavior
internally consistent with the other three) without answering the
*research* question H1 posed (does isolating it unlock a config that beats
baseline more broadly) — those turned out to be separate questions.

**`other` is the weak point across every augmented config** (0.061-0.114,
vs. 0.027 for the unaugmented baseline) — augmentation clearly helps this
class relative to no augmentation at all, but it's still the lowest-scoring
category everywhere, matching the paper's characterization of `other` as
the least-grounded, hardest-to-synthesize class of the five.

Solution 01 has never been run, so no ensemble comparison (01 + one of
03-06) is possible yet — still a reasonable next step once it has real
numbers to compare against.
