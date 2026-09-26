# Aim 2 preliminary finding: synthetic-vs-natural lexical separability

**Status**: first pass, local classical-ML analysis (no GPU/VLM needed) —
run via `analysis/aim2_synthetic_vs_natural_classifier.py`. Extends the
paper's Appendix span-*length*-only comparison (`app:span-length`) to
lexical form, per thesis Aim 2 / Hypothesis H2.

## Setup

- **Natural spans**: gold `labels` directly from `data/train.jsonl`
  (unaugmented — avoids double-counting the upsampled duplicates that
  live in any `train_augmented.jsonl`).
- **Synthetic spans**: `solutions/05_vlm_lora_synthetic_tuned/
  train_augmented.jsonl`'s `synth_*`-tagged records (tuned r_max=3
  config — arbitrary pick among 03-06, since the heuristic-injection
  logic itself is identical across all four; only upsampling/LLM differ).
- **Features**: length, alpha/digit/space character ratios, word count,
  and character-trigram Jaccard similarity to a **held-out** reference
  pool of natural spans of the same category (the reference pool is
  excluded from the classifier's own training/eval instances, so a
  natural span can never trivially match itself).
- **Classifier**: per-category logistic regression, 5-fold stratified CV,
  `class_weight='balanced'` (natural spans vastly outnumber synthetic
  ones per category — up to ~14,000:150 for invention).

## Results

| Category | n natural | n synthetic | AUC | Dominant feature |
|---|---|---|---|---|
| mischaracterization | 1500 | 150 | **0.859 ± 0.034** | `length` (-6.11) |
| other | 1180 | 450 | **0.828 ± 0.030** | `length` (+1.80), `max_trigram_jaccard` (-1.56) |
| miscounting | 1500 | 450 | 0.742 ± 0.021 | `space_ratio`, `alpha_ratio`, `digit_ratio` (all ~2) |
| invention | 1500 | 150 | 0.727 ± 0.040 | `digit_ratio` (-3.58), `word_count` (+2.42) |
| **OCR** | 984 | 450 | **0.726 ± 0.031** | `max_trigram_jaccard` (-1.23) — **not** length (+0.28, weakest of all 6 features) |

(Full standardized coefficients for every category in
`aim2_lexical_classifier_results.json`.)

## Interpretation

**All five categories are separable well above chance** — synthetic spans
are not lexically indistinguishable from natural ones anywhere, which on
its own is a useful confirmation that there's real signal here for a
downstream classifier (the second half of Aim 2 not yet attempted:
distinguishing at the *response* level, or using this as an auxiliary
training signal).

**But the ranking does not match a naive reading of H2.** H2 predicted
OCR would be the *most* separable category, explaining why augmentation
consistently degrades it (RESULTS_SUMMARY.md's multi-seed analysis: OCR is
the single most seed-robust *hurt* category, baseline beats tuned in all
3 seeds). Instead, **OCR is tied for the *least* separable** (0.726,
essentially identical to invention's 0.727) — and invention is a category
augmentation *helps*. Raw lexical separability by these features does not
cleanly predict the direction of the performance effect.

**What the feature breakdown does explain**, and where it lines up with
the paper's own Appendix table (`tab:spanlen`):

- `mischaracterization` and `other` — the two most separable categories —
  are exactly the two with the *largest* natural-vs-synthetic length gaps
  in the paper's table (mischaracterization: natural median 7 vs synthetic
  median 1; other: natural median 9 vs synthetic median 145). Their
  classifiers are dominated almost entirely by the `length` feature,
  consistent with that.
- `OCR` and `miscounting` were the two categories the paper's table found
  *closest* length-matched (OCR: natural mean 8.0 vs synthetic mean 7.7).
  Consistent with that, `length` is OCR's *weakest* feature here (+0.283,
  last of six) — but OCR is still separable, just by a different
  mechanism: `max_trigram_jaccard` (character-level lexical identity, not
  span length) is what's actually driving it. This matches the
  qualitative gap already flagged in `solutions/03_vlm_lora_synthetic_
  augmented/README.md`'s own OCR section: the confusable-character swap
  heuristic closed the alpha-only-character gap from 34%→55% against a
  74% real-data target, but explicitly did **not** fully close it. The
  trigram feature here is plausibly picking up exactly that residual
  character-composition mismatch — grounded in something already
  independently observed, not a new coincidence.

**Refined framing for H2, going forward**: separability alone (by generic
lexical features) doesn't explain *why* OCR specifically degrades while
invention specifically improves, since both are equally separable. The
more specific, already-motivated lead is the **character-composition**
finding — OCR's separability is driven by *how* the text differs
(trigram/character identity) rather than *how much* (length), and that
specific kind of difference is the one plausible enough to explain a model
learning a systematically wrong pattern for what "OCR-like" text looks
like. Testing this properly needs a feature that isolates it directly
(e.g. exact alpha-only-span rate, matching the README's own 34%/55%/74%
diagnostic) rather than inferring it indirectly through trigram Jaccard.

## Not yet attempted (next steps for Aim 2)

- **Image-question context features** (thesis proposal's other named
  dimension for Aim 2) — nothing here uses the image or the question at
  all yet; this analysis is response-text-only.
- **Direct alpha-only-span-rate feature** for OCR specifically, isolating
  the exact diagnostic already used qualitatively in solution 03's README,
  rather than relying on trigram Jaccard as an indirect proxy.
- **Cross-solution check**: this used solution 05's (r_max=3) synthetic
  spans only; worth confirming the same pattern holds using solution 03/04
  (r_max=8, more upsampled duplicates of the same underlying heuristic
  injections) before treating it as robust to which augmentation config
  generated the synthetic spans.
- **Response-level (not span-level) classifier**: this only classifies
  isolated span text, not spans in their full response/context — the
  actual "can a classifier distinguish synthetic from natural
  hallucinations" question from the thesis proposal is broader than this
  first pass.
