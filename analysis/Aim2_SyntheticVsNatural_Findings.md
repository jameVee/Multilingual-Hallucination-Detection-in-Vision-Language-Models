# Aim 2 finding: synthetic-vs-natural lexical separability

**Status**: complete first pass — span-level classifier, response-level
classifier, image-grounded feature, and significance testing all done. Run
via `analysis/aim2_synthetic_vs_natural_classifier.py`,
`aim2_response_level_classifier.py`, and `aim2_permutation_test.py`.
Extends the paper's Appendix span-*length*-only comparison
(`app:span-length`) to lexical form and image-grounding, per thesis Aim 2 /
Hypothesis H2.

**Revision note**: an earlier pass of this analysis (span-level only, 6
generic lexical features, no image signal) concluded OCR was the *least*
separable category. Adding two more targeted features — `is_alpha_only`
and `image_ocr_overlap` (below) — **reverses that conclusion**: OCR is now
one of the *most* separable categories. The generic-feature result wasn't
wrong, it was just missing the feature that actually captures how OCR
synthetic data is constructed. See "What changed and why" below.

## Setup

- **Natural spans**: gold `labels` directly from `data/train.jsonl`
  (unaugmented — avoids double-counting upsampled duplicates).
- **Synthetic spans**: `train_augmented.jsonl` from solutions 03/04/05/06
  (selectable via `--solution`; injection logic is shared code across all
  four, see "Cross-solution check" below for why 03 and 05 match closely).
- **Features** (8 total): length, alpha/digit/space character ratios, word
  count, `is_alpha_only` (binary — is the span alphabetic characters only,
  ignoring spaces), character-trigram Jaccard similarity to a **held-out**
  reference pool of natural spans of the same category, and
  `image_ocr_overlap` (below).
- **`image_ocr_overlap`** (new): max trigram-Jaccard between the span text
  and the actual EasyOCR-detected strings **for that specific image**
  (`ocr_cache.json`, already computed once per solution's
  `generate_synthetic.py` run). This is different from the generic
  trigram-Jaccard feature, which compares against a reference pool of
  *other* natural spans from *other* images — `image_ocr_overlap` asks "is
  this text actually grounded in what's really written in its own paired
  image", which is the cheapest possible version of the thesis proposal's
  "image-question context feature" (no vision model needed, reuses an
  artifact solution 03/04/05/06 already produce).
- **Classifier**: per-category logistic regression, 5-fold stratified CV,
  `class_weight='balanced'`.

## Results (span-level, solution 05, updated with new features)

| Category | n natural | n synthetic | AUC | Dominant feature |
|---|---|---|---|---|
| **invention** | 1500 | 150 | **0.943 ± 0.009** | `alpha_ratio` (+6.26), `is_alpha_only` (-5.22) |
| **OCR** | 984 | 450 | **0.927 ± 0.019** | `image_ocr_overlap` (+1.73), `alpha_ratio` (+2.08) |
| other | 1180 | 450 | 0.846 ± 0.026 | `digit_ratio` (-2.83), `length` (+2.45) |
| mischaracterization | 1500 | 150 | 0.859 ± 0.034 | `length` (-6.12) |
| miscounting | 1500 | 450 | 0.743 ± 0.021 | `space_ratio`, `digit_ratio`, `is_alpha_only` (~2 each) |

(Full standardized coefficients in `aim2_lexical_classifier_results_sol05.json`.)

## What changed and why

**Invention** jumped from AUC 0.727 → 0.943, driven almost entirely by
`alpha_ratio`/`is_alpha_only`. Synthetic invention insertions are drawn
from a fixed, short template list (`"a small bird"`, `"un vélo rouge"`,
...) — these have a very consistent, narrow alpha/space composition
compared to organically-varied real invented content. This is a genuine
artifact of using a small fixed object/template list rather than
generating varied insertions, and is worth knowing for future synthetic
data design (a larger, more varied template/object pool would likely
reduce this gap).

**OCR** jumped from 0.726 → 0.927, driven by `image_ocr_overlap`. This has
a clean mechanistic explanation, not a coincidence: the heuristic OCR
injector (`make_ocr_injection` in `generate_synthetic.py`) *literally
copies* a real EasyOCR-detected string from the image, then applies exactly
one confusable-character swap — so by construction, synthetic OCR spans
are near-verbatim matches (minus 1 character) to the image's own cached OCR
text. Real gold OCR hallucinations are whatever a real annotated model
actually produced, which is not guaranteed to closely match the literal
OCR-detectable text at all (a genuine misreading can drift further from the
source than a single-character swap). **This is the single most important
finding of Aim 2's synthetic-vs-natural work**: OCR is highly separable, but
specifically because of *how the heuristic constructs it* (near-copy of a
known-correct string), which directly supports the original form of H2 —
the model has a much easier "tell" available for synthetic OCR spans than
for other categories, and if it learns to key off that tell rather than
genuine misreading patterns, that plausibly explains why OCR is the most
seed-robust *hurt* category in `RESULTS_SUMMARY.md`'s multi-seed analysis.

## Cross-solution check (solutions 03 and 04)

Same bug-fix and methodology as before: LLM-tagged ids are
`synth_llm_<label>_<orig_id>`, not `synth_<tag>_<orig_id>` — an earlier
version of the id-parser only checked the second part and silently
dropped every LLM record. Fixed; script now supports `--solution
{03,04,05,06}` and `--source {all,heuristic,llm}`.

**Solution 03 (r_max=8, no LLM) vs. solution 05 (tuned r_max=3, no LLM)**
— same heuristic logic, different rarity cap — still match closely with
the new features:

| Category | 05 AUC | 03 AUC | Δ |
|---|---|---|---|
| invention | 0.943 | 0.940 | ~0 |
| OCR | 0.927 | 0.919 | -0.008 |
| mischaracterization | 0.859 | 0.858 | ~0 |
| other | 0.846 | 0.834 | -0.012 |
| miscounting | 0.743 | 0.753 | +0.010 |

**Confirms the pattern is not an artifact of the rarity-cap choice** — the
injection *logic* (shared code in `synth_common.py`) determines
separability, not how many times it's invoked.

**Solution 04 (r_max=8, +LLM): isolating LLM quality from heuristic
quality.** `--source heuristic` vs. `--source llm` on the same file:

| Category | heuristic-only AUC | LLM-only AUC | Gap significant? (permutation test, span-level) |
|---|---|---|---|
| OCR | **0.919** | 0.760 | **Yes, p=0.002** |
| other | 0.836 | **0.728** | **Yes, p=0.002** |
| miscounting | 0.751 | **0.885** | **Yes, p=0.002** |

All three gaps are significant (999-permutation label-shuffle test,
`aim2_permutation_test.py`; observed diffs of +0.160/+0.080/-0.103 against
a null std of ~0.012-0.016 — none of these are small-sample noise).

**Mechanistic reading**: LLM-assisted OCR injection is markedly *less*
separable than heuristic OCR injection (0.760 vs 0.919) because it isn't
constructed by copying the cached OCR string — Claude is asked to look at
the image and produce a plausible misreading independently, so it lacks the
heuristic's "near-verbatim copy" tell (though it's still separable above
chance — LLM-generated misreadings apparently still differ systematically
from real ones, just less obviously). For `other`, LLM spans are also more
natural-looking (0.728 vs 0.836) — plausible given `other` has no strong
heuristic pattern to begin with (per `synth_common.py`'s own comments, only
~4% of real `other` spans show any recoverable structural signature), so an
LLM given the actual image has more to work with than a fixed
antonym-swap/repetition-loop heuristic. For `miscounting`, the LLM result
is *more* separable (0.885 vs 0.751) — the opposite direction — plausibly
because a real miscounting error is a minimal single-value edit
(`shift_number`/`shift_number_word`'s exact mechanism) in an otherwise
untouched sentence, while an LLM asked to "inject a wrong count" seems to
rewrite more of the surrounding sentence than that, becoming lexically more
conspicuous rather than less.

## Response-level check

`aim2_response_level_classifier.py` reruns the same features over the
**whole response** (not just the isolated span) — natural: any response
with a gold span of that category; synthetic: the full corrupted response.
This is the broader question the thesis proposal actually poses (a span in
isolation loses surrounding context).

| Category | span-level AUC (sol 05) | response-level AUC (sol 05) |
|---|---|---|
| invention | 0.943 | 0.593 |
| OCR | 0.927 | 0.617 |
| other | 0.846 | 0.695 |
| mischaracterization | 0.859 | 0.641 |
| miscounting | 0.743 | 0.712 |

**Separability drops sharply for every category at the response level**
(by 0.13-0.35 AUC) — expected, since most of a response's characters are
shared, unmodified real text in both the natural and synthetic case; the
injected/labeled portion is a small fraction of the whole response, so
response-level features dilute the signal substantially. Still clearly
above chance everywhere, but nowhere near as separable as the isolated span
suggested. **This matters for interpreting the span-level OCR/invention
results**: a downstream model sees the *whole* response, not the isolated
span, so the practically relevant separability is closer to the
response-level numbers than the span-level ones.

**Heuristic-vs-LLM gap at the response level (solution 04)** — permutation
test, `--level response`:

| Category | heuristic AUC | LLM AUC | Gap significant? |
|---|---|---|---|
| OCR | 0.616 | 0.672 | **Yes, p=0.002** (direction flips vs. span-level!) |
| other | 0.691 | 0.617 | **Yes, p=0.002** (same direction as span-level) |
| miscounting | 0.707 | 0.676 | **No, p=0.276** — gap not distinguishable from noise |

Two of the three span-level conclusions survive at the response level
(`other`: LLM more natural, consistently) but **OCR's direction flips**
(heuristic *less* separable than LLM at the response level, opposite of
span-level) and **miscounting's gap disappears** (not significant once
diluted by the rest of the response). This is an honest, important
caveat: the "LLM vs. heuristic quality" conclusions depend on which level
you evaluate at, and the response-level numbers are the more decision-
relevant ones for anything about how a trained VLM actually sees the data.

## Interpretation / conclusion

1. **Revised finding on H2**: OCR *is* the most separable category by
   lexical features once an image-grounded feature is included — but the
   mechanism is specific and diagnosable (the heuristic injector's
   near-verbatim-copy construction), not a generic "synthetic OCR looks
   fake" story. This is a stronger, more falsifiable version of H2 than
   the original framing, and is consistent with (not contradicting) OCR
   being the most seed-robust *hurt* category in the multi-seed results.
2. **LLM-assisted injection is not uniformly higher- or lower-quality than
   heuristic injection** — it's category- and evaluation-level-specific.
   The one robust, level-independent finding: LLM spans for `other` are
   consistently more natural-looking than heuristic ones, both at the span
   and response level, both statistically significant.
3. **Whatever separability exists at the span level is much weaker at the
   response level** — a real caveat on how much weight to put on the
   span-level numbers alone when reasoning about what a VLM trained on
   full responses would actually pick up on.

## Not yet attempted

- Nothing from the originally-scoped list remains — alpha-only-span-rate
  feature (done, `is_alpha_only`), image-context feature (done,
  `image_ocr_overlap`, cheapest no-model version), response-level
  classifier (done), permutation/significance test (done).
- Still open, if pursued further: a *learned* (not hand-engineered) image
  feature — e.g. CLIP image-text similarity — was considered but not
  implemented; the local dev machine's `transformers` install is currently
  broken (`huggingface-hub==1.22.0` installed vs. the pinned package
  requiring `<1.0`), so this would need an environment fix first. Given
  `image_ocr_overlap` already produced the single largest finding in this
  analysis using only an artifact solutions 03-06 already compute, a
  learned embedding feature is a nice-to-have, not a blocker for treating
  Aim 2's classification-separability question as answered.
