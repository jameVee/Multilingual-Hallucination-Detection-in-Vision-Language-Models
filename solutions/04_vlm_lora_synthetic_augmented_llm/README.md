# Solution 04 — VLM LoRA Fine-tune, Synthetic Augmented + LLM-Assisted Injection

## Method

Same LoRA recipe as solution 02 (reuses its `train_lora.py`/`predict.py`
unchanged via `--train_file`), but trained on `data/train.jsonl` **plus**
three synthetic-augmentation methods stacked together, specifically aimed at
the classes the released data starves: OCR (6.0% of train examples), other
(8.9%), miscounting (12.3%), vs. invention/mischaracterization at ~41-44%
each.

This solution is identical to [solution 03](../03_vlm_lora_synthetic_augmented/README.md)
(same `generate_synthetic.py`, same r_max=8 rarity cap, methods (a) and (b)
unchanged) **except that method (c), LLM-assisted injection, is included by
default here** whenever `ANTHROPIC_API_KEY` is set — solution 03 always
skips it. The two live in separate folders (rather than one folder with a
flag) specifically so both can run in parallel Colab sessions without ever
touching the same Drive paths, and so the ablation between "r_max=8" and
"r_max=8 + LLM-assisted injection" is a folder-vs-folder diff, not a
config toggle to remember.

### (a) Rarity + certainty aware upsampling — `S4_AgreementUpsampling` successor

The one prior strategy (`last_experiment/S4_AgreementUpsampling`) that beat
its baseline. Extends it: the original duplicated examples only by
annotator-agreement tier (unanimous spans 4x, majority-agreement 2x).
This version *also* multiplies by an explicit per-class rarity factor sized
off the real current class counts (`compute_rarity_multipliers` in
`solutions/common/synth_common.py`), so OCR/other get boosted harder than a
flat tier multiplier alone would — pure deterministic duplication, no
model/API, so it's free and fully reproducible.

### (b) Grounded heuristic injection

Corrupts *clean* responses to synthesize new hallucinated examples, but
grounded in real signal rather than S1a's ungrounded dictionary swaps (which
was the worst-performing prior strategy):

- **OCR**: runs EasyOCR (`ocr_extract.py`, pure-pip, no system Tesseract
  needed, covers en/fr/it/zh via `ch_sim` **using two separate readers**,
  one for en/fr/it and one for ch_sim/en — EasyOCR hard-errors if you try to
  load all four in one `Reader()` call, "Chinese_sim is only compatible with
  English"; caught by actually running it, not just reading the docs) once
  over every training image, then corrupts *text EasyOCR actually detected
  in that image* with confusable-character swaps (rn↔m, cl↔d, vv↔w, I/l
  preferred; O↔0, S↔5, etc. as a fallback) — the fabricated "misreading" is
  grounded in something real in the picture, not an arbitrary word.

  **Checked against real gold data, not just asserted**: extracted all 1,284
  real gold OCR spans from `data/train.jsonl` (median length 5 chars, 74%
  alpha-only / 16% numeric-only / 10% mixed — a VLM misreading text usually
  outputs a different plausible *word*, not a word visibly contaminated with
  digit lookalikes) and generated a real sample of synthetic OCR spans to
  compare. Length distribution matched well from the start (synthetic
  mean 7.2 vs. real 8.0 chars). Character composition did not: the first
  version, swapping uniformly at random and scaling swap count with text
  length, produced 64% "mixed" alpha+digit text against the real data's 10%.
  Fixed in two steps — bias `confusable_swap` (`solutions/common/synth_common.py`)
  toward letter-for-letter confusions that stay alphabetic, and cap
  injections at one swap regardless of source-text length (matching a
  single localized misreading rather than a length-proportional smear) —
  which closed the gap from 34%→55% alpha-only. **Not fully closed**: real
  gold data is 74% alpha-only; getting materially closer with pure
  character-substitution runs into a structural limit (many detected
  strings simply contain no letter-for-letter-confusable substring at all,
  forcing a digit-style fallback regardless of tuning) — a genuinely richer
  fix would need a learned or dictionary-based word-confusion model, not
  attempted here.

- **miscounting**: shifts a number mention *already present* in the clean
  response by a small offset (±1 most likely, ±2 less likely — a modeling
  assumption about typical counting-error magnitude, since the released
  data marks which spans are wrong but not what the original correct value
  was, so this isn't literally fit from gold data).

  **Checked against real gold data**: extracted all 2,747 real gold
  miscounting spans — 90% are a single word, median 3 chars — and found
  **79% contain no digit at all**; they're spelled-out number words ("two",
  "trois", "四", "个"). An earlier version of `shift_number` only matched
  `\d+`, so it could only ever fire on the 21% digit minority and had zero
  ability to even attempt the 79% majority case. Fixed by adding
  `shift_number_word` (per-language number-word tables, 0-20, for en/fr/it/
  zh — zh via substring match since it has no word boundaries), tried
  first, falling back to the digit version only if no word-number is found.
  Re-measured on synthetic output: 90% word-form / 9% digit-form, close to
  the real 79%/21% split (previously 0%/100%).

- **mischaracterization**: swaps a color/size/material/texture word already
  in the response for a plausible antonym.

  **Checked against real gold data**: the antonym dictionary was English
  words only, matched via an `[A-Za-z]+` regex applied to all 4 languages'
  responses. Measured against the actual clean-seed pool, it fired on 14.8%
  of English seeds but only 3-6% of fr/it/zh ones (occasional English
  loanwords) — this injector barely ever ran for 3 of the 4 task languages.
  Fixed by translating the dictionary into fr/it/zh (zh matched by
  substring, same approach as the OCR/miscounting fixes above). Re-measured:
  fires on 25-64% of clean seeds across all 4 languages (previously ~0% for
  fr/it/zh). Real gold spans are also far more varied than a fixed
  color/size dictionary can cover (species names, verb phrases like "have
  four legs", agent phrases like "created by") — not checked further; this
  fix closes the *language*-coverage gap, not full semantic coverage.

- **other**: swaps a state/direction/relation word (open↔closed,
  left↔right, ...) — kept in a *separate* category from the visual-attribute
  swap above so the injected label actually matches how docs/2026.md itself
  distinguishes "wrong description of visible content" (mischaracterization)
  from the catch-all "other".

  **Checked against real gold data — this is the weakest-grounded class of
  the five, and stays that way even after fixing what could be fixed.**
  Read 25 real gold `other` spans with surrounding context directly. They
  are dominated by generation-*degeneration* artifacts that have nothing to
  do with semantic content errors: repetition loops (a phrase repeated
  50+ times), prompt/format leakage ("You are a helpful assistant", `` ```json ``
  fragments), and wrong-language generation (a "French" example that was
  actually Portuguese) — not state/direction word swaps. Before assuming
  that meant "make repetition-loops the primary method," quantified it
  first to avoid overfitting to a few dramatic examples: only **~4%** of
  the 1,480 real `other` spans show a clear repeated-chunk pattern; the
  other ~96% is a heterogeneous mix (truncated word fragments, odd word
  choices, foreign loanwords) with no single recoverable structural
  signature. So `other` now draws from **two** contributors — the
  (now-translated, same language-coverage fix as mischaracterization above)
  state-word swap most of the time, plus a new `inject_repetition_loop`
  (`solutions/common/synth_common.py`) the rest of the time, grounded in
  that measured ~4% pattern rather than invented. **Known remaining gap**:
  `inject_repetition_loop` splits on spaces to find a repeatable chunk,
  which doesn't work well for Chinese (no space-separated words), so it
  fires on ~53% of zh attempts vs. ~99% for en/fr/it — not fixed, since
  adding a proper Chinese tokenizer dependency for a category already this
  hard to ground didn't seem worth the added complexity.

- **invention**: the one category with no real signal to ground by
  definition (fabricated content is, by definition, absent from the image)
  — a short per-language templated addendum (localized object nouns, not a
  single English list reused across languages — an earlier version of this
  code mixed languages within one sentence, e.g. a Chinese response ending
  in "...trash can。", which was caught and fixed during development).

  **Checked against real gold data**: extracted all 14,473 real gold
  invention spans and their position within the response — only 14% land
  at/near the response's end; position is roughly uniform across the
  response (37%/35%/28% early/mid/late thirds). The original version always
  *appended* the fabricated sentence at the very end (100% late by
  construction). Fixed by inserting at a randomly chosen sentence boundary
  instead of always the last one. Re-measured on synthetic output:
  51%/24%/26% early/mid/late — a large improvement over 0/0/100, though
  still skewed toward "early" relative to the real 37/35/28 split (not
  perfectly calibrated, diminishing returns on tuning further).

### (c) LLM-assisted injection (`llm_inject.py`) — included by default in this solution

Adapted from `last_experiment/S1b_LLMInjection`'s generate-then-validate
two-pass (Claude proposes an injection, a second call checks it's genuinely
wrong given the image), fixed to never hardcode a key. **Entirely gated on
`ANTHROPIC_API_KEY` being set in the environment** — skips cleanly (zero API
calls) if absent; methods (a) and (b) never depend on it. Caches
incrementally so an interrupted run resumes rather than re-spending budget.
**Never paste a key into a chat or commit one** — set it as an environment
variable on the machine that runs this. Pass `--skip_llm` to this solution's
`run_pipeline.py` to reproduce solution 03's data (methods a+b only) from
this folder instead, though ordinarily you'd just run solution 03's own
pipeline for that.

## Files

- `ocr_extract.py` — EasyOCR pass, caches to `ocr_cache.json`
- `generate_synthetic.py` — runs (a) + (b) + optional (c), writes `train_augmented.jsonl`
  (identical copy of solution 03's script)
- `llm_inject.py` — method (c), importable/skippable independently
- `run_pipeline.py` — generate → train (via solution 02's script) → predict → score
  (defaults to including method (c), unlike solution 03's)

## Usage

```bash
python generate_synthetic.py --output ./train_augmented.jsonl   # inspect the augmented set first if you like
python run_pipeline.py --smoke_test --skip_llm --skip_ocr_extract --gpu_index 0   # verify the pipeline first
ANTHROPIC_API_KEY=... python run_pipeline.py --gpu_index 0 --epochs 2 --grad_accum 8

# bigger MoE alternative (24GB 3090 Ti only -- gpu_index 0, 3, or 5):
python run_pipeline.py --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --gpu_index 0
```

Base model and GPU guidance are identical to solution 02 (this solution
reuses its `train_lora.py`/`predict.py` unchanged) — default is
**Qwen3-VL-8B-Instruct** (dense), which fits any of the VM's 5 GPUs in
4-bit, including the 11 GB 2080 Tis. Pass `--model_id Qwen/Qwen3-VL-30B-A3B-Instruct`
for the bigger sparse-MoE alternative, restricted to `--gpu_index 0`, `3`,
or `5` (the 24 GB 3090 Tis; its 4-bit weights don't fit an 11 GB 2080 Ti).
See solution 02's README for the model card details, the MoE-specific LoRA
handling, and why this integration is newer/less locally-tested than the
rest of the repo.

## Local verification note

`generate_synthetic.py` (methods a + b, `--skip_llm --skip_ocr_extract`) was
run end-to-end locally and its output verified compatible with
`train_lora.py --dry_run` — **but that compatibility check ran against the
original Qwen2-VL-2B default**, before the model was switched to Qwen3-VL,
so it confirms the synthetic data's schema/format is fine but not either
Qwen3-VL size's specific processing path. The LLM-assisted path (c) was
verified to cleanly no-op (zero API calls) when `ANTHROPIC_API_KEY` is
unset. The actual model download+train+generate step was **not**
smoke-tested locally at all against Qwen3-VL, 8B or 30B-A3B (see solution
02's README for why) — run `--smoke_test` on the VM first.

## Results

Fill in after running on the VM — compare directly against solution 02's
and solution 03's `val_scores.json` (same model/recipe, only the training
data differs) to see whether LLM-assisted injection helps on top of methods
(a)+(b) alone, under the *official* scorer, unlike the prior
`last_experiment` strategies which were evaluated with self-rolled metrics:

| | Overall IoU | Overall Cor | Overall CorLbl | OCR Cor | miscounting Cor | other Cor |
|---|---|---|---|---|---|---|
| val (solution 02, real data only) | _pending VM run_ | | | | | |
| val (solution 03, + synthetic, no LLM) | _pending VM run_ | | | | | |
| val (solution 04, + synthetic + LLM) | _pending VM run_ | | | | | |
