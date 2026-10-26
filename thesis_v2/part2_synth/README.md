# Part 2: isolated synthetic-data conditions

| Cond | Training set | Built by |
|---|---|---|
| C0 | real train only | (Part 1 baseline, same backbone) |
| C1 | real + `C1_upsample.jsonl` (8,301 duplicates, rarity multiplier 3) | `build_conditions --stage c1c4` |
| C2 | real + `C2_heuristic.jsonl` | `--stage sites` then `--stage finalize` |
| C3 | real + `C3_llm.jsonl` (same sites as C2, Claude writes the content) | `llm_paired.py` then `--stage finalize` |
| C4 | real + `C4_retrieval.jsonl` (real annotated spans from same-image siblings) | `--stage c1c4` then `--stage finalize` |

C2/C3/C4 have identical record counts per (category, language), asserted in
`stage_finalize`. C2 and C3 edit the same response at the same site with the same
`prob`; only the replacement text differs.

## Run order (laptop; no GPU needed)

```bash
python -m thesis_v2.part2_synth.build_conditions --stage c1c4
python -m thesis_v2.part2_synth.build_conditions --stage sites
python -m thesis_v2.part2_synth.llm_paired --dry_run        # check prompts and the cost estimate
python -m thesis_v2.part2_synth.llm_paired                  # needs ANTHROPIC_API_KEY in the environment
python -m thesis_v2.part2_synth.build_conditions --stage finalize   # writes C2/C3/C4 + realism gate
```

Everything is deterministic (`DATA_SEED = 7`) except the Claude calls, whose results are
cached in `thesis_v2/runs/part2/llm_paired_cache.json`.

## Realism gate (train data only)

Thresholds were fixed before any data was generated: per category, KS(span length) and
KS(relative position) ≤ 0.15, TV(prob over 1/3, 2/3, 1) ≤ 0.10, and OCR in-quote rate and
miscounting bare-number rate within ±0.10 of real. Real spans are reweighted to the
synthetic set's language mix, because the cell volumes are fixed by design and zh spans
are much shorter. Results go to `thesis_v2/results/part2_realism.json`.

## Development iterations (train data only, before the gate was frozen)

The generators were revised against train-data statistics and by reading samples. No
validation or test data was used. Each revision fixed a measured artifact:

**C4 (retrieval)**
- Volume was too small at first. Added insertion at sentence boundaries, not only
  replacement of a similar sentence.
- Real code-switched donor sentences were transplanted into other-language answers.
  Added a donor language check, and later a target language check (some records labeled
  `it` are written in English or Portuguese).
- Spans sat too early. Insertion at the end is allowed when the sampled real position
  calls for it, so OCR ends up 12% at the end vs 14% real (the old heuristic put 79% there).
- Truncated donor responses left unterminated fragments. These are now rejected.
- The count-contradiction check missed `**6** 个` and measure words like 根 and 人. Fixed.

**C2 (heuristic) / shared sites**
- Kind fallbacks skewed the mix: quoted OCR text and repetition are easy to find. Each
  kind now has a quota, filled in one pass.
- `shift_count` produced "zero car" and "deux chemise". Counts of 1 are now used only
  with noun pluralization and verb agreement ("There is one dog" → "There are two dogs").
  The wrong value follows the real distribution of miscount values.
- Real non-number miscount spans are mostly a number and its noun labeled as two
  adjacent spans (`four« legs»`, `四«个»`). Added the `count_split` kind.
- zh single-character antonym swaps broke compound words (展开 → 展关). Replaced by a
  whole-word list.
- `**bold**` text was treated as OCR. Bold now counts only when EasyOCR saw the text.
- Leak strings now come from real leak spans ("Human" plus an unlabeled ":").

**Known remaining gap (C2).** The miscounting span-length KS ranges from 0.14 to 0.16
across random streams. This sits at the 0.15 threshold and inside sampling noise (±0.07 at
n≈300). The cause is the number surface form: real errors include "one"/"une"/digits,
which the heuristic does not produce. That is a difference of 1–2 characters, not a
detectable shortcut like the old 144-character spans. Whatever the final gate reports is
recorded as-is.

**Final selection.** In each cell, the kept records are chosen from a 2× oversampled pool
by stratified sampling on span length against that (category, language)'s real quintiles.
The same procedure is applied to every method (`realism.matched_order`).
