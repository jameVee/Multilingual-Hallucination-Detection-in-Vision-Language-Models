# Pre-registered rules added after the original plan

## R1: second judge for the 3b CRF probe (2026-10-31)

Written before Pixtral-12B's val_select score existed and before Decision 1 was made.

**Risk:** if the Part 1 winner and the frozen 3b judge come from the same pretrained
language model, the detector and the probe may make the same errors, so combining them
(F3) adds little.

**Rule:**

| Part 1 winner | Judge(s) for 3b |
|---|---|
| Gemma-3-12B or Pixtral-12B | Qwen3-VL-8B only (already a different family) |
| Qwen3-VL-8B or InternVL3-8B (its language model is Qwen2.5) | Qwen3-VL-8B **and** Gemma-3-12B in 4-bit |

- Qwen3-VL-8B always runs, so the size of the old Aim 3 leak stays measurable with an
  unchanged judge.
- Gemma-3-12B runs in 4-bit because 12B does not fit a 24 GB card unquantized. This is
  reported as a difference from the Qwen judge's bf16 features.
- With two judges, each gets the same probe, hyperparameters and seeds. The better
  mean val_select Cor+Lbl over seeds decides which probe enters Decision 4 as F3 (one
  candidate); this choice is logged. Both results are reported.
- Reported either way: the correlation between the detector's and the probe's per-record
  Cor+Lbl on val_select. This is a diagnostic only and selects nothing.

**Outcome (2026-11-01):** Decision 1 chose Qwen3-VL-8B, so R1 applies: 3b runs with the
Qwen3-VL-8B judge and with a Gemma-3-12B judge in 4-bit.

## R2: Stage 1 synthetic pilot (2026-11-01)

Written before any pilot run started. Decision 1 and val_select diagnostics motivated the
pilot: rare categories account for a ceiling of +0.103, invention + mischaracterization for
+0.458. That diagnostic used val_select only; val_confirm and test are untouched by it.

**Question:** does any no-cost synthetic condition help before money is spent on an LLM?

**Setup:** Qwen3-VL-8B, seed 42, same recipe, val_select only.
Baseline: `p1_qwen3vl8b_seed42`, Cor+Lbl 0.4269; wrong-label records and clean false alarms
are reported alongside it.

| Condition | Added to real train |
|---|---|
| C1 | 8,301 upsampled duplicates |
| C4 | 719 real-span transplants: OCR, miscounting, other |
| C4x | C4 + 1,356 transplants: invention, mischaracterization |

C4x data: `build_conditions --stage pilot`. It adds a text-only polarity-conflict check for
the two claim categories, which rejects about half of the candidates. It has no stemming
(`chien`/`chiens` do not match), so some conflicts remain. Every cell came in under the 230
target, so all surviving candidates are used. C1 keeps its upsampling dose unchanged.

**Promotion rule, fixed now:**
- A condition is promoted to seeds 123 and 2024 only if its val_select gain is at least
  **+0.0055** (the known Qwen seed standard deviation).
- The plan's full "beats baseline" rule (3-seed mean gain above seed std, same sign on
  val_confirm, paired bootstrap lower bound above 0) still decides any claim.
- **Stage 2 (prompt testing) runs only if C4 or C4x is promoted.** If neither is, no LLM
  money is spent. If only C1 is promoted, synthetic text has not been shown to help.
- These three pilots are logged as one decision with three candidates, within the budget.

**Outcome (2026-11-03):** no condition promoted, so no LLM money is spent (Stage 2 does
not run). val_select Cor+Lbl against 0.4269, with the paired-bootstrap 95% CI:

| Condition | Cor+Lbl | Gain | 95% CI | Cor | IoU | Wrong-label records | Clean false alarms |
|---|---|---|---|---|---|---|---|
| C0 baseline | 0.4269 | – | – | 0.4771 | 0.4460 | 121 | 22 |
| C1 | 0.4245 | -0.0024 | [-0.0142, +0.0100] | 0.4856 | 0.4548 | 211 | 44 |
| C4 | 0.4229 | -0.0040 | [-0.0130, +0.0056] | 0.4806 | 0.4488 | 144 | 21 |
| C4x | 0.4293 | +0.0024 | [-0.0082, +0.0135] | 0.4866 | 0.4524 | 174 | 37 |

Every condition raised Cor and IoU (spans found better) and raised wrong-label records,
so Cor+Lbl stayed within noise. This is the old solutions 03–06 pattern, now isolated
condition by condition. Training compute is not held fixed: more records means more
optimizer steps (C0 3,020; C4 3,200; C4x 3,538; C1 5,094).

## R3: label-structure fixes (2026-11-05)

Written before any candidate below was scored on val.

**What motivated it.** These diagnostics used existing predictions on val_select (counts and
gold oracles, no candidate system scored) plus train:
- Wrong labels in the Stage 1 pilot are mostly invention <-> mischaracterization, not the
  synthesized categories. C1, which adds no new text, has the most (232; C0 128).
- Annotators often disagree: gold gives the same text both invention and mischaracterization
  in 221 of 1,515 val_select records (1,766 train records). Every system outputs this in
  0-11 records.
  - The shared decoder (`solutions/common/vlm_io.map_spans_to_positions`) drops a span that
    overlaps an already-placed span by more than half, whatever the label.
  - The 3a vote keeps one label per character (argmax).
  - The 3b probe is trained on one label per token (the first gold span wins) and keeps the
    argmax head.
- Fed perfect output (gold targets), the shared decoder caps val_select Cor+Lbl at 0.925.
  On the train round trip it loses 22-38% of gold spans per category.
- Oracle (gold used, diagnostic only): adding the second label where gold has it gives
  +0.032 to +0.040 Cor+Lbl for C0, C1 and C4x.

**Fixes, all fixed now:**
1. **Decoder** `common/span_io.py`, variant `near_prev`. The variant was chosen on the train
   round trip only: records not reproduced 3,939 (old) -> 2,780 / 2,693 / 2,531. Most
   remaining losses are whitespace trimming. With perfect output it gives val_select
   0.984 (old 0.925). Raw generations are saved from now on.
   - Every Qwen run is re-predicted on val; the new baseline is `dec2_p1_qwen3vl8b_seed42`.
   - The old-decoder numbers stay in the log and are reported as such.
   - Check: the new run decoded the old way must reproduce the old predictions.
2. **Part 2 re-check (decision D2b, 3 candidates).** C1, C4 and C4x under the new decoder
   against the new C0. The R2 promotion rule is unchanged (val_select gain >= +0.0055),
   and so is the Stage 2 rule (LLM money only if C4 or C4x is promoted).
3. **3a v2 (decision D3_3a2_qwen_seed42, candidates k in {2,3,4}).**
   - 8 decodes, T=0.7, re-sampled with raw saved and the new decoder.
   - The same adapter's greedy prediction is a 9th voter. Single decodes score below greedy
     on val_select (mean 0.4078 vs 0.4269), so greedy is included.
   - Per-label vote (`voting.vote(per_label=True)`).
   - Chosen on val_select and confirmed on val_confirm. The original D3 stays as recorded.
4. **3b v2 (`crf_probe --multilabel`).**
   - Every gold span covering a token sets its category (checked to contain the old single
     label), and each head is decoded independently.
   - Same hyperparameters, seeds 42/123/2024, early stopping on val_select loss. Nothing is
     selected.
   - The five CRFs are batched: same loss (checked), about 4.8x faster.
5. **Integration (F1-F4) is unchanged** and uses the fixed versions. F2 = F1 + 3a v2 is where
   the 3a per-label vote also acts as a label-consistency gate for a Part 2 model.

**Not decided here:** a revised synthetic condition (e.g. synthetic spans carrying the real
dual-label structure, training steps held equal to C0). It will be written as R4 after
D2b, before any training.

## R4: prior-matched synthetic conditions (2026-11-06)

Written after D2b and before any R4 training. Nothing below has been trained or scored.

**The R3 guess is ruled out.** Synthetic records already carry the real dual-label
structure. Records with two different labels on overlapping text, among records with a
hallucination: real train 29.1%, C1 33.4%, C4 31.4%, C4x 31.9%.

**What the D2b runs show (diagnostic, existing predictions, both val halves).** Cor+Lbl is
split into the part from gold-clean records and the part from gold-labeled records
(4-language mean, change vs dec2 C0, val_select / val_confirm):

| | Labeled-record part | Clean-record part | Clean records flagged (val_select) |
|---|---|---|---|
| C1 | +0.0150 / +0.0125 | -0.0145 / -0.0073 | 22 -> 44 |
| C4x | +0.0120 / +0.0126 | -0.0099 / -0.0067 | 22 -> 37 |
| C4 | -0.0032 / -0.0061 | +0.0006 / +0.0006 | 22 -> 21 |

- C1 and C4x detect hallucinations better on both halves, and lose the same amount by
  flagging clean records (a flagged clean record scores 0 instead of 1).
- Every added record carries a hallucination. The hallucinated share of training records
  is 71.56% in real train, 73.2% with C4, 75.7% with C4x and 83.1% with C1. False alarms
  follow the same order.
- Compute was not held fixed either: C0 3,020 optimizer steps, C4x 3,538, C1 5,094.

**Hypothesis.** The extra false alarms come from the shifted hallucinated share and the
extra steps, not from the synthetic text. Holding both fixed keeps the labeled-record
gain and removes the clean-record loss.

**Conditions (one decision, D2c_R4, 2 candidates).**
1. **C4x-b:** real train + C4 + C4x transplants (2,075) + 825 clean duplicates
   (`part2_synth/build_r4.py`). The duplicates are sampled without replacement (seed 0)
   from the 1,201 clean base records the transplants were made from, so the same image
   and prompt are also seen without the hallucination. Hallucinated share 71.56%.
2. **C1-b:** real train + C1 upsampling (8,301) + 3,299 clean duplicates drawn from all
   clean train records (without replacement, seed 0). Hallucinated share 71.56%.
- Both train for exactly **3,020 optimizer steps** (`train.py --step_budget 3020`; the LR
  schedule spans the budget). That is C0's step count, about 1.6 passes for C4x-b and 1.0
  for C1-b. Every other hyperparameter is C0's. Seed 42.
- The data files are fixed before training: `R4_C4xb_clean.jsonl`, `R4_C1b_clean.jsonl`.

**Rules (unchanged).**
- Baseline `dec2_p1_qwen3vl8b_seed42`, val_select 0.4310. A condition is promoted to seeds
  123 and 2024 if its val_select gain is at least +0.0055. The threshold applies to each
  condition on its own, so C4x-b may be checked as soon as it is scored. Both are logged
  as one decision once both are scored.
- LLM spend (Stage 2) runs only if C4x-b is promoted. If only C1-b is promoted,
  synthetic text has not been shown to help.
- Any claim still needs the full 3-seed "beats baseline" rule.
- Reported in every case: the clean / labeled split above. If the clean-record part
  still drops, the share and steps do not explain the false alarms, and that is reported.
  No further revision of Part 2 follows from R4.

**Compute order (by what each result decides).**
1. C4x-b seed 42: train, val, score. Decides whether Part 2 has a candidate and whether LLM
   money is spent.
2. C0 seeds 123 and 2024: train, greedy val and test. Every claim needs them (baseline
   seed std, 3-seed baseline).
3. 8 sampled decodes (T=0.7) on val and test for seeds 123 and 2024. Needed for the 3a v2
   3-seed claim (D3_3a2 k=3 reused, not re-decided).
4. C1-b seed 42: train, val, score.
If C4x-b is promoted, its seeds 123 and 2024 go in right after step 2.

**Outcome, C4x-b (2026-11-07): not promoted.** val_select 0.4228, -0.0083 vs dec2 C0
(CI [-0.0190, +0.0023]). So no LLM money is spent: Stage 2 does not run. Exactly 3,020
optimizer steps (checked).

| val_select | Cor+Lbl | Labeled-record part | Clean-record part | Records flagged | Clean records flagged |
|---|---|---|---|---|---|
| C0 | 0.4310 | 0.1536 | 0.2774 | 575 | 22 |
| C4x | 0.4332 | +0.0120 | -0.0099 | 652 | 37 |
| C4x-b | 0.4228 | -0.0050 | -0.0033 | 577 | 27 |

- The false-alarm half of the hypothesis holds. With the hallucinated share restored, the
  number of flagged records returns to C0's (577 vs 575), and clean false alarms mostly
  go away (27 vs 37).
- The other half fails: the labeled-record gain disappears with them. C4x's gain on
  labeled records came from flagging more records, not from telling hallucinated text
  apart better.
- Confound, introduced by R4 itself: at 3,020 steps C4x-b makes 1.61 passes, so real
  records are seen 19,483 times instead of C0's 24,160 (-19%). This alone could cost the
  -0.005. Per R4, no further Part 2 revision follows. C1-b stays in the queue (step 4) and
  is logged with C4x-b as D2c_R4.

**Outcome, 3a v2 3 seeds (2026-11-11): consistent gain, but the strict rule is not met.**
k=3 comes from D3_3a2_qwen_seed42 and is reused for seeds 123 and 2024 (`--k_from`, not
re-decided). Every gain compares a vote with the plain greedy decode of the same adapter.
The 3-seed CI averages each record's score over the seeds first, then runs the paired
bootstrap.

| Cor+Lbl | Seed 42 | Seed 123 | Seed 2024 | 3-seed mean gain | 3-seed CI |
|---|---|---|---|---|---|
| val_select | +0.0094 | +0.0063 | +0.0080 | +0.0079 (C0 std 0.0023) | [+0.0030, +0.0128] |
| val_confirm | +0.0052 | -0.0007 | +0.0014 | +0.0020 | [-0.0034, +0.0073] |
| test (official) | +0.0022 | +0.0078 | +0.0089 | +0.0063 (C0 std 0.0034) | -- |

- (a) passes. The val_select gain of +0.0079 is more than three times the C0 seed std.
- (b) passes the sign check: val_confirm moves the same way.
- (b) fails the CI check: the val_confirm lower bound is -0.0034, below 0. Under the
  pre-registered rule, 3a v2 is therefore **not** reported as beating the baseline.
- Test (3 official submissions per method) agrees with val_select. 3a v2 is 0.4195
  (std 0.0024) against C0's 0.4132 (std 0.0034); its worst seed beats C0's best seed.
  Its IoU is +0.0120 higher and its Cor is -0.0010 lower. The thesis reports this as
  a consistent gain on two of three evaluations, with confirmation inconclusive on
  val_confirm (half of val, 1,510 records). The bar is not relaxed afterwards.

**Outcome, C1-b and D2c_R4 (2026-11-12): not promoted, so Part 2 ends.** C1-b's val_select
is 0.4213, -0.0097 vs dec2 C0 (CI [-0.0201, +0.0010]). It ran exactly 3,020 optimizer
steps (checked). D2c_R4 (candidates C4x-b, C1-b) is logged with chosen = none.

| val_select | Cor+Lbl | Labeled-record part | Clean-record part | Records flagged | Clean records flagged |
|---|---|---|---|---|---|
| C0 | 0.4310 | 0.1536 | 0.2774 | 575 | 22 |
| C1 | 0.4315 | -- | -- | -- | -- |
| C1-b | 0.4213 | -0.0118 | +0.0021 | 507 | 19 |

- C1-b's training set is 23,682 records (1.02 passes at 3,020 steps). Counting each copy
  as a separate exposure, it sees as many hallucinated and clean records as C0
  (about 17,290 and 6,870). What changes is where the hallucinated exposures go: the
  rare-category records upsampled by C1 take them from the common-category records,
  which C0 sees twice and C1-b about once.
- The result is under-flagging: 507 records flagged against 575. The labeled part loses
  -0.0118 and the clean part gains only +0.0021. At fixed compute, rarity upsampling
  costs more on common categories than it gains on rare ones. C1's earlier +0.0004 came
  from the extra steps, not from the redistribution.
- **Part 2 conclusion.** Five conditions, C1, C4, C4x, C4x-b and C1-b, failed to beat C0
  under the parser fix and the prior match. No synthetic or upsampled condition beats
  fine-tuning on the original data alone at this dose. Per R4, no further revision follows.

## R5: Decision 4, integration as run (2026-11-12, written before any F3/F4 score is seen)

The original plan fixes the candidates. Three things have changed since it was written,
so this section records how each candidate is built now. Nothing else is tried.

- **F1 = C0, greedy (dec2), all 3 seeds.** No Part 2 condition was promoted (D2, D2b,
  D2c_R4), so "Part-2 best" is the original-data baseline itself.
- **F2 = 3a v2, exactly as decided in D3_3a2_qwen_seed42.** Votes are per label (R3). The
  voters are 8 samples (T=0.7) plus F1's greedy prediction, and k=3. Check: F2 must
  reproduce the 3a v2 val_select scores (0.4404 / 0.4352 / 0.4344).
- **F3 = F2's voters plus the 3b v2 multilabel CRF probe (crfml) of the same seed, as ONE
  more voter.** That makes 10 voters, with the same per-label vote and the same k=3 (the
  plan says "same k"; k is not re-chosen). Seed s of the probe pairs with adapter seed s.
- **F4 = the 4-backbone committee from D1, seed 42.** The members are Qwen3-VL-8B (dec2),
  Gemma-3-12B, InternVL3-8B and Pixtral-12B, with a per-label vote and k = ceil(4/2) = 2.
  Limitation: the three non-Qwen backbones kept no raw generations, so they appear with
  the old decoder. Re-predicting them is not worth several GPU-days. F4 has one seed and,
  as planned, is reported separately.

**Rule.**
- D4 picks the highest mean val_select Cor+Lbl over seeds; F4 enters with its one
  seed. The decision is logged once.
- val_confirm is attached for the chosen system.
- If F3 is chosen, its claim against C0 uses the same 3-seed "beats baseline" rule as
  3a v2.
- Its test files, one per seed, are written for the user to submit.
- If F2 is chosen, its 3 test results already exist (0.4195 mean), and nothing new is
  submitted.
- If F4 is chosen, it is still compared against F1 on seed 42 only. It cannot make a
  3-seed claim.

**Outcome, Decision 4 (D4_final, 2026-11-12): F3 is chosen.** F2 reproduced 3a v2
record for record on all three seeds (checked before D4 was logged).

| val_select Cor+Lbl | Seed 42 | Seed 123 | Seed 2024 | Mean |
|---|---|---|---|---|
| F1 = C0 | 0.4310 | 0.4289 | 0.4264 | 0.4288 |
| F2 = 3a v2 | 0.4404 | 0.4352 | 0.4344 | 0.4367 |
| **F3 = F2 + crfml probe voter** | **0.4436** | **0.4402** | **0.4372** | **0.4404** |
| F4 = 4-backbone committee | 0.4355 | -- | -- | 0.4355 |

3-seed "beats baseline" check, F3 against C0 (95% CI from records averaged over seeds):

| | Mean gain | CI | Per seed |
|---|---|---|---|
| val_select | +0.0116 (C0 std 0.0023) | [+0.0061, +0.0171] | +0.0126 / +0.0114 / +0.0109 |
| val_confirm | +0.0029 | [-0.0031, +0.0090] | +0.0059 / +0.0007 / +0.0020 |

- (a) passes, and (b) passes the sign check.
- (b) fails the CI check: the lower bound is -0.0031. As with 3a v2, the strict rule
  is not met.
- What the probe adds over F2: on val_select +0.0037 [+0.0015, +0.0059], positive on
  every seed; on val_confirm +0.0009 [-0.0015, +0.0034].
- F4 (one seed, old decoder for three of its members) is +0.0045 over Qwen seed 42 alone
  and below F2 and F3. It is reported separately, as planned.
- Test files for F3, one per seed (format-checked), are in
  `runs/integration/final/submission_F3_seed{42,123,2024}/`, for the user to submit.

## R6: graded per-character probabilities (post-hoc extension, 2026-11-12, written before any score)

**Why.** Cor and Cor+Lbl are a Spearman correlation between per-character predicted
and gold probabilities. Gold probability is the share of annotators who marked a
character, so it varies inside a span (core vs edges). All our voting outputs average
prob over each span (`voting.py`), so every character of a span gets the same value.
The scorer sees ties and the within-span ranking is lost. F3's test gain came almost
entirely from IoU (+0.018) while Cor stayed flat, and Cor is where we trail the top
teams (EN Cor 0.44 vs 0.55). This extension is decided after D4 and is reported as
post-hoc.

**Candidates (3; same voters, same per-label vote and k=3 as F3, so the kept characters
and IoU are identical. Only the prob values change):**
- G0: F3 as is. Span prob = mean over the span of prob_L(c) = (1/N) sum_i flag_i(c, L) prob_i(c, L).
- G1: graded F3. prob_L(c) is kept per character, rounded to 2 decimals, and consecutive
  characters with equal rounded prob form one output span.
- G2: graded vote fraction. prob_L(c) = votes_L(c) / N (N = 10 voters), likewise per
  character.

**Rule.**
- Decision D6_graded picks the highest mean val_select Cor+Lbl over the 3 F3 seeds and is
  logged once.
- A claim that the chosen variant improves on F3 uses the 3-seed "beats baseline" rule
  with F3 as the baseline (F3 seed std 0.0032), on both val halves.
- If G1 or G2 is chosen, its 3 test files are written for the user. Otherwise nothing is
  submitted.
- No further variants (other rounding, temperature, or calibration) follow from this
  result.

**Outcome, R6 (D6_graded, 2026-11-12): G1 is chosen.** Before logging, G0 was checked to
reproduce F3 exactly, and G1/G2 to keep exactly F3's characters (IoU unchanged).

| val_select Cor+Lbl (Cor) | Seed 42 | Seed 123 | Seed 2024 | Mean |
|---|---|---|---|---|
| G0 = F3 | 0.4436 (0.4898) | 0.4402 (0.4878) | 0.4372 (0.4804) | 0.4404 |
| **G1 graded prob** | **0.4463** (0.4945) | **0.4434** (0.4931) | **0.4401** (0.4876) | **0.4433** |
| G2 graded vote fraction | 0.4451 (0.4917) | 0.4424 (0.4929) | 0.4390 (0.4871) | 0.4422 |

| G1 vs ... (3 seeds, 95% CI from records averaged over seeds) | val_select | val_confirm |
|---|---|---|
| F3 | +0.0029 [+0.0017, +0.0041] | +0.0057 [+0.0040, +0.0076] |
| C0 | +0.0145 [+0.0089, +0.0199] (C0 std 0.0023) | +0.0086 [+0.0027, +0.0148] |

- **Against F3** (the R6 rule): the gain is positive on every seed and both halves, and
  both CIs are above 0. But (a) fails narrowly: the val_select gain of +0.0029 is just
  below F3's seed std (0.0032). By the letter of the rule, G1 does not "beat" F3. It is a
  small, highly consistent gain (paired per seed, since G1 changes only prob values).
- **Against C0**: G1 meets every part of the "beats baseline" rule. (a) +0.0145 is more
  than 6 times the C0 std. (b) val_confirm moves the same way, with the CI lower bound
  +0.0027 > 0. It is the first system in the program to pass the strict rule. Caveat
  for the write-up: R6 was decided after D4, so this is reported as a post-hoc
  extension of the final system, not as a pre-registered result.
- Test files for G1, one per seed (format-checked), are in
  `runs/integration/graded/submission_G1_seed{42,123,2024}/`.

## R7: second frozen judge for the probe, completing R1 (2026-11-15, written before any Gemma feature exists)

**Why.** R1 required a second probe judge from another family (Gemma-3-12B, 4-bit) because
the Part 1 winner is Qwen3-VL-8B, but it was not run before D4. The thesis lists this as
a limitation, and it explains the probe's small gain by correlated errors (record-level
Pearson 0.76 between judge and Qwen probe). R7 runs it now. Because D4 and D6 are already
made, any change to the final system is reported as post-hoc, like R6.

**What is fixed in advance.**
- Judge: `google/gemma-3-12b-it`, frozen, not fine-tuned, language model in 4-bit NF4
  (vision tower bf16), eager attention. Same plain VQA framing and the same per-record
  feature function as the Qwen probe (`extract_features_judge2.py`, which calls the old
  extractor's `process_record` unchanged): entropy, NLL, look-back ratio of every LM
  attention head (48 layers x 16 heads = 768), last-layer hidden state (3,840).
  Gemma 3 attends locally (1,024-token window) in 5 of every 6 layers. The look-back
  ratio is computed as is in every layer; nothing is dropped or re-weighted.
- Compaction: same PCA sizes (hidden 256, look-back 128) fitted on a 200k-token train
  sample only, same standardisation. Every train/val/test record must have features or an
  explicit skip (enforced).
- Probe: the multi-label BiGRU + 5-binary-CRF probe with the unchanged HP set, seeds
  42/123/2024, early stopping on val_select loss (`crf_probe --multilabel --source gemma3_12b`).

**Candidates for D7 (3, all on the per-seed G1 voters, per-label vote, k = 3 unchanged,
graded="prob"; seed s uses the judge, samples and probes of seed s):**
- H0: G1 as is (8 samples + greedy + Qwen probe, M = 10).
- H1: G1 + Gemma probe as an 11th voter (M = 11).
- H2: G1 with the Gemma probe in place of the Qwen probe (M = 10).

**Rule.**
- D7 picks the highest mean val_select Cor+Lbl over the 3 seeds and is logged once.
- A claim that H1 or H2 improves on G1 uses the "beats baseline" rule with G1 as the
  baseline (G1 seed std 0.0031): (a) mean val_select gain > 0.0031, (b) same sign on
  val_confirm, (c) val_confirm paired-bootstrap CI above 0.
- Test files (3 seeds) are written only if H1 or H2 is chosen. If H0 is chosen, nothing
  is submitted and G1 stays the final system.
- No further variants follow from this result (no other k, no probe-only weighting, no
  other layers or judges).

**Diagnostics (reported, select nothing).**
- Gemma probe alone on val_select / val_confirm (3 seeds), next to the Qwen probe (0.3844).
- Record-level Pearson correlation on val_select between the judge (C0) and each probe's
  per-record Cor+Lbl, and between the two probes. Hypothesis H5 (stated now): the Gemma
  probe's correlation with the judge is below the Qwen probe's 0.76.
- Spearman over the 20 category-language cells between F2 and the Gemma probe (H4 had
  0.95 for the Qwen probe).

**Note before any Gemma result (2026-11-15).** A smoke test of `integration/r7.py` with
the Qwen probe standing in for the Gemma probe (not a candidate, selects nothing) checked
that H0 reproduces G1 exactly. In that stand-in, counting the Qwen probe twice gave seed-42
val_select 0.4471 against 0.4463 for H0. So part of any H1 gain can come from giving
probe votes more weight alone; H1 is read as evidence of new information only to the
extent that it exceeds this.
