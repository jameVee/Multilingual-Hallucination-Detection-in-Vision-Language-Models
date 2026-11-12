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
