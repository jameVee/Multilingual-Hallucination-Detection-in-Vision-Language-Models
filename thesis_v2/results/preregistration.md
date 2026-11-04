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
