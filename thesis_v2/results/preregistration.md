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
