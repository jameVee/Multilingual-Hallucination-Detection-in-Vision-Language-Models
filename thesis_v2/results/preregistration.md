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
