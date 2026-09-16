# Experiment Plan: Model-Agnostic, Uncertainty-Based Hallucination Detection

**Date**: 2026-06-25

---

## Data

Official shared task split, downloaded to `data/`:

| File | N | Languages | Labeled? |
|---|---|---|---|
| `data/train.jsonl` | 12,082 | en=3070, fr=3013, it=2989, zh=3010 | Yes (8,646 hallucinated / 3,436 clean) |
| `data/val.jsonl` | 3,020 | en=729, fr=754, it=757, zh=780 | Yes (2,161 hallucinated / 859 clean) |
| `data/test.jsonl` | 4,898 | en=1201, fr=1233, it=1254, zh=1210 | No (held out) |

Schema: `{id, split, language, prompt, image_name, response, labels}`. `labels` is a list of
`{start, end, prob, label}`; `prob ∈ {0.25, 0.333, 0.5, 0.667, 0.75, 1.0}` (annotator-agreement
fraction, not binary). `image_name` matches files already in `shroom-vis-images/`.

Category distribution (train): invention=14,473, mischaracterization=13,609, miscounting=2,747,
other=1,480, **OCR=1,284 (smallest)**.

A real `val` split now exists — use it for all threshold/layer/hyperparameter tuning; `test` stays
untouched until final submission.

---

## Experiment List

Full system scope throughout: **all 5 categories × all 4 languages**, using the full `data/train.jsonl`
(12,082 samples) and `data/val.jsonl` (3,020 samples). Ordered by increasing feature complexity; each
gates the next.

| # | Name | Signal | Probe training | Compute | Status |
|---|---|---|---|---|---|
| 0 | Fine-tuning baseline (reference) | — | Full LoRA SFT | Already run | Done (prior track) |
| 1 | Token-level entropy / perplexity | Judge VLM's next-token distribution | None or 2-param logistic regression | Low | Not started |
| 2 | Frozen hidden-state linear probe | Residual-stream hidden states at answer tokens | Per-category logistic regression | Medium | Not started |
| 3 | Attention lookback-ratio probe | Per-head attn ratio (image vs. prior-answer tokens) | Per-category logistic regression | Medium | Not started |
| 4 | Ensemble of 1+2+3 | Concatenated features | Per-category logistic regression | Medium | Gated on 1–3 |
| 5 | Span-level PAS | Whichever signal(s) validate, at span granularity | Same as winning signal | Medium–High | Gated on 1–4 |
| 6 (stretch) | Cross-model / multi-judge ensemble | Multiple frozen LVLMs | Per-category logistic regression | High | Gated on 5 |

---

### Experiment 1 — Token-level entropy / perplexity

- **Setup**: teacher-force image + prompt + response through frozen judge VLM (Qwen3-VL-8B). At each
  response token, take `outputs.logits` → next-token distribution.
- **Features**: per token, `H_t = -Σ p(v) log p(v)` (entropy) and `-log p(token_t)` (NLL).
- **Probe**: raw threshold (zero-shot) or 2-feature logistic regression calibrated on train, tuned on val.
- **Scope**: all 5 categories × 4 languages, run together (categories share the same forward pass; only
  the positive/negative token labels differ per category).
- **Span construction**: merge contiguous above-threshold tokens → char spans (shared logic with Exp 2).
- **Eval**: Span IoU / Spearman ρ / Category F1 on val, reported per language and per category (5×4 grid).
- **Compute**: shares the same forward-pass infrastructure as Exp 2 — capture `logits` instead of/alongside
  `hidden_states`. No extra model load if built together with Exp 2.

### Experiment 2 — Frozen hidden-state linear probe

- **Setup**: frozen Qwen3-VL-8B, `output_hidden_states=True`, extract hidden states at answer-token
  positions for every sample in `data/train.jsonl` + `data/val.jsonl` (12,082 + 3,020), align gold char
  spans to tokens.
- **Probe**: per-category logistic regression, swept across ~4 evenly-spaced candidate layers, best
  picked by val ROC-AUC.
- **Scope**: all 5 categories × 4 languages. Default design: **one probe per (category, language) pair**
  (20 small logistic regressions) so each language's hidden-state geometry is fit separately — but train
  a **shared multilingual probe per category as a fallback/ablation** for any (category, language) cell
  with too few positive train spans to fit a stable probe (OCR is the risk case: only 1,284 train spans
  total across 4 languages, ≈ 320/language before token-level expansion).
- **Output**: val ROC-AUC reported as a 5×4 grid (category × language), so weak cells are visible rather
  than averaged away.
- **Compute note**: full-dataset extraction (~15k forward passes through an 8B model, image+text each)
  is the heaviest single step in this plan — budget accordingly, cache features to disk once extracted so
  Experiments 1/3/4 can reuse them without re-running the model.

### Experiment 3 — Attention lookback-ratio probe

- **Setup**: same forward pass as Exp 2, capture `output_attentions=True` instead.
- **Features**: per answer token, per head, `lookback_ratio = attn_to_image / (attn_to_image +
  attn_to_prior_answer_tokens)`. Concatenate across heads/layers.
- **Scope**: same 5 categories × 4 languages, same per-(category, language) probe design as Exp 2.
- **Probe**: same logistic regression design as Exp 2.
- **Compute caveat**: `output_attentions=True` stores full seq_len × seq_len matrices per head/layer —
  much heavier than hidden states for long image-token sequences, and at full dataset scale this is the
  most memory-sensitive step in the plan. May need to subset to image-token columns only, process in
  smaller batches, or use a smaller judge model.

### Experiment 4 — Ensemble of 1+2+3

- **Gate**: only run if ≥2 of Experiments 1–3 individually show val AUC clearly above 0.5 in a
  meaningful number of (category, language) cells.
- **Setup**: concatenate Exp 1's 2 scalars + Exp 2's best-layer hidden state + Exp 3's attention ratios →
  one feature vector per token, retrain per-(category, language) probe.

### Experiment 5 — Span-level system integration

- **Gate**: explicit go/no-go on Experiments 1–4 showing real signal.
- **Setup**: Experiments 1–4 are already full-scope (5×4), so this step is not a scale-up — it's
  **consolidation**: per (category, language) cell, pick whichever signal (entropy, hidden-state,
  attention, or ensemble) had the best val AUC, lock in its tuned threshold, and assemble one unified
  inference function that runs all winning per-category probes on a single response and merges their
  outputs into one span list (handling any overlaps between categories).
- **Eval**: same `span_iou` / `spearman_rho` / `category_f1` functions, directly comparable to Exp 0
  (fine-tuning track).

### Experiment 6 (stretch) — Cross-model / multi-judge ensemble

- **Gate**: after Experiment 5 works, only if pushing further is worth it.
- **Setup**: add a second frozen judge (e.g. LLaVA) alongside Qwen3-VL-8B, concatenate or ensemble their
  uncertainty signals — same spirit as the fine-tuning track's S5 Multi-LVLM Fusion.

---

## Recommended order

```
1. Implement Exp 2 (hidden-state probe, full 5×4 scope) on data/{train,val}.jsonl   (no blocker — data ready)
2. Implement Exp 1 (entropy/perplexity) — shares Exp 2's forward pass, cheap add-on
3. Read AUC grids from 1 & 2 → decide on Exp 3 (attention ratios)
4. If signal exists in ≥2 of 1–3 → Exp 4 (ensemble)
5. If validated → Exp 5 (span-level system integration)
6. Stretch: Exp 6 (multi-judge ensemble)
```

Gates are explicit go/no-go, not a fixed calendar — if Exp 1 alone shows strong signal across most
(category, language) cells, skip straight to Exp 5 integration instead of building 2/3.
