# Solution 07 — Model-agnostic uncertainty quantification (Aim 3)

Implements Experiments 1-3 of the thesis proposal's Appendix A
(Uncertainty-Quantification Experiment Plan), Phase 2 of the work plan.
Unlike solutions 02-06, **nothing here is fine-tuned** — every signal comes
from a frozen `Qwen/Qwen3-VL-8B-Instruct`, used exactly as released.

| # | Experiment | Script |
|---|---|---|
| 1 | Token-level entropy / perplexity | `extract_features.py` |
| 2 | Frozen hidden-state linear probe | `extract_features.py` (features) + `train_probes.py` (probe) |
| 3 | Attention lookback-ratio probe | `extract_features.py` (features) + `train_probes.py` (probe) |
| 4 | Ensemble of 1-3 | `train_probes.py` (gated: needs >=2 of 1-3 individually >= `--gate_auc`) |

## How it works

`extract_features.py` runs each labeled record through the frozen model
**once**, in a plain VQA framing — the model is given `(image, question)`
and the *real* response is teacher-forced in as if it were the model's own
answer (not the span-finding-JSON framing solutions 02-06 use). From that
single forward pass it extracts, per response token:

- **Entropy** of the model's own next-token distribution, and the
  **negative log-probability** it assigned the actual token (Experiment 1).
- The **hidden state** at one or more chosen decoder layers (Experiment 2;
  `--hidden_layers`, default just the last layer).
- Per attention head, per layer, the **lookback ratio**: fraction of that
  token's attention mass spent on image-token columns vs. on this turn's
  own prior response tokens (Experiment 3) — the multimodal analogue of
  Lookback Lens's context-vs-generation ratio, motivated by OPERA's finding
  that multimodal hallucination correlates with attention drifting away
  from the image.

`train_probes.py` then trains a logistic-regression probe per
`(category, language)` cell on each signal separately (5-fold CV AUC), with
a pooled multilingual fallback for cells with too few positive tokens
(e.g. OCR in any single language), and reports the AUC grid Hypothesis H3
is evaluated against.

## Why forward hooks, not `output_attentions=True` / `output_hidden_states=True` directly

Requesting attention weights or hidden states for *every* layer at the top
level makes the model class itself retain all layers simultaneously before
returning — for attention specifically, that's `num_layers * num_heads *
seq_len^2` elements held at once, which is exactly the OOM risk the thesis
proposal's Phase 2 contingency plan names ("subset attention to image-token
columns... if needed"). Both collectors here use `register_forward_hook`
instead: attention hooks reduce each layer's raw tensor down to a tiny
`(n_response_tokens, n_heads)` ratio *inside the hook* and then **replace**
the module's returned tensor with `None` (a forward hook's return value
replaces the module's output — standard `torch.nn.Module` behavior), so the
large tensor never propagates into the top-level model's own
`all_attentions` accumulation. The hidden-state hook only ever touches the
specific layer(s) requested. Net effect: peak memory is bounded by one
layer's raw attention tensor at a time, not all layers at once.

## Unverified assumptions — run `--dry_run` on nlp2 first

This was written without access to a downloaded Qwen3-VL checkpoint (no
local GPU with enough VRAM, and this dev machine's `transformers` install
is currently broken — see the note left in `Aim2_SyntheticVsNatural_Findings.md`'s
sibling conversation). Three things are logically designed but genuinely
unverified against the real model:

1. **`find_image_token_id()`** — tries `config.image_token_id` /
   `config.image_token_index` first (Qwen2-VL/2.5-VL/3-VL's documented
   convention at time of writing), then a couple of literal special-token
   strings. **Always check the printed `image_token_id = ...` line under
   `--dry_run`** — a wrong id here breaks Experiment 3 silently (no error,
   just meaningless ratios).
2. **Attention/decoder-layer module discovery** — `AttentionLookbackCollector`
   and `HiddenStateCollector` find modules by class-name pattern
   (`"attention" in type(m).__name__.lower()`, `"language_model" in name`;
   `type(m).__name__.endswith("DecoderLayer")`), not a hardcoded path,
   specifically so this doesn't silently break if Qwen3-VL's exact module
   tree differs from assumptions made without seeing it. Run with
   `--dry_run --verbose` and check the printed module list actually looks
   like decoder self-attention layers, not vision-tower internals.
3. **Token alignment** — `token_char_spans()` tokenizes `record["response"]`
   in isolation to get response-relative character offsets, then
   `verify_alignment()` checks those token ids match the response region of
   the real multimodal sequence. If they don't (possible BPE boundary
   difference right at the turn transition), it falls back to an
   approximate even-split alignment and sets `approx_alignment=True` for
   that record instead of crashing. **Check the dry-run output and the
   real run's printed "N records used approx_alignment" count** — if it's
   more than a handful, the token-to-label mapping needs a better fix
   before trusting probe results built on it.

## Data note

Both scripts skip clean (`labels: []`) records entirely — a clean record
contributes no positive tokens to any category, only more negative
(clean-token) examples, and `data/train.jsonl` already has 3,436 of those
if more negatives are ever needed (`data_io.load_jsonl` + a `labels`-empty
filter).

See `../NLP2_AIM3_RUNBOOK.md` for how to actually run this on the
university server.
