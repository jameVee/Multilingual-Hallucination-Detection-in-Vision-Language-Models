# Solution 02 — VLM LoRA Fine-tune, Real Data Only

## Method

LoRA fine-tune of **Qwen3-VL-8B-Instruct** by default (standard Hugging
Face stack — `transformers` + `peft` + `bitsandbytes`, not `unsloth`, for
portability across a VM setup this wasn't developed on) to directly
generate the hallucination span-list JSON for a given `(image, prompt,
response)` triple. Trained on `data/train.jsonl` only — no synthetic
augmentation. This is the "real data only" reference point for the
VLM-fine-tune approach, redone cleanly against the new train/val split and
scored through the *official* `participant_kit/scorer.py` (the prior
`last_experiment/` runs used different, self-rolled metrics that aren't
directly comparable to this).

### Base model: Qwen3-VL-8B-Instruct (dense), Qwen3-VL-30B-A3B-Instruct (MoE) also supported

**8B-Instruct is the default** — dense, ~9B params including the vision
tower, ~18 GB in bf16 / ~4.5 GB in 4-bit. Fits **any of the VM's 5 GPUs**
in 4-bit, including the 11 GB 2080 Tis, and fits a 24 GB 3090 Ti even
unquantized (`--no_4bit`).

Pass `--model_id Qwen/Qwen3-VL-30B-A3B-Instruct` for the bigger,
sparse-MoE alternative (31B total params, ~3B active per token — trains at
roughly dense-3B compute cost, but its weights are ~62 GB in bf16 / ~15.5 GB
in 4-bit, so it needs a 24 GB 3090 Ti specifically — `--gpu_index 0`, `3`,
or `5` only, and never `--no_4bit` on a single GPU). `resolve_model_class`/
`resolve_target_modules` in `train_lora.py` branch automatically on
MoE ("A3B"/"Moe" in the model id) vs. dense, so switching `--model_id` is
the only change needed — no code edits either way.

**This is a newer, less-tested integration than the rest of this repo** —
Qwen3-VL's own model card says its code isn't in any released
`transformers` version yet (`pip install git+https://github.com/huggingface/transformers`
required), so this was written directly against the documented usage
pattern, not verified by running it (the dev machine's `transformers`
can't even import the model class, for either model size). The verified
model-card example this code follows (shown here for the 30B-A3B/MoE class
name; the 8B default uses `Qwen3VLForConditionalGeneration` instead, see
below):

```python
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

messages = [{"role": "user", "content": [
    {"type": "image", "image": "<url or path>"},
    {"type": "text", "text": "Describe this image."},
]}]
inputs = processor.apply_chat_template(
    messages, tokenize=True, add_generation_prompt=True,
    return_dict=True, return_tensors="pt",
)
generated_ids = model.generate(**inputs, max_new_tokens=128)
```

This code does **not** actually use that one-call `apply_chat_template`
flow above, despite it being the documented example — see "the image-drop
bug" below. Integration details the model swap(s) drove:

- **No `qwen_vl_utils` dependency, but also not the one-call flow shown
  above** — tested locally (against Qwen2-VL-2B, the largest model the dev
  machine could download) that `apply_chat_template(tokenize=True,
  return_dict=True)` silently returns `input_ids`/`attention_mask` with
  **no `pixel_values`** — the image is dropped with no error, which would
  train a "VLM" on text alone. `make_collate_fn()` in `train_lora.py` (and
  the equivalent in `predict.py`) instead loads the image with `PIL.Image`
  directly and calls `processor(text=, images=)` — the older, more
  foundational two-step pattern, verified locally to reliably include
  `pixel_values`. A hard assertion checks for `pixel_values` right after
  every encode call, so if this ever regresses on the VM's transformers
  build, it fails loudly at `--dry_run` instead of training silently
  without images.
- **Model class varies by checkpoint**: `Qwen3VLForConditionalGeneration`
  for dense checkpoints (the 8B default), `Qwen3VLMoeForConditionalGeneration`
  for MoE ("A3B") ones — `resolve_model_class()` picks the right one from
  `--model_id`, falling back to the generic `AutoModelForImageTextToText`
  for anything else (e.g. if you point `--model_id` at Qwen2-VL).
- **LoRA target modules — attention only for MoE checkpoints, not
  "all-linear"**: an earlier version of this code used peft's
  `target_modules="all-linear"` for MoE checkpoints, reasoning that expert
  layer names weren't inspectable without a 62 GB download. That was wrong,
  caught by actually computing the numbers from the 30B-A3B model's real
  `config.json` (128 experts x 48 layers): "all-linear" (and, equally, the
  dense model's `q/k/v/o/gate/up/down` list, since peft matches by name
  suffix and every expert's projections share those same suffixes) ends up
  applying LoRA to **all 128 experts on every layer** — ~843M trainable
  params at rank 16, versus ~12.6M for attention-only. That adds roughly
  10 GB of gradient + AdamW optimizer-state overhead on top of the
  ~15.5 GB of 4-bit base weights, likely enough to OOM a 24 GB GPU once
  activations are added. `resolve_target_modules()` now defaults MoE
  checkpoints to attention-only LoRA (`q_proj`/`k_proj`/`v_proj`/`o_proj`).
  This doesn't affect the 8B default at all — dense checkpoints keep the
  fuller `q/k/v/o/gate/up/down` list, which is standard and safe (no
  per-expert multiplication to worry about).
- **Gradient checkpointing on by default** (`--gradient_checkpointing`,
  new flag) — added as a safety margin because this repo has no measured
  activation-memory profile for a real forward/backward pass on either
  model size at real image-token counts (never ran on real hardware
  locally). Trades ~20-30% more compute time for substantially lower peak
  activation memory. Turn off with `--no_gradient_checkpointing` once
  you've confirmed there's headroom to spare and want the speed.

### Everything else (unchanged from the original design)

- 4-bit QLoRA by default (`bnb_4bit_quant_type="nf4"`); pass `--no_4bit` for
  plain-precision LoRA (realistic for the 8B default on a 24 GB card; for
  the 30B-A3B alternative only with multi-GPU sharding).
- Compute dtype auto-selects bf16 on Ampere+ (3090 Ti) or fp16 on Turing
  (2080 Ti) per-GPU (`best_dtype()` in `train_lora.py`).
- Image resolution is capped (`--max_pixels`, default ~1024 patches) —
  uncapped, some dataset images blow up to 30k+ visual tokens, which risked
  OOM even on a 24 GB card during local testing (with Qwen2-VL).
- Loss is only computed on the assistant's response tokens (prompt tokens
  masked to `-100`), standard SFT masking.
- At inference, generated text is parsed back to character offsets via
  `solutions/common/vlm_io.py`, which includes a sequential-cursor fuzzy
  span locator (fixed during development: a naive first-occurrence
  `.find()` mis-locates short/common spans like a bare number or "her" when
  that text also appears earlier in a long response — exactly the failure
  mode that would have hurt the OCR/miscounting classes most).

## Files

- `train_lora.py` — LoRA fine-tune; also used by solutions 03-06 via `--train_file`
- `predict.py` — checkpoint → `val_predictions.jsonl`
- `run_pipeline.py` — train → predict → score

## Usage

```bash
python train_lora.py --dry_run                        # data pipeline only, no model download
python run_pipeline.py --smoke_test --gpu_index 0      # tiny real run -- do this before a full one
python run_pipeline.py --gpu_index 0 --epochs 2 --grad_accum 8

# bigger MoE alternative (24GB 3090 Ti only -- gpu_index 0, 3, or 5):
python run_pipeline.py --model_id Qwen/Qwen3-VL-30B-A3B-Instruct --gpu_index 0
```

## Local verification note

Developed on a laptop with a slow/metered connection to Hugging Face (a
measured ~50 KB/s during development). The `--dry_run` data pipeline
(conversation building, chat templating, image processing, label masking)
was verified locally, but only against the **original** default,
Qwen2-VL-2B-Instruct, before the model was switched to Qwen3-VL — **neither
Qwen3-VL size (8B or 30B-A3B) has had any local execution**, not even
`--dry_run`, since the local `transformers` install predates Qwen3-VL's
model classes entirely. The code above follows the model card's documented
usage exactly, but **run `--dry_run` then `--smoke_test` on the VM first**
(genuinely the first real test of this path) before committing to a full
run.

## Results

Fill in after running on the VM:

| | Overall IoU | Overall Cor | Overall CorLbl |
|---|---|---|---|
| val | _pending VM run_ | | |

For comparison, the prior (differently-split, differently-scored,
different-model) `last_experiment` real-data-only baseline reported
(self-rolled metrics, `comparison_vs_baseline_S4_AgreementUpsampling.csv`):
Span IoU 0.3102 / Spearman 0.3500 / Cat F1 0.2861 — treat as a rough
historical reference point only, not an apples-to-apples number.
