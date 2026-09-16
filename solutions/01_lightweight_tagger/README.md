# Solution 01 — Lightweight Multimodal Span Tagger

## Method

A custom, cheap-to-train multimodal token classifier, not a VLM fine-tune:

- **Image**: frozen CLIP ViT-B/32 (`openai/clip-vit-base-patch32`) patch-grid
  features, precomputed once for every unique image and cached to
  `image_cache.pt` (`solutions/common/image_cache.py`) — CLIP never runs
  during training.
- **Text**: trainable XLM-RoBERTa-base (`xlm-roberta-base`), which natively
  covers all 4 dataset languages (en/fr/it/zh), tokenizing `(prompt,
  response)` as a pair so the model sees what was asked.
- **Fusion**: 2 lightweight cross-attention blocks where response-token
  hidden states attend to the CLIP patch tokens.
- **Head**: per-token 5-way **sigmoid** (multi-label, not softmax) — gold
  spans do overlap with different labels on the same characters in this
  dataset (annotator disagreement), which a single-label softmax can't
  represent.
- **Loss**: class-weighted BCE against the *soft* gold probability per
  token (not binarized), so the model is trained to output calibrated
  probabilities directly — matching what the Spearman-correlation half of
  the official metric rewards.
- **Decoding**: per-label thresholding (rare classes default to a lower
  threshold, favoring recall — see `DEFAULT_THRESHOLDS` in `model.py`) +
  contiguous-token merging → character spans.

No synthetic data — this solution trains on `data/train.jsonl` as released.
It exists as a fast, architecturally-independent reference point next to the
VLM LoRA fine-tunes (solutions 02/03), and is cheap enough to run start-to-
finish on any of the VM's 5 GPUs (even the 11 GB 2080 Tis) in well under an
hour.

## Files

- `model.py` — architecture + `decode_spans` (token probs → char spans)
- `dataset.py` — tokenization + per-token soft-label target construction
- `train.py` — training loop (class-weighted BCE, AdamW, linear warmup/decay)
- `predict.py` — checkpoint → `val_predictions.jsonl`
- `run_pipeline.py` — builds the image cache if missing, then train → predict → score

## Usage

```bash
python run_pipeline.py --gpu_index 0 --epochs 3 --batch_size 16
python run_pipeline.py --smoke_test --gpu_index 0   # tiny sanity run
```

## Results

Fill in after running `run_pipeline.py` on the VM (values come straight from
`val_scores.json`):

| | Overall IoU | Overall Cor | Overall CorLbl | OCR Cor | miscounting Cor | other Cor |
|---|---|---|---|---|---|---|
| val | _pending VM run_ | | | | | |

Per-language and full per-label breakdowns are in `val_scores.json` directly.
