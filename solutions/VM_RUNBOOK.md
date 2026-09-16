# Running these solutions on the VM (`nlp1`)

This code was written and locally smoke-tested on a laptop with a 6 GB GPU;
it's meant to be transferred to and actually trained on the VM (376 GB RAM,
5 GPUs: three RTX 3090 Ti @ 24 GB — indices `0`, `3`, `5` — and two RTX 2080
Ti @ 11 GB — indices `1`, `2`). **Device index `4` errors out** on that
machine (`Unable to determine the device handle for GPU4`) — never pass
`--gpu_index 4`.

## 1. Transfer

Copy (or `git`-sync) the whole repo, including `data/`, `shroom-vis-images/`,
and `participant_kit/` — `solutions/common/scorer_local.py` imports
`participant_kit/scorer.py` directly by relative path, so the directory
layout must stay intact.

## 2. Environment setup

A plain venv with pip installs is enough (no conda required). From the repo
root on the VM:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# transformers: solutions 02-06 default to Qwen3-VL-30B-A3B-Instruct, whose
# code is NOT in any released transformers version as of this writing --
# its own model card says to build from source. Solution 01 (XLM-R + CLIP)
# works fine on this same install too, so one bleeding-edge transformers
# covers everything. This is less pinned/reproducible than a version-locked
# install -- once it works, consider `pip freeze > requirements.lock.txt`
# so you can recreate the exact working environment later.
pip install "git+https://github.com/huggingface/transformers"
pip install "huggingface-hub<1.0" "accelerate>=0.34" "peft>=0.13" sentencepiece

# Solution-specific extras (qwen-vl-utils is NOT needed -- the Qwen3-VL code
# path uses AutoProcessor's own apply_chat_template directly)
pip install bitsandbytes anthropic easyocr sacremoses

# torch: install the CUDA build matching the VM's driver (580.126.09 / CUDA 13.0
# supports any recent cu12x wheel — the driver is backward compatible)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

**If you'd rather stay on Qwen2-VL-2B-Instruct** (the originally-planned, fully
locally-verified default before this model swap): pin `transformers==4.46.3`
instead of the source install above, add back `pip install qwen-vl-utils`,
and pass `--model_id Qwen/Qwen2-VL-2B-Instruct` to both `run_pipeline.py`
scripts — the code still supports it via the `AutoModelForImageTextToText`
fallback path, just untested against this specific model card's newer
`apply_chat_template` flow (see solution 02's README for the caveat).

Verify CUDA is visible and pick a safe GPU before running anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi   # confirm which of 0,1,2,3,5 are free right now -- this is a shared machine
```

## 3. Sanity check the scorer first

Before trusting any solution's `val_scores.json`, confirm the local scorer
wrapper matches the official one:

```bash
python -m solutions.common.scorer_local
# expect: IoU=1.0000, CorLbl=1.0000, Cor≈0.99 (see the docstring/comment in
# scorer_local.py for why overall Cor isn't exactly 1.0 on identity input)
```

## 4. Run jobs so they survive disconnecting

Every solution here can run for hours (VLM training, or the ~2,054-image
EasyOCR pass shared by solutions 03-06 — measured in practice at 25% after 90+ minutes,
so budget several hours, slower than the original rough estimate). Don't run
these directly over a plain SSH session — use `tmux` (or `screen`) so the
job keeps running after you disconnect, and you can reattach later to the
exact live view:

```bash
tmux new -s <name>              # e.g. tmux new -s sol02
# ... run your commands normally inside this session ...
# detach without stopping anything: Ctrl+B, then D
```

Reconnect later with:
```bash
tmux attach -t <name>           # or: tmux a -t <name>
tmux ls                         # list your sessions if you forget the name
```
While attached, `Ctrl+B` then `[` scrolls back through output (`q` to exit).
If `tmux` isn't available, `screen -S <name>` / `Ctrl+A` `D` / `screen -r <name>`
works the same way. As a last resort with no reattach capability: `nohup
<command> > run.log 2>&1 & disown`, then `tail -f run.log` after reconnecting.

Either way, the actual deliverables (`val_predictions.jsonl`, `val_scores.json`,
`checkpoint/`) land on disk as soon as a `run_pipeline.py` finishes — they
exist independently of whether you're attached to any session, so you can
just check the files after reconnecting instead of reattaching at all.

## 5. Run each solution

Each solution is independent and can run on a different GPU in parallel
(this is a shared, multi-GPU machine — check `nvidia-smi` first so you don't
collide with someone else's job).

**Important — verification level differs by model.** The `--dry_run` data
pipeline (conversation building, chat templating, image processing, label
masking) was verified locally, but only against the *original* default,
Qwen2-VL-2B-Instruct — the dev laptop's `transformers` install can't even
import the Qwen3-VL model classes (they don't exist in any released
version yet), so **the current default, Qwen3-VL-8B-Instruct (and the
bigger Qwen3-VL-30B-A3B-Instruct alternative), have had zero local testing
of any kind** — not `--dry_run`, not `--smoke_test`, nothing. The code was
written directly against Qwen3-VL's model-card usage example (fetched and
quoted in solution 02's README), not verified by running it. **Run
`--dry_run` then `--smoke_test` for 02 and 03 here first**, in that order,
before committing to a full run — this is the actual first real test of
this code path.

### 01 — Lightweight tagger (cheapest, ~20-40 min even on a 2080 Ti)

```bash
cd solutions/01_lightweight_tagger
python run_pipeline.py --gpu_index 1 --epochs 3 --batch_size 16
```
CLIP is frozen and only runs once (image features are cached to
`image_cache.pt`), so this is XLM-R-base-sized training -- fine on any of
the 5 GPUs, including the 11 GB cards.

### 02 — VLM LoRA fine-tune, real data only (reference point)

Run in this exact order — cheapest check first, real-but-tiny second, full
job last. `--model_id` is left out below so the commands rely on
`train_lora.py`'s default (`Qwen/Qwen3-VL-8B-Instruct`); pass it explicitly
if you ever want to be unambiguous about which model ran:

```bash
cd solutions/02_vlm_lora_real_data
python train_lora.py --dry_run
python run_pipeline.py --smoke_test --gpu_index 0
python run_pipeline.py --gpu_index 0 --epochs 2 --grad_accum 8
```
**Qwen3-VL-8B-Instruct** (dense, ~9B params including the vision tower,
~18 GB in bf16 / ~4.5 GB in 4-bit) is the default and **fits any of the
VM's 5 GPUs** in 4-bit, including the 11 GB 2080 Tis (`--gpu_index 1` or
`2`), and fits a 24 GB 3090 Ti even unquantized (`--no_4bit`).

For the bigger, sparse-MoE alternative, pass `--model_id
Qwen/Qwen3-VL-30B-A3B-Instruct` on every command above (31B total params,
~3B active per token — trains at roughly dense-3B compute cost, but its
weights are ~62 GB in bf16 / ~15.5 GB in 4-bit). **Only use `--gpu_index
0`, `3`, or `5`** (the three 24 GB 3090 Tis) — the 4-bit weights alone
don't fit an 11 GB 2080 Ti. Do not pass `--no_4bit` on a single GPU (needs
~62 GB unquantized).

### 03 / 04 — VLM LoRA fine-tune, synthetic augmented, r_max=8 (main "push for highest results" run)

Solutions 03 and 04 share identical methods (a) upsampling and (b) grounded
heuristic injection at the original r_max=8 rarity cap; the only difference
is method (c), LLM-assisted injection, which solution 03 always skips and
solution 04 always includes (whenever `ANTHROPIC_API_KEY` is set). Run
whichever one you want — or both, in separate checkouts/sessions, since
they write to entirely separate folders:

```bash
cd solutions/04_vlm_lora_synthetic_augmented_llm   # or solutions/03_vlm_lora_synthetic_augmented for the no-LLM variant

# For solution 04 only: set a FRESH Anthropic key (never reuse the one that
# was found hardcoded in last_experiment/S1b -- that has been scrubbed and
# should be rotated) to include LLM-assisted injection for the rarest
# classes. If unset, methods (a) upsampling and (b) grounded heuristic
# injection still run -- only the optional (c) step is skipped. Solution 03
# never uses this at all (its run_pipeline.py always passes --skip_llm).
export ANTHROPIC_API_KEY=sk-ant-...          # set this yourself; do not paste it into any chat

# 1. Smoke-test the pipeline -- deliberately WITHOUT --skip_ocr_extract this
#    time. An earlier version of this runbook passed --skip_ocr_extract here,
#    which (on a fresh VM, no cache file yet) skips EasyOCR entirely rather
#    than reusing anything -- the OCR path would then run for the very first
#    time during the ~90-minute full-scale pass below, the worst possible
#    place to discover an EasyOCR bug (one was found and fixed during
#    development: it hard-errors combining ch_sim with fr/it in one reader).
#    Without the flag, --smoke_test caps the real EasyOCR pass at 15 images
#    (~1-2 min), so this actually exercises that path first.
python run_pipeline.py --smoke_test --gpu_index 0   # solution 03 always skips LLM on its own;
                                                      # pass --skip_llm too if running this from
                                                      # solution 04's folder without a key

# 2. Full run. First call builds the real OCR cache (~2,054 unique training
#    images) -- saved to ocr_cache.json, any re-run automatically reuses it
#    (no flag needed; generate_synthetic.py checks whether the file already
#    exists, and either solution's cache can be copied to the other's
#    folder to skip rebuilding it twice). Budget several hours for this
#    pass, not ~90 minutes: an actual full run (not the 15-image smoke
#    test) measured 25% done after 90+ minutes with wide per-image variance
#    (some images took 30+ seconds) -- run it in tmux (see section 4
#    above), don't wait on it interactively.
python run_pipeline.py --gpu_index 0 --epochs 2 --grad_accum 8
```
Same model/GPU guidance as solution 02 — default is `Qwen/Qwen3-VL-8B-Instruct`,
which fits any of the 5 GPUs in 4-bit including the 2080 Tis; pass
`--model_id Qwen/Qwen3-VL-30B-A3B-Instruct` on both commands above for the
bigger MoE alternative, restricted to `--gpu_index 0`, `3`, or `5`. Solution
04's LLM injection step (method c) caches incrementally to
`llm_synth_cache.json`, so an interrupted run resumes instead of
re-spending API budget.

Solutions 05 and 06 are the same pair with the rarity cap tuned down to
r_max=3 instead of 8 (`solutions/05_vlm_lora_synthetic_tuned` and
`solutions/06_vlm_lora_synthetic_tuned_llm` respectively) — same commands,
different folder; see each solution's `generate_synthetic.py` docstring for
why that cap was lowered.

## 6. What you get back

Each solution folder ends up with:
- `val_predictions.jsonl` — submission-format predictions on `data/val.jsonl`
- `val_scores.json` — Span IoU / Spearman (overall + per-label + per-language),
  computed via the *official* `participant_kit/scorer.py` functions (not a
  reimplementation)

Test-set prediction is intentionally not produced by any of these pipelines
-- only val, so you can inspect/compare before deciding which approach (or
combination) to run on `data/test.jsonl` for the actual submission.

## 7. Comparing results

Once you have `val_scores.json` for whichever solutions you ran, fill in
`solutions/RESULTS_SUMMARY.md` (a table template is already there) to decide
which one to carry forward.
