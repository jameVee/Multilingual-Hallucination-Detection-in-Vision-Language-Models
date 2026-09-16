# Running these solutions on the university server (`nlp2`)

This is a separate machine from `nlp1` (see `VM_RUNBOOK.md`) — different
GPU inventory, so it gets its own runbook rather than folding into that
one. Everything trained on this repo up to now (solutions 02-06) ran on
Google Colab Pro (A100 40GB); this doc covers moving the rest of that work
onto `nlp2`.

**Hardware**: 3 GPUs, all plain RTX 3090 @ 24 GB (not 3090 Ti — same VRAM,
slightly lower clocks), indices `0`, `1`, `2`. Driver 590.48.01, CUDA 13.1.
**This is a shared machine** — at the time of writing, GPU `0` is at 100%
util with 18.8 GB used by someone else's job; GPUs `1` and `2` are idle.
Always run `nvidia-smi` immediately before launching anything, not just
once at setup time — availability changes as other people start/stop jobs.

**A 3090 is meaningfully slower than the A100 Pro Colab used for every
prior run here** (lower memory bandwidth, older tensor cores) — don't
assume Colab wall-clock times carry over. Run a `--smoke_test` first and
read its per-step timing from the log to get a real nlp2-specific estimate
before scheduling a long full run.

## 1. Transfer

This repo is **not** a git repo, so use `rsync` over SSH rather than
`git clone`/`git pull`:

```bash
rsync -avz --progress \
  --exclude 'solutions/*/checkpoint/' \
  --exclude '**/__pycache__/' \
  "/path/to/local/Shroom-visions Shared Task/data" \
  "/path/to/local/Shroom-visions Shared Task/shroom-vis-images" \
  "/path/to/local/Shroom-visions Shared Task/participant_kit" \
  "/path/to/local/Shroom-visions Shared Task/solutions" \
  you@nlp2:/path/to/remote/Shroom-visions-Shared-Task/
```

`--exclude 'solutions/*/checkpoint/'` skips the already-trained Colab
adapters (~180 MB each, add nothing to new nlp2 runs). **Do not exclude**
`solutions/03_vlm_lora_synthetic_augmented/ocr_cache.json` — that's the
one artifact worth keeping, since it saves the ~6hr EasyOCR pass shared by
solutions 03-06. `solutions/common/scorer_local.py` imports
`participant_kit/scorer.py` by relative path, so the directory layout must
stay intact end to end.

## 2. Environment setup

Same recipe already validated on a 24GB-class GPU for `nlp1` — nothing
nlp2-specific here beyond the CUDA wheel choice below:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

# Qwen3-VL isn't in any released transformers version yet (its model card
# says to build from source) -- this one bleeding-edge install covers
# solutions 01-06.
pip install "git+https://github.com/huggingface/transformers"
pip install -r solutions/requirements.txt

# torch: nlp2's driver (590.48.01 / CUDA 13.1) is backward-compatible with
# cu12x wheels, same reasoning VM_RUNBOOK.md documents for nlp1's slightly
# older driver -- cu124 is the proven-working choice, not just a guess.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Verify CUDA is visible and pick a genuinely free GPU before running anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi   # confirm 1 or 2 are still free -- this is a shared machine
```

## 3. Sanity check the scorer first

```bash
python -m solutions.common.scorer_local
# expect: IoU=1.0000, CorLbl=1.0000, Cor≈0.99
```

## 4. Run jobs so they survive disconnecting

**Yes, jobs keep running after you log out — but only if launched the
right way.** Unlike Colab (where the whole runtime dies after ~24h and
takes everything with it), nlp2 is a persistent Linux box: a process
started over SSH survives your disconnecting *only if it isn't tied to
that login shell*. Typing `python run_pipeline.py ...` directly into a
plain SSH session gets killed (SIGHUP) the moment you disconnect — this
really does lose the run. Use `tmux` (or `screen`) instead:

```bash
tmux new -s <name>              # e.g. tmux new -s sol03
# ... run your command normally inside this session ...
# detach without stopping anything: Ctrl+B, then D
```

Reconnect later, from anywhere, with:
```bash
tmux attach -t <name>           # or: tmux a -t <name>
tmux ls                         # list your sessions if you forget the name
```
While attached, `Ctrl+B` then `[` scrolls back through output (`q` to exit).
If `tmux` isn't available, `screen -S <name>` / `Ctrl+A` `D` / `screen -r
<name>` works the same way. Last resort with no reattach capability:
`nohup <command> > run.log 2>&1 & disown`, then `tail -f run.log` after
reconnecting.

Either way, the actual deliverables (`val_predictions.jsonl`,
`val_scores.json`, `checkpoint/`) land on disk as soon as a run finishes —
they exist independently of whether you're attached to any session, so
checking the files after reconnecting works just as well as reattaching.

## 5. Aim 1: the missing ablation cell (solution 03)

```bash
cd solutions/03_vlm_lora_synthetic_augmented
python run_pipeline.py --smoke_test --gpu_index 1

tmux new -s sol03
python run_pipeline.py --gpu_index 1 --epochs 2 --grad_accum 8 --save_steps 250 --resume
# Ctrl+B, D once it's running
```
No `--skip_llm` needed (hardcoded on already) and no `--skip_ocr_extract`
needed either — `generate_synthetic.py` reuses the transferred
`ocr_cache.json` automatically since it already exists at that path. This
is the one cell of the `r_max` × LLM-assisted-injection ablation the
original shared-task submission never trained (`paper/main.tex`'s
Limitations section) — see `solutions/03_vlm_lora_synthetic_augmented/
README.md` for the full rationale.

## 6. Aim 1: multi-seed reruns (baseline + tuned, 2 extra seeds each)

None of the `run_pipeline.py` scripts expose `--seed` — rerunning one
as-is retrains the *identical* model every time (`train_lora.py`'s default
is a hardcoded 42, never wired through). `solutions/common/
run_seed_variant.py` fixes this for exactly the two configs in scope: it
trains + predicts + scores a given `--seed` against an *already-existing*
`--train_file`, skipping synthetic-data generation entirely — solution 05's
`train_augmented.jsonl` gets reused unchanged across all 3 seeds, isolating
pure training-seed variance from synthetic-generation randomness. Output
lands in `<checkpoint_dir>_seed<seed>/`, so seeds and the original
default-seed run never collide.

```bash
# round 1 -- one baseline seed + one tuned seed in parallel on the two free GPUs
tmux new -s sol02-seed123
python solutions/common/run_seed_variant.py \
  --train_file data/train.jsonl \
  --checkpoint_dir solutions/02_vlm_lora_real_data/checkpoint --seed 123 --gpu_index 1
# Ctrl+B, D

tmux new -s sol05-seed123
python solutions/common/run_seed_variant.py \
  --train_file solutions/05_vlm_lora_synthetic_tuned/train_augmented.jsonl \
  --checkpoint_dir solutions/05_vlm_lora_synthetic_tuned/checkpoint --seed 123 --gpu_index 2
# Ctrl+B, D

# round 2 -- once round 1 finishes and frees both GPUs (check with tmux ls / nvidia-smi)
tmux new -s sol02-seed2024
python solutions/common/run_seed_variant.py \
  --train_file data/train.jsonl \
  --checkpoint_dir solutions/02_vlm_lora_real_data/checkpoint --seed 2024 --gpu_index 1
# Ctrl+B, D

tmux new -s sol05-seed2024
python solutions/common/run_seed_variant.py \
  --train_file solutions/05_vlm_lora_synthetic_tuned/train_augmented.jsonl \
  --checkpoint_dir solutions/05_vlm_lora_synthetic_tuned/checkpoint --seed 2024 --gpu_index 2
# Ctrl+B, D
```
`tmux ls` shows which sessions are still running; each finishes
independently onto disk (`checkpoint_seed<N>/val_scores.json`), so none of
this requires staying attached or connected between rounds.

## 7. Every other solution (01, 04, 06)

Same commands as documented in `VM_RUNBOOK.md` sections 5-6 — only the GPU
indices differ (nlp2 only has 3 uniform 24GB GPUs at `0`/`1`/`2`, none of
nlp1's 11GB 2080 Tis, so the low-VRAM fallback flags documented there
aren't needed here; skip `0` specifically while it's occupied). Default
model (`Qwen/Qwen3-VL-8B-Instruct`, 4-bit) fits any of nlp2's 3 GPUs
comfortably.

## 8. Comparing results

Once you have `val_scores.json` for whichever solutions/seeds you ran, fill
in `solutions/RESULTS_SUMMARY.md`. For the multi-seed reruns specifically,
aggregate the 3 `val_scores.json` files per config (default-seed run +
seed 123 + seed 2024) into a mean ± std per metric before writing that up —
not automated yet (Phase 3 of the Aim 1 plan).
