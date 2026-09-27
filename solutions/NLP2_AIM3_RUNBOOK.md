# Running Aim 3 (solution 07, uncertainty quantification) on `nlp2`

Assumes you've already done Sections 1-3 of `NLP2_RUNBOOK.md` (repo cloned,
`data/`/`shroom-vis-images/` pulled, venv set up with the Qwen3-VL-capable
`transformers` build). Nothing new to install for this — same environment
as solutions 02-06.

## Why this is a different kind of job than Aim 1's

Aim 1's solutions fine-tune (backward pass, optimizer, LoRA adapters) —
that's the expensive, multi-hour part. Aim 3 only runs the frozen model
**forward**, once per labeled record, with no gradients and no optimizer —
per-example cost is much lower, but `attn_implementation="eager"` (required
to actually get attention weights out) is slower and more memory-hungry per
forward pass than the fused kernel solutions 02-06 use, and every labeled
train record needs to be processed at least once (unlike training, which
only needs one pass per epoch over the same data either way, so this isn't
necessarily faster in wall-clock — budget for it separately, and definitely
run `--dry_run` and one small real batch before committing to a schedule).

## Step 1 — `--dry_run` first, always

```bash
cd solutions/07_uncertainty_quantification
python extract_features.py --dry_run --gpu_index 1 --verbose
```

This loads the model, runs exactly 3 records, and prints (see the
README's "Unverified assumptions" section for why each of these matters):

- `image_token_id = ...` — sanity check this is a real special token, not
  e.g. an ordinary text token id (a wrong guess here won't error, it'll
  just silently produce meaningless Experiment 3 features).
- The hooked module list for both collectors (`--verbose`) — confirm the
  attention list looks like decoder self-attention layers (not the vision
  tower), and the decoder-layer count looks sane for an 8B model.
- Per-record `approx_alignment=False` (should be `False` for all 3 — if
  any are `True`, the isolated-tokenization assumption doesn't hold and
  needs a look before trusting a real run's labels).
- `lookback_shape` — `(n_response_tokens, n_layers * n_heads)` — a quick
  gut-check that the hook actually fired for every decoder layer (compare
  against the printed module count).

**Do not proceed to a real extraction job until this looks right.**

## Step 2 — GPU/memory check before a real run

Same shared-machine etiquette as Aim 1: `nvidia-smi` immediately before
launching, never target a busy GPU. Unlike training, there's no
`--max_seq_len`-based skip here yet — if a genuinely long example causes an
OOM despite the hook-based memory bounding (see solution 07's README for
why that should be rare), the fallback is to lower `--max_pixels` (fewer
image tokens -> shorter sequences) or add a length-based skip to
`extract_features.py` the same way `train_lora.py --max_seq_len` does, not
to disable the hooks.

## Step 3 — real extraction, sharded across the 3 GPUs

Feature extraction is embarrassingly parallel across records — split
`data/train.jsonl` and `data/val.jsonl` into shards and run one process per
free GPU, each in its own `tmux` session (see `NLP2_RUNBOOK.md` section 5
for the `tmux` primer if you haven't used it yet on this box):

```bash
# val set first -- smaller, gives you real per-record timing fast
tmux new -s aim3-val-s0
python extract_features.py --split val --shard_id 0 --num_shards 1 --gpu_index 1
# Ctrl+B, D

# once that's timed and looks right, shard train.jsonl across GPUs 1 and 2
# (never target GPU 0 if it's still someone else's busy job -- check first)
tmux new -s aim3-train-s0
python extract_features.py --split train --shard_id 0 --num_shards 2 --gpu_index 1
# Ctrl+B, D

tmux new -s aim3-train-s1
python extract_features.py --split train --shard_id 1 --num_shards 2 --gpu_index 2
# Ctrl+B, D
```

Each shard writes its own `features/{split}_shard{K}of{N}.npz` — no merge
step needed before probing, `train_probes.py` concatenates every shard file
it finds under `--features_dir` automatically.

Reconnect anytime with `tmux attach -t aim3-train-s0` (or `tmux ls`), or
just check whether `features/train_shard0of2.npz` exists yet — it lands on
disk only once that shard's `extract_features.py` process finishes.

## Step 4 — train probes and read the AUC grid

This step is CPU-only (no GPU, no `tmux` strictly needed, but harmless to
run in one anyway if you want to detach and let it run):

```bash
python train_probes.py --features_dir features/
```

Prints the full `(category, language)` AUC grid to stdout and saves
`aim3_auc_grid.json`. The line at the end —
`"N/20 testable (category, language) cells have >=1 signal with AUC >= 0.55"`
— is the number that actually answers Hypothesis H3 and gates whether
Phase 2's go/no-go for system integration (Aim 4) is a "go".

## Pulling results back

**Only `aim3_auc_grid.json` should go back through git** — the raw
`features/*.npz` shards are excluded via `.gitignore`
(`solutions/07_uncertainty_quantification/features/`), added alongside
this code for the same reason `checkpoint*/` is excluded for solutions
02-06: full-train-set hidden-state features are easily multi-GB
uncompressed (roughly: response-token count x (hidden_dim + n_layers x
n_heads) x 4 bytes — for an 8B model's hidden size and the full ~8,646
labeled train records, this lands well past GitHub's 100MB single-file
limit long before you'd get to the whole set). `aim3_auc_grid.json` (the
actual per-cell AUC numbers `train_probes.py` produces) is small and IS
meant to be committed.

If you want the raw features off nlp2 for local inspection or a different
analysis, `scp`/`rsync` them directly rather than routing through git:

```bash
rsync -avz nlp2:.../solutions/07_uncertainty_quantification/features/ ./features/
```

**Before running on the full `data/train.jsonl`**, consider whether you
need every labeled record at all — `train_probes.py`'s probes are simple
logistic regressions that don't need hundreds of thousands of tokens to
fit; `extract_features.py --max_examples N` caps how many records a shard
processes, which also caps extraction wall-clock and disk use. A few
thousand positive tokens per category (a small fraction of the full train
set) is very likely enough; extracting the full set first and only
discovering the shard files are unmanageably large second is the failure
mode this note exists to head off.
