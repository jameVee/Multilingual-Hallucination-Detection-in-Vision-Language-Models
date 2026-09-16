# Running these solutions on the university server (`nlp2`)

Everything trained on this repo up to now (solutions 02, 04, 05, 06 — real
`checkpoint/` + `val_scores.json` for each) ran on Google Colab Pro (A100
40GB). This doc covers running the rest on `nlp2` instead. Solutions 01 and
03 have never been run at all yet — no checkpoint, no scores.

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

## 1. Get the code

The code is on GitHub now — clone it directly on nlp2, no transfer from a
local machine needed for this part:

```bash
git clone https://github.com/jameVee/Multilingual-Hallucination-Detection-in-Vision-Language-Models.git
cd Multilingual-Hallucination-Detection-in-Vision-Language-Models
```

This already includes `solutions/03_vlm_lora_synthetic_augmented/
ocr_cache.json` (and the other solutions' copies) — the ~6hr EasyOCR pass
shared by solutions 03-06 never needs to be repeated. It does **not**
include `data/`, `shroom-vis-images/`, `docs/`, `paper/`, or `thesis
proposal/` (excluded from the repo on purpose — see its `.gitignore`), and
excludes every solution's `checkpoint/` (167MB adapter files, over GitHub's
100MB limit) — get those two pieces separately, below.

## 2. Get the data

`data/` and `shroom-vis-images/` aren't in the git repo. Pull them from a
hosted copy instead of copying from a local machine — run this from the
repo root (`Multilingual-Hallucination-Detection-in-Vision-Language-Models/`):

```bash
curl -L -o /tmp/shroom-data.zip "https://drive.google.com/uc?export=download&id=1peJu1P79J_JsthDySdXj_B-kVuOyYWi-"

# Images archive is 2.47GB -- Drive gates anything this size behind a
# click-through "can't scan for viruses" confirmation page that plain curl
# can't get past on its own (see explanation below). This extracts that
# page's confirm token and re-requests with it -- no extra tools needed,
# unlike gdown (which needs `pip install` and hit "externally-managed-
# environment" on nlp2's system Python -- this avoids that entirely).
IMG_ID="1Cpb2A8hUMl3_tPSIDPLIDn5MP7nTwKl-"
curl -sL "https://drive.google.com/uc?export=download&id=${IMG_ID}" -o /tmp/confirm_page.html
UUID=$(grep -oE 'name="uuid" value="[^"]*"' /tmp/confirm_page.html | sed -E 's/.*value="([^"]*)"/\1/')
curl -L "https://drive.usercontent.google.com/download?id=${IMG_ID}&export=download&confirm=t&uuid=${UUID}" \
  -o /tmp/shroom-images.tar.gz
rm /tmp/confirm_page.html

unzip /tmp/shroom-data.zip
tar -xzf /tmp/shroom-images.tar.gz
```

**On the data link**: a plain Google Drive share URL
(`.../file/d/<id>/view?usp=sharing`, an earlier version of this doc) does
**not** work with `curl` — it returns Drive's ~80KB HTML viewer page, not
the file, and `unzip` fails on it with "End-of-central-directory signature
not found." The `uc?export=download&id=<id>` form above is the fix for
this file specifically — verified end-to-end: it 303-redirects to a real
`application/octet-stream` download and inflates to a valid
`data/{train,val,test}.jsonl` zip (14.7MB uncompressed).

**On the images link**: the same `uc?export=download` form that works for
the small data.zip returns a *"Google Drive - Virus scan warning"* HTML
page for this one instead — Drive can't scan archives past a certain size
(this one is 2.47GB) and gates the download behind a click-through
confirmation. That page is a real HTML `<form>` whose hidden inputs
(`id`, `export=download`, `confirm=t`, and a fresh `uuid` generated per
request) point at the actual file, hosted at a different subdomain
(`drive.usercontent.google.com/download`) — the two-step command above
just extracts that `uuid` with `grep`/`sed` and replays the form as a
second `curl` request. Verified against this exact file with a partial
byte-range request: the response starts with the real gzip magic bytes
(`1f 8b 08 00`), not HTML. This needs no extra tools (works around the
`pip install gdown` → "externally-managed-environment" error nlp2's
system Python throws — `gdown` remains a fine alternative if you'd rather
install it *inside* the venv from section 3, or if Drive ever changes this
confirmation flow again and the scraping above stops matching it). Expect
the 2.47GB download itself to take a while depending on nlp2's link speed
— run it inside `tmux` (section 5 below) if it's going to take more than a
few minutes.

If a Drive link like this one is ever swapped out again: a link that
returns HTML instead of the expected binary — check with `curl -sIL <url>
| grep -i content-type`, or just peek at the first bytes — is the standard
failure mode for anything hosted on Drive past a few dozen MB, `a3s.fi`
(the original shared-task hosting) doesn't have this problem at all, so
prefer it over a personal Drive copy for anything sizeable when there's a
choice.

**No `-d`/`-C` target and no `mkdir` first, on purpose**: both archives
already contain their own top-level folder internally
(`data/train.jsonl`, `data/val.jsonl`, `data/test.jsonl` in the zip;
`shroom-vis-images/*.jpg` in the tarball — checked directly against both
files above) — extracting at the repo root reproduces `data/` and
`shroom-vis-images/` correctly. Extracting into a pre-made same-named
directory instead (`unzip ... -d data`) double-nests it to `data/data/...`.
Sanity-check afterward regardless:
`python -c "from solutions.common import data_io; print(len(data_io.load_jsonl(data_io.TRAIN_FILE)))"`
should print `12082`.

If these links are ever down, `rsync` from a local checkout that already
has both folders is the fallback:
```bash
rsync -avz --progress \
  "/path/to/local/Shroom-visions Shared Task/data" \
  "/path/to/local/Shroom-visions Shared Task/shroom-vis-images" \
  you@nlp2:/path/to/remote/Multilingual-Hallucination-Detection-in-Vision-Language-Models/
```

## 3. Environment setup

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
# cu12x wheels -- cu124 is the proven-working choice, not just a guess.
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124
```

Also set this once per shell session, before running anything that trains —
the Colab notebooks always set it, but nothing in `train_lora.py`/
`run_pipeline.py` does, so it's silently missing from this CLI workflow
unless you export it yourself:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```
Reduces CUDA memory fragmentation (PyTorch reserving memory it isn't
currently using) — matters more here than it did on Colab's A100, since
nlp2's 24GB 3090s have much less headroom to absorb the waste. See
section 6 for a real OOM this was a contributing factor in.

Verify CUDA is visible and pick a genuinely free GPU before running anything:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
nvidia-smi   # confirm 1 or 2 are still free -- this is a shared machine
```

## 4. Sanity check the scorer first

```bash
python -m solutions.common.scorer_local
# expect: IoU=1.0000, CorLbl=1.0000, Cor≈0.99
```

## 5. Run jobs so they survive disconnecting

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

## 6. Aim 1: the missing ablation cell (solution 03)

```bash
cd solutions/03_vlm_lora_synthetic_augmented
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_pipeline.py --smoke_test --gpu_index 1

tmux new -s sol03
python run_pipeline.py --gpu_index 1 --epochs 2 --grad_accum 8 --save_steps 250 --resume --max_seq_len 3072
# Ctrl+B, D once it's running
```
No `--skip_llm` needed (hardcoded on already) and no `--skip_ocr_extract`
needed either — `generate_synthetic.py` reuses the cloned `ocr_cache.json`
automatically since it already exists at that path. This is the one cell
of the `r_max` × LLM-assisted-injection ablation the original shared-task
submission never trained (`paper/main.tex`'s Limitations section) — see
`solutions/03_vlm_lora_synthetic_augmented/README.md` for the full
rationale.

**Known CUDA OOM risk on 24GB cards, hit for real on this run** — `--gpu_index 1`
above is deliberate, not a typo for `0`: a real nlp2 run of this exact
solution OOM'd after ~5 minutes with `--gpu_index 0`
(`torch.OutOfMemoryError`, 21.91 GiB used by the training process alone,
1.65 GiB free on a 23.56 GiB card — the arithmetic rules out another job
sharing GPU 0 at the time; this process alone used nearly the whole card).
Two compounding, already-documented causes, both addressed in the command
above:
- `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` was missing (only
  ever set inside the Colab notebooks, never in the scripts themselves —
  the OOM message's own suggested fix, and it reduces the "reserved but
  unallocated" fragmentation waste, which was 5.67 GiB on the failed run).
- `train_lora.py`'s own docstring (around `--max_seq_len`) documents this
  *exact* failure signature happening before, on a 40GB Colab A100: "a run
  stable for 726 steps suddenly needed to allocate 25GB more for one
  attention op, almost certainly from one unusually long response (e.g.
  an `inject_repetition_loop`-generated 'other' example)". Solutions
  03-06 all share this injection method, so the same risk applies to all
  of them on 24GB, just with less warning before it hits than it had on
  the A100. `--max_seq_len 3072` (down from the 4096 default) lowers the
  worst-case sequence length this can spike on; drop further (e.g. `2048`)
  or add `--max_pixels 401408` if it still OOMs.

If a run does OOM anyway, re-launching the exact same command is safe —
`--resume` picks up from the last `--save_steps 250` checkpoint rather
than restarting, *if* one was actually written yet (`ls checkpoint/` —
5 minutes in is usually too early to have reached step 250, so check
before assuming there's progress to resume).

## 7. Aim 1: multi-seed reruns (baseline + tuned, 2 extra seeds each)

None of the `run_pipeline.py` scripts expose `--seed` — rerunning one
as-is retrains the *identical* model every time (`train_lora.py`'s default
is a hardcoded 42, never wired through). `solutions/common/
run_seed_variant.py` fixes this for exactly the two configs in scope: it
trains + predicts + scores a given `--seed` against an *already-existing*
`--train_file`, skipping synthetic-data generation entirely — solution 05's
`train_augmented.jsonl` (already in the repo) gets reused unchanged across
all 3 seeds, isolating pure training-seed variance from synthetic-
generation randomness. Output lands in `<checkpoint_dir>_seed<seed>/`, so
seeds and the original default-seed run (from Colab) never collide.

`--max_seq_len 3072` on the two `sol05-*` commands (not `sol02-*`) for the
same reason as section 6 — solution 05's `train_augmented.jsonl` contains
the same heuristic-injection `other`-class examples that OOM'd solution 03;
the real-data baseline (`data/train.jsonl`, no synthetic amplification of
rare long responses) is markedly less exposed to this, so it's left at the
default there. `export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
once per shell (section 3) applies to all four of these regardless.

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
  --checkpoint_dir solutions/05_vlm_lora_synthetic_tuned/checkpoint --seed 123 --gpu_index 2 \
  --max_seq_len 3072
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
  --checkpoint_dir solutions/05_vlm_lora_synthetic_tuned/checkpoint --seed 2024 --gpu_index 2 \
  --max_seq_len 3072
# Ctrl+B, D
```
`tmux ls` shows which sessions are still running; each finishes
independently onto disk (`checkpoint_seed<N>/val_scores.json`), so none of
this requires staying attached or connected between rounds.

## 8. Reference: every other solution

Not part of Aim 1, but here for completeness if you ever need to rerun or
extend one of these on nlp2. Default model everywhere is
`Qwen/Qwen3-VL-8B-Instruct` (dense, 4-bit), and the *baseline* (solution 02,
real data only, no long synthetic `other`-class responses) fits nlp2's 24GB
3090s comfortably with no extra flags. **The synthetic-augmented ones
(03-06) are less safe than this section originally claimed** — solution
03 actually OOM'd on nlp2 (section 6 has the full diagnosis); set
`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` and consider
`--max_seq_len 3072` for 04/05/06 too, not just 03, since they share the
same injection method that triggers it. Pass `--model_id
Qwen/Qwen3-VL-30B-A3B-Instruct` for the bigger sparse-MoE alternative on
any command below if you want it (expect it to need those same flags too).

**Verification level differs by solution.** Solutions 02, 04, 05, and 06
already have a real, complete Colab run each (`val_scores.json` on disk) —
proof the code path genuinely works end to end. Solutions 01 and 03 have
never been run anywhere — treat their first nlp2 run as the actual first
real test, `--smoke_test` (or `--dry_run` where available) first, no
skipping straight to a full run.

### 01 — Lightweight tagger (cheapest, ~20-40 min even on a 2080 Ti-class card)

```bash
cd solutions/01_lightweight_tagger
python run_pipeline.py --gpu_index 1 --epochs 3 --batch_size 16
```
CLIP is frozen and only runs once (image features cached to
`image_cache.pt`), so this is XLM-R-base-sized training — cheap on any GPU.

### 02 — VLM LoRA fine-tune, real data only (already has a real Colab run)

```bash
cd solutions/02_vlm_lora_real_data
python train_lora.py --dry_run
python run_pipeline.py --smoke_test --gpu_index 1
python run_pipeline.py --gpu_index 1 --epochs 2 --grad_accum 8
```

### 04 — VLM LoRA, synthetic augmented, r_max=8 + LLM (already has a real Colab run)

```bash
cd solutions/04_vlm_lora_synthetic_augmented_llm
export ANTHROPIC_API_KEY=sk-ant-...   # set this yourself; never paste into a chat
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_pipeline.py --smoke_test --gpu_index 1
python run_pipeline.py --gpu_index 1 --epochs 2 --grad_accum 8 --max_seq_len 3072
```
Reuses the cloned `ocr_cache.json` automatically; `llm_synth_cache.json`
caches API calls incrementally, so an interrupted run resumes instead of
re-spending budget. Same OOM risk as solution 03 (section 6) — same fix.

### 06 — VLM LoRA, synthetic augmented, tuned r_max=3 + LLM (already has a real Colab run)

```bash
cd solutions/06_vlm_lora_synthetic_tuned_llm
export ANTHROPIC_API_KEY=sk-ant-...
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python run_pipeline.py --smoke_test --gpu_index 1
python run_pipeline.py --gpu_index 1 --epochs 2 --grad_accum 8 --max_seq_len 3072
```

## 9. Comparing results

Once you have `val_scores.json` for whichever solutions/seeds you ran, fill
in `solutions/RESULTS_SUMMARY.md`. For the multi-seed reruns specifically,
aggregate the 3 `val_scores.json` files per config (default-seed run +
seed 123 + seed 2024) into a mean ± std per metric before writing that up —
not automated yet (Phase 3 of the Aim 1 plan).
