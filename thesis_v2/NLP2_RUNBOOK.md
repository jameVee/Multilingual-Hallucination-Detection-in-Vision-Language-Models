# nlp2 runbook (thesis_v2)

Code reaches nlp2 through git. Push from the laptop, then `git pull` on nlp2. Run every
job in `tmux` with a log file. Check `nvidia-smi` before each launch, and run at most
**2 heavy GPU jobs at once** (nlp2 has crashed under more). All outputs go under
`thesis_v2/runs/`, which is gitignored. Copy back only prediction files and score JSONs.

```bash
ssh 10.16.88.2
cd ~/Shroom-visions-Shared-Task && git pull
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
tmux new -s <job>      # detach: Ctrl-b d   reattach: tmux attach -t <job>
```

## Phase 0: feasibility gate (one backbone at a time, ~15 min each)

```bash
python -m thesis_v2.part1_models.feasibility --backbone qwen3vl8b   --gpu_index 0 2>&1 | tee thesis_v2/runs/logs/feas_qwen.log
python -m thesis_v2.part1_models.feasibility --backbone gemma3_12b  --gpu_index 1 2>&1 | tee thesis_v2/runs/logs/feas_gemma.log
python -m thesis_v2.part1_models.feasibility --backbone internvl3_8b --gpu_index 0 ...
python -m thesis_v2.part1_models.feasibility --backbone pixtral12b  --gpu_index 1 ...
# if one fails: llama32v11b, then phi4mm
```
Results go to `thesis_v2/results/feasibility/<backbone>.json` (commit these). Use
`sec_per_example` to plan the compute.

## Part 1: model diversity (seed 42; seeds 123/2024 only if the tie rule fires)

Qwen3-VL-8B seed 42 is NOT retrained: solution 02 used the identical recipe and is
imported by `python -m thesis_v2.part1_models.import_solution02` (laptop) into
`runs/part1/qwen3vl8b_seed42/` (val_select 0.4214, val_confirm 0.4107). The other
backbones run the same commands as below with their own `--backbone`:

```bash
python -m thesis_v2.part1_models.train   --backbone qwen3vl8b --seed 42 --gpu_index 0
python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter thesis_v2/runs/part1/qwen3vl8b_seed42 --split val  --gpu_index 0
python -m thesis_v2.part1_models.predict --backbone qwen3vl8b --adapter thesis_v2/runs/part1/qwen3vl8b_seed42 --split test --gpu_index 0
python -m thesis_v2.common.evaluate --pred thesis_v2/runs/part1/qwen3vl8b_seed42/val_predictions.jsonl --split val_select --name p1_qwen3vl8b_seed42
```
Training and prediction are resumable (`--resume`). If nlp2 crashes, rerun the same
command with `--resume`.

## Part 2: synthetic conditions (Part 1 winner backbone `<bb>`, seeds 42/123/2024)

Data files are committed under `thesis_v2/part2_synth/data/` (built on the laptop).
```bash
# C1
python -m thesis_v2.part1_models.train --backbone <bb> --seed 42 \
  --train_files data/train.jsonl thesis_v2/part2_synth/data/C1_upsample.jsonl \
  --output_dir thesis_v2/runs/part2/C1_seed42 --gpu_index 0
# C2 / C3 / C4: same, with C2_heuristic.jsonl / C3_llm.jsonl / C4_retrieval.jsonl
```
That is 4 conditions × 3 seeds = 12 runs, plus C0 seeds 123/2024. Then predict val and test
for each run and score it on val_select.

## Part 3

```bash
# 3.0 leak fix: clean-record features (judge Qwen3-VL-8B, no fine-tuning)
python -m thesis_v2.part3_uncert.extract_features_all --split val --gpu_index 0 --dry_run
python -m thesis_v2.part3_uncert.extract_features_all --split val   --gpu_index 0
python -m thesis_v2.part3_uncert.extract_features_all --split train --shard_id 0 --num_shards 2 --gpu_index 0
python -m thesis_v2.part3_uncert.extract_features_all --split train --shard_id 1 --num_shards 2 --gpu_index 1
python -m thesis_v2.part3_uncert.features --build          # PCA + compact files; asserts full coverage

# 3a sampling vote (adapter of the Part 1 winner, one run per seed adapter)
python -m thesis_v2.part1_models.predict --backbone <bb> --adapter <adapter> --split val  --num_samples 8 --temperature 0.7 --gpu_index 0
python -m thesis_v2.part1_models.predict --backbone <bb> --adapter <adapter> --split test --num_samples 8 --temperature 0.7 --gpu_index 0
python -m thesis_v2.part3_uncert.sample_vote --val <adapter>/val_samples8_T0.7.jsonl --diversity
python -m thesis_v2.part3_uncert.sample_vote --val <...> --test <...> --name 3a_<bb>_seed42 --decide

# 3b CRF probe (3 seeds)
python -m thesis_v2.part3_uncert.crf_probe --selftest
python -m thesis_v2.part3_uncert.crf_probe --seed 42 --gpu_index 0
```

## Integration
Fill in `thesis_v2/integration/config.json`, then run
`python -m thesis_v2.integration.build_candidates --config thesis_v2/integration/config.json`.

## Test submission (the user does this)
`split_predictions_by_language.py` → `participant_kit/format_checker.py --reference-file data/test.jsonl`
→ upload → append the official numbers to `thesis_v2/results/test_results.txt`.
