# Pipeline check: new code vs solution 02 (Qwen3-VL-8B, seed 42, same recipe)

Pre-registered: the new-pipeline run must land within ~0.01 val_select Cor+Lbl of the
solution 02 reference (old Colab code). Only val_select is used.

| val_select Cor+Lbl | 4-lang mean | en | fr | it | zh |
|---|---|---|---|---|---|
| new pipeline (`p1_qwen3vl8b_seed42`) | 0.4269 | 0.3807 | 0.4339 | 0.3994 | 0.4934 |
| solution 02 (`ref_sol02_qwen3vl8b_seed42`) | 0.4214 | 0.3627 | 0.4320 | 0.3884 | 0.5027 |
| difference | +0.0055 | +0.018 | +0.002 | +0.011 | -0.009 |

Result (2026-10-27): **pass** (+0.0055, inside the ~0.01 band; per-language differences
of both signs, consistent with seed noise). New run: 3,020 optimizer steps, 2.23 s/example,
no sequence over 4,096 tokens, peak 20.0 GB.
