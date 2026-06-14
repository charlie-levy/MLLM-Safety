# LLaVA-CoT Safety Under Image Corruption

Do safety adapters (TIS, SAGE, MSR-Align) on LLaVA-CoT survive image corruption
(noise, blur, JPEG, low-light, pixelation, 0–80%)?

Metrics: **ASR** (FigStep, ↓ safer) · **ORR** (XSTest + MMSA, ↓ less over-refusal)
· **SQA Utility** (ScienceQA, ↑ better).

## Folders

| Folder | What's in it |
|---|---|
| `code/` | all Python (flat — slurm calls `code/x.py` and modules import by name) |
| `code/archive/` | old / superseded scripts |
| `slurm_scripts/` | **run on Newton** — SLURM submit scripts |
| `scripts/` | **run on your Mac** — `pull_results.sh` (rsync results down) |
| `results_newton/` | pulled results (mirrors Newton's `results/`, so its data folders stay flat) |
| `results_newton/plots/` | charts, grouped: `presentation/` `analysis/` `deep/` `per_model/` |

## `code/` quick map

- **libs:** `config`, `model_loader`, `dataset_loader`, `evaluator`, `metrics`,
  `{noise,blur,jpeg,brightness,pixelate}_utils`
- **eval (Newton):** `eval_figstep_noise_sweep`, `eval_orr_noise_sweep`,
  `eval_sqa_noise_sweep`, `judge_sqa_utility_hf`
- **post-process:** `rescore_from_responses`, `refresh_response_csvs`,
  `combine_figstep_tis_responses`, `make_corruption_strips`
- **charts (Mac):** `safety_analysis`, `presentation_chart`,
  `analysis_noise_failsafe`, `analysis_deep`, `plot_results`

## Workflow

```bash
# Newton: submit jobs, then after they finish —
bash slurm_scripts/submit_judge_realistic.sh
python3 code/rescore_from_responses.py
python3 code/refresh_response_csvs.py
# Mac:
bash scripts/pull_results.sh
python3 code/safety_analysis.py && python3 code/presentation_chart.py && python3 code/analysis_deep.py
```

## ⚠️ Apostrophe bug
`metrics.py` once missed curly-apostrophe refusals ("I'm"/"can't" = U+2019),
inflating ASR. Fixed via `_normalize`. **Any ASR/ORR from before the fix is wrong** —
repair from saved responses with `rescore_from_responses.py` (no GPU needed).
