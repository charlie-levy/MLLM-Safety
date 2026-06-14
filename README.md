# LLaVA-CoT Safety Under Image Corruption

Evaluating multimodal safety adapters (TIS, SAGE, MSR-Align) on LLaVA-CoT
(`Xkev/Llama-3.2V-11B-cot`) as the input image is corrupted (Gaussian noise,
Gaussian blur, JPEG, low-light, pixelation), at 0–80%.

Metrics: **ASR** (FigStep, lower=safer) · **ORR** (XSTest + MMSA, lower=less
over-refusal) · **SQA Utility** (ScienceQA accuracy, higher=better).

## Repository layout

```
code/            active pipeline (flat — eval scripts are called as `code/x.py`
                 by slurm and import each other by module name; do not nest)
  archive/       superseded / one-off scripts kept for reference
slurm_scripts/   Newton SLURM submit scripts (+ archive/)
scripts/         Mac-side helpers (pull_results.sh)
results_newton/  pulled results (JSON metrics, response CSVs, plots, strips)
```

## `code/` — what each file does

**Core libraries** (imported everywhere — keep in `code/`):
- `config.py` — model + dataset paths on Newton
- `model_loader.py` — load LLaVA-CoT (+ optional LoRA adapter) / R1-OneVision
- `dataset_loader.py` — FigStep / XSTest / MMSA / ScienceQA loaders
- `evaluator.py` — batched inference + image-corruption dispatch
- `metrics.py` — refusal detection, ASR, ORR, accuracy (**`_normalize` folds
  curly apostrophes — see the apostrophe-bug note below**)
- `noise_utils.py`, `blur_utils.py`, `jpeg_utils.py`, `brightness_utils.py`,
  `pixelate_utils.py` — percentage-based corruptions (0–80%)

**Eval scripts** (run on Newton via slurm; flags `--use_tis/--use_sage/--use_msr`,
`--noise_pct/--blur_pct/--corrupt <jpeg|brightness|pixelate> --corrupt_pct`):
- `eval_figstep_noise_sweep.py` — FigStep ASR
- `eval_orr_noise_sweep.py` — XSTest + MMSA ORR
- `eval_sqa_noise_sweep.py` — ScienceQA generations
- `judge_sqa_utility_hf.py` — LLaMA-3 judge for SQA accuracy

**Post-processing** (run after jobs finish; CPU, no GPU):
- `rescore_from_responses.py` — recompute ASR/ORR JSONs from saved responses
- `refresh_response_csvs.py` — sync derived columns in every response CSV
- `combine_figstep_tis_responses.py` — one combined TIS FigStep CSV (all conditions)
- `make_corruption_strips.py` — 0–80% example strips for all 5 corruptions × 3 datasets

**Analysis & charts** (run on Mac, read `results_newton/`):
- `safety_analysis.py` — validated tables + combo (stacked-bar+utility-line) charts
- `presentation_chart.py` — per-model 3-metric figure + tables (big, white/black)
- `analysis_noise_failsafe.py` — noise: utility erodes, safety holds
- `analysis_deep.py` — per-category FigStep ASR + CoT degradation-awareness
- `plot_results.py` — legacy per-model bar charts

## End-to-end workflow

```bash
# On Newton (submit, then leave):
bash slurm_scripts/submit_<...>.sh          # e.g. submit_realistic_corruptions.sh
# after jobs finish:
bash slurm_scripts/submit_judge_realistic.sh   # judge SQA (run once)
python3 code/rescore_from_responses.py          # fix metric JSONs from responses
python3 code/refresh_response_csvs.py           # sync CSV derived columns
python3 code/combine_figstep_tis_responses.py   # rebuild combined CSV

# On the Mac:
bash scripts/pull_results.sh                    # rsync everything (one ssh session)
python3 code/safety_analysis.py
python3 code/presentation_chart.py
python3 code/analysis_deep.py
```

## ⚠️ Apostrophe bug (important context)

Before the fix in `metrics.py` (`_normalize`), the refusal detector matched ASCII
apostrophes but LLaVA-CoT writes refusals with U+2019 ("I'm"/"can't"), so refusals
were missed and counted as attack successes — inflating ASR, TIS-specifically.
**Any ASR/ORR produced before that fix is wrong.** Because every per-sample response
is saved, repair without re-running: `rescore_from_responses.py` + `refresh_response_csvs.py`.

## Newton notes
- Only the repo owner SSHes to Newton. Compute nodes have flaky outbound internet —
  eval/​judge jobs export `HF_HUB_OFFLINE=1` / `TRANSFORMERS_OFFLINE=1` (model is cached).
- GPUs: H100 PCIe (`--gres=gpu:nvidia_h100_pcie:1`), `--exclude=evc42`.
