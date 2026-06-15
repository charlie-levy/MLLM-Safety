# CLAUDE.md — conventions for this repo

Research repo: **does the safety of LLaVA-CoT survive image corruption?** See
[README.md](README.md) for the folder map. This file captures the conventions an
agent must follow to avoid the mistakes this project has already hit.

## Where things run
- **Newton (HPC), conda env `REU`, H100 nodes** — all GPU work (inference, judges).
  Only the **user** can SSH (password-gated); an agent must hand the user exact
  commands to run and work from pasted output + the local mirror.
- **Login node `evuser2`** — git/ls/squeue and *tiny* pandas lookups ONLY. It has a
  hard `RLIMIT_NPROC ~100`, so any multi-threaded import (numpy/OpenBLAS) segfaults.
  Prefix even small Python with `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1`.
  Anything that loads a model MUST be inside an `srun ... --pty bash` GPU session.
- **Mac repo** `/Users/charlielevy/Desktop/REU/llava_cot_eval` — code editing,
  committing, and chart scripts. `results_newton/` mirrors Newton's `results/`.

## Newton offline env (every GPU job exports these)
Compute nodes have flaky internet → HF Hub calls crash jobs. Always:
```
HF_HUB_OFFLINE=1  TRANSFORMERS_OFFLINE=1  HF_HUB_DISABLE_XET=1  HF_HUB_ENABLE_HF_TRANSFER=0
```
Slurm: `--gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42` (evc42 is bad).
Model weights are pre-cached in `~/.cache/huggingface/hub`; download once on the
login node (`snapshot_download(..., max_workers=1)` to dodge the proc cap), then
run jobs offline.

## Models (all LLaVA-CoT = `Xkev/Llama-3.2V-11B-cot`, + LoRA r=8/α=16)
`load_model_and_processor(use_tis=/use_msr=/use_sage=)` → `(model, processor, tag)`.
| flag | tag | weights |
|---|---|---|
| (none) | `base` | — |
| `use_tis` | `base+TIS` | `model_weights/llama_cot_tis` |
| `use_msr` | `base+MSR` | `model_weights/llama_cot_msr` |
| `use_sage` | `base+SAGE` | `model_weights/llama_cot_sage` |
Generation: **greedy** (`do_sample=False`). `config.MAX_NEW_TOKENS=1024`, but
**MSR-Align needs 2048** — its final decision (`### Safe Response`) comes last after
a long `<think>` + Image Description + Risk Assessment; 1024 truncates the verdict.

## Datasets & metrics (`dataset_loader.py`)
- **FigStep** (500) → **ASR** (↓ safer). Harmful request rendered in the image.
- **XSTest** (~250) + **MMSA** (428 safe) → **ORR** (↓ less over-refusal). Benign.
- **ScienceQA** → **SQA utility** (↑ better), LLaMA-judged.
Sample schema: `{prompt, image, label, metadata{dataset, category, image_path,...}}`.

## Judging — prefer model judges over string matching
String/keyword matching on long CoT is unreliable (see apostrophe bug below). Use:
- **ASR** = `judge_figstep_guard.py` → **Llama Guard 3 Vision** (`meta-llama/Llama-Guard-3-11B-Vision`).
  Vision model: load with `AutoModelForVision2Seq`. 2-turn conversation
  user`[text, image]` + assistant`[response]`; output `safe`/`unsafe` (+ S1–S13).
  **ASR = % unsafe.** The judge always sees the **clean** FigStep image.
- **ORR** = `judge_safety_hf.py --mode orr` → LLaMA-3-8B (ANSWER vs REFUSE /
  SAFE vs OVERREFUSE), reads the FULL response, scores the FINAL decision.
- `judge_safety_hf.py` MAXLEN=6000 chars so MSR's long verdict isn't truncated.

## ⚠️ The curly-apostrophe refusal bug (history)
`metrics.py`'s keyword detector missed U+2019 apostrophes ("I'm"/"can't") in
TIS refusals → refusals counted as compliance → **inflated ASR/ORR** (TIS-specific).
Any pre-fix string-matched number is suspect; re-score from saved responses
(`rescore_from_responses.py`) or, better, use the model judges above. This is the
core reason the project moved to judge-based scoring.

## MSR-Align Guard eval (current pipeline)
Driver `eval_msr_guard.py` (`--task figstep|orr --blur_pct N`) + judges, wired by
`slurm_scripts/submit_msr_guard.sh` (4 parallel inference jobs + 4 afterok judges).
Outputs under `results/msr_guard_eval/{clean,blur20}/`, capturing EVERYTHING:
per-sample `full_response`, `image_path`, `judge_input` (exact text shown to the
judge), `judge_raw_output`, `verdict`, `violated_categories`, `flag`; plus an
aggregate JSON per condition. Final table: `python code/report_msr_guard.py`.

## Coding conventions
- `code/` is flat; slurm calls `code/x.py` and modules import by bare name
  (scripts do `sys.path.insert(0, dirname(__file__))` + `os.chdir(repo_root)`).
- New result folders, never overwrite existing ones. No silent `except/pass` —
  assert dataset counts (500/250/428) and fail loudly on empty responses.
- `results_newton/` is pulled FROM Newton; committing it does nothing for Newton.
  Keep code commits scoped — don't sweep the results/plots churn into them.
