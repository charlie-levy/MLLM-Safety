# Part 7 — Eval stage (SIUO + ScienceQA + MathVista) for the 3 module-ablation checkpoints

Runs on `llm`, `vision`, `both` (LoRA on LLM / vision-encoder / both). Every eval generates
**one example at a time** via the frozen `run_eval.generate_one` (greedy 2048-tok) — exactly
as the papers do. For speed, SQA/MathVista fan the *examples* out across GPUs (data-parallel
sharding) and merge; the merged file is byte-identical to a single-GPU run. Newton compute
nodes are offline, so all downloads happen ONCE on the login node.

## What each eval measures + how it's scored (apples-to-apples)
| Eval | Data | Scoring | Faithfulness |
|---|---|---|---|
| **SIUO** | SIUO 167, zoom_blur(sev2) whole set | (safety) judged elsewhere, responses-only | byte-identical to Part 6 SIUO |
| **ScienceQA** | full **test** (4,241) | **paper's exact** accuracy: regex `The answer is X.` → `get_pred_idx` → acc | verbatim from lupantech/ScienceQA — NO LLM judge |
| **MathVista** | **testmini** (1,000), built-in `query` | **official** `normalize_extracted_answer`+`safe_equal`; extractor = local Llama-3-8B (offline stand-in for GPT-4) | scoring imported verbatim from lupantech/MathVista |

## Step 1 — login node (internet, REU env): download data + official MathVista code
```bash
conda activate REU
export HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
mkdir -p /home/ch169788/experiments/part7/logs
E=~/llava_cot_eval/experiments/part7_module_ablation/eval

python3 $E/prep_sqa_test.py          # -> data/scienceqa_test.json + sqa_test_images/  (4,241)
python3 $E/prep_mathvista.py         # -> data/mathvista_testmini.json + mathvista_images/ (1,000)
git clone --depth 1 https://github.com/lupantech/MathVista ~/MathVista   # official scoring/prompt
pip install python-Levenshtein       # used by MathVista's official multi_choice normalizer
```

## Step 2 — inference (GPU) — AFTER the 3 checkpoints exist
```bash
J=~/llava_cot_eval/experiments/part7_module_ablation/eval/jobs
for m in llm vision both; do
  sbatch $J/siuo_$m.sbatch          # 1 job    -> siuo/siuo_zoom_blur_<m>_responses.jsonl
  sbatch $J/sqa_$m.sbatch           # array 0-7 -> sqa/shard_<m>_*of8.jsonl        (8 GPUs)
  sbatch $J/mathvista_$m.sbatch     # array 0-3 -> mathvista/shard_<m>_*of4.jsonl  (4 GPUs)
done
```
Per-idx/per-pid resume-safe: if an array task hits walltime, resubmit — it fills only the gaps.

## Step 2b — merge shards (login node, CPU) once each eval's arrays finish
```bash
for m in llm vision both; do
  python3 $E/merge_shards.py --dir /home/ch169788/experiments/part7/sqa       --model $m --key idx --out raw_$m.jsonl       --expect 4241
  python3 $E/merge_shards.py --dir /home/ch169788/experiments/part7/mathvista --model $m --key pid --out responses_$m.jsonl --expect 1000
done
# each must print "count OK ✅"
```

## Step 3 — scoring
```bash
# ScienceQA: paper-exact accuracy (CPU — no model)
python3 $E/score_sqa.py --dir /home/ch169788/experiments/part7/sqa                      # -> judged_<m>.json

# MathVista: (a) Llama-3-8B extraction on GPU, then (b) official scoring (CPU)
sbatch $J/mathvista_extract.sbatch                                                       # -> extracted_<m>.jsonl
python3 $E/score_mathvista.py --dir /home/ch169788/experiments/part7/mathvista           # -> scores_<m>.json
```

## Step 4 — pull local (from your Mac)
```bash
mkdir -p ~/Desktop/REU/judge_responses/results/part7_module_ablation
scp -r 'ch169788@newton.ist.ucf.edu:/home/ch169788/experiments/part7/{siuo,sqa,mathvista}' \
       ~/Desktop/REU/judge_responses/results/part7_module_ablation/
```

## Newton layout
```
/home/ch169788/experiments/part7/
  data/        scienceqa_test.json, sqa_test_images/, mathvista_testmini.json, mathvista_images/
  siuo/        siuo_zoom_blur_<model>_responses.jsonl
  sqa/         shard_<model>_*of8.jsonl -> raw_<model>.jsonl        + judged_<model>.json   (accuracy)
  mathvista/   shard_<model>_*of4.jsonl -> responses_<model>.jsonl  + extracted_<model>.jsonl + scores_<model>.json
  logs/
```
