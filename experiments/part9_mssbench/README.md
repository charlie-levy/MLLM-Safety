# Part 9 — MSSBench (Multimodal Situational Safety)

Run our models on [MSSBench](https://mssbench.github.io) ([paper](https://arxiv.org/abs/2410.06172),
[data](https://huggingface.co/datasets/kzhou35/mssbench), [code](https://github.com/eric-ai-lab/MSSBench)),
**exactly** per the paper's main **instruction-following (`if`)** setting. Responses only —
GPT-4o safe/unsafe judging is done externally.

## Protocol (verbatim from eric-ai-lab/MSSBench)
- Model input = single image + `PROMPT_{CHAT,EMBODIED}_IF + query/instruction` (`mss_prompts.py`,
  byte-verified against the repo).
- **chat:** the SAME query is asked once with the safe image and once with the unsafe image.
- **embodied:** `(safe_instruction[i], unsafe_instruction[i])` → safe image+safe instr, unsafe
  image+unsafe instr.
- Full data: 300 chat + 76 embodied records → 1960 `if` evaluations. Judge computes
  safe-accuracy / unsafe-accuracy / average per class.

## Subset (small, class-stratified, reproducible)
`prepare_subset.py` draws `--per_category` distinct records per (task × category), one random
sub-item each, keeping the safe+unsafe **pair** together. Default (per_category=8, seed=0):
**48 pairs → 96 items**, balanced 48 safe / 48 unsafe, 16 items in each of the 6 strata:
`chat×{property,harmful,offensive,illegal}`, `embodied×{property,harmful}`.
The generated `mss_subset.jsonl` is committed for reproducibility — every model runs identical items.

## Files
- `mss_prompts.py` — verbatim `if` prompts (do not edit).
- `prepare_subset.py` — builds the stratified subset manifest.
- `run_inference.py` — `--model` runs the subset via the frozen Part-4 generate paths.
- `gen_jobs.py` — one sbatch per model.
- `mss_subset.jsonl` — the committed 96-item subset.
- Output: `mss_<model>_responses.jsonl` (per-uid resume). Each record carries
  `uid, pair_id, task, category, context(safe|unsafe), image, text, prompt, model, response`.

## Run (Newton)
```bash
# 1) download the dataset (login node has internet) into the data_root:
cd ~/experiments/part9
HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0 \
  hf download kzhou35/mssbench --repo-type dataset --local-dir data --max-workers 1
ls data                     # expect: combined.json  chat/  embodied/

# 2) place the committed manifest where the jobs expect it:
cp ~/llava_cot_eval/experiments/part9_mssbench/mss_subset.jsonl ~/experiments/part9/mss_subset.jsonl
#   (or regenerate identically: python ~/llava_cot_eval/experiments/part9_mssbench/prepare_subset.py \
#      --data_root ~/experiments/part9/data --out ~/experiments/part9/mss_subset.jsonl)

# 3) DEBUG one model on 2 items first (confirms images load + prompt is right):
cd ~/llava_cot_eval/experiments/part9_mssbench
python run_inference.py --model llava_cot --subset ~/experiments/part9/mss_subset.jsonl \
  --data_root ~/experiments/part9/data --debug_n 2     # (run under srun --gres=gpu:...)

# 4) materialize + submit all model jobs:
python gen_jobs.py
bash ~/experiments/part9/jobs/submit_part9.sh
```
Then pull `~/experiments/part9/results/mss_*_responses.jsonl` and run the MSSBench GPT-4o judge
(group by `context` and `category` for safe-acc / unsafe-acc).
