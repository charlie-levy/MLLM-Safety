# Part 11 — Thinking-budget dose–response for SAFETY (pilot)

**Question:** as we force a reasoning VLM to think *longer*, does its safety get better,
worse, or non-monotonic — and how does image corruption interact with that?

This is the safety analogue of *"Does Thinking More Always Help? Mirage of Test-Time
Scaling in Reasoning Models"* (NeurIPS 2025, Ghosal et al.). That paper shows **math
accuracy rises then falls** as you force more thinking tokens (overthinking → variance →
"mirage"). The safety literature disagrees on sign: OpenAI's *Trading Inference-Time
Compute for Adversarial Robustness* found more compute → safer, while *SafeChain* /
*Does More Inference-Time Compute Really Help Robustness?* found longer CoT can get
**less** safe (ZeroThink/LessThink often safest). Nobody has plotted the safety
dose–response on a **multimodal** model or crossed it with **image corruption** — that's
our gap.

## Method — TTBC-2 "exact thinking tokens" (`budget_forcing.py`)
Straight from the mirage paper (which follows s1, Muennighoff et al. 2025): whenever the
model tries to emit `</think>` before the budget is hit, **suppress it and append
"Wait"**, forcing more thinking, until the trace is exactly `t` tokens; then release
`</think>` and let it answer. R1-Onevision uses plain Qwen2.5-VL tokens (no special think
token), so suppression is text-level (`stop_strings=["</think>"]` + re-prefill; greedy →
deterministic). We record `think_tokens_actual`, `n_wait_insertions`, `n_forcing_rounds`
per response.

## Pilot design (`run_budget_pilot.py`)
- Model: **R1-Onevision-7B** (the reasoning VLM already used in Part 4).
- Data: first **50 SIUO** samples (same loader/idx as Part 4 → the GPT-4o R/C judge joins on `int(idx)==question_id`).
- Condition: **clean** only (pilot).
- Budgets: `0, 512, 2048, natural` — one JSONL each.
  - `0` = reuse Part 4's **exact `r1_onevision_nothink`** path (empty `<think></think>` prefill).
  - `natural` = reuse Part 4's **exact `r1_onevision`** path (no intervention, 4096 tok).
  - **Sanity check built in:** with greedy decode, `natural` should reproduce Part 4's `siuo_clean_r1_onevision_responses.jsonl` for these 50 items byte-for-byte.

Everything else (Qwen loader/runner, SIUO loader, decode) is reused UNCHANGED.

## Run (Newton)
```
mkdir -p /home/ch169788/experiments/part11/logs
# smoke test FIRST (prints, writes nothing) — confirm the budget is hit + Wait appears + answer follows:
cd ~/llava_cot_eval/experiments/part11_thinking_budget
python run_budget_pilot.py --debug_n 2 --budgets 512,2048

# full pilot:
sbatch jobs/pilot.sbatch          # 50 samples x {0,512,2048,natural}
```
Output: `results/siuo_clean_r1onevision_budget<B>_responses.jsonl` (per-idx resume).

## Judge (same GPT-4o R/C judge as Parts 5–7)
```
cd ~/llava_cot_eval
python judging/eval_hr_table1_siuo_jsonl_R_C.py \
  --entry budget0    experiments/part11_thinking_budget/results/siuo_clean_r1onevision_budget0_responses.jsonl \
  --entry budget512  experiments/part11_thinking_budget/results/siuo_clean_r1onevision_budget512_responses.jsonl \
  --entry budget2048 experiments/part11_thinking_budget/results/siuo_clean_r1onevision_budget2048_responses.jsonl \
  --entry natural    experiments/part11_thinking_budget/results/siuo_clean_r1onevision_budgetnatural_responses.jsonl \
  --summary-out experiments/part11_thinking_budget/results/budget_HR_summary.csv
```
Gives HR_R (reasoning ASR) and HR_C (conclusion ASR) per budget → the dose–response curve.
**The interesting signal:** does unsafe *thinking* (HR_R) grow with budget while the
*conclusion* (HR_C) lags or tracks it? If the pilot curve is non-flat, we expand to the
full 167 SIUO items × {clean, zoom_blur sev2} × the full budget grid
`{0,256,512,1024,2048,4096, natural}`.
