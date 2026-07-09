# Part 8 — Prompt-based defense (training-free)

**Question.** Corruption (blur/snow/…) breaks VLM safety (Parts 1/3/4). TIS fixes it
but needs fine-tuning (Parts 5–7). **Can a system prompt recover safety for free?**
And if it helps — is it the *blur-awareness* that matters, or just any safety nudge?

## Design
- **Dataset / metric:** SIUO-167, responses only. Judge HR_R / HR_C **externally**
  (same as Parts 4–7). No judging here.
- **Decode:** identical frozen greedy path as Part 4 (Llama 2048 tok / Qwen 4096 tok).
  The **only** change is a prepended safety **system prompt**. `--prompt none` calls
  the original Part-4 functions verbatim (that cell == Part 4).
- **Prompt variants** (`prompts.py`):
  | variant | mentions blur? | in small grid? | isolates |
  |---|---|---|---|
  | `none` | — | reuse Part 4 | undefended baseline (not re-run) |
  | `safety` | no | ✅ | does *any* safety nudge help? |
  | `blur_safe` | **yes** | ✅ | the idea: "image may be degraded, stay cautious" |
  | `perceive` | yes | optional | "if you can't see it clearly, don't guess" |

  **`safety` vs `blur_safe` is the money comparison** — it isolates whether
  blur-awareness specifically beats a generic safety nudge.
- **Model:** `llava_cot` only — the TIS base (→ compare prompt-defense vs the
  fine-tuned TIS) and a reasoning model (HR_R vs HR_C both interpretable). Driver
  supports all 6 Part-4 models; widen `MODELS` in `gen_jobs.py` to expand.
- **Grid (small):** 1 model × 4 conditions × 2 interventions = **8 cells** × 167.

## Files
- `prompts.py` — frozen system-prompt registry.
- `run_inference.py` — driver. `--model --condition --prompt [--debug_n N]`.
- `gen_jobs.py` — materializes static sbatch (one per model×condition×prompt) + submit scripts.
- Output: `siuo_<condition>_<model>_<prompt>_responses.jsonl` (per-idx resume).

## Run (Newton)
```bash
cd ~/llava_cot_eval/experiments/part8_prompt_defense

# 0) DEBUG FIRST — confirm the system prompt actually takes effect (nothing written):
python run_inference.py --model llava_cot --condition zoom_blur --prompt blur_safe --debug_n 1

# 1) materialize jobs (edit PARTITION in gen_jobs.py first if needed):
python gen_jobs.py

# 2) submit all 8 cells:
bash ~/experiments/part8/jobs/submit_part8.sh         # 8 cells (== submit_part8_llava.sh)
```
Then pull `~/experiments/part8/results/*.jsonl` locally and run the external SIUO judge.
Baseline for the figure = Part-4 `none` numbers already in `figures/siuo_asr_scores.json`.