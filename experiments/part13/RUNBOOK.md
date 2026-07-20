# Part 13 (E1) — Qwen3-VL-8B Instruct vs Thinking under corruption

**Goal:** test whether the reasoning-tax + corruption-fragility we found persists in
a current frontier reasoning VLM, and whether reasoning post-training buys
corruption-robust safety. **Primary comparison = two native checkpoints** (the
approach Qwen recommends for vision — no soft-switch, no prefill hack):

| key | reasoning | weights |
|---|---|---|
| `qwen3_vl_instruct` | OFF | `Qwen/Qwen3-VL-8B-Instruct` |
| `qwen3_vl_thinking` | ON (native `<think>`) | `Qwen/Qwen3-VL-8B-Thinking` |

`qwen3_vl_thinking_nothink` (empty-`<think></think>` prefill on the Thinking model)
is an **optional probe only** — Qwen VL Thinking editions often ignore soft
switches, so we don't rest anything on it; the smoke test just tells us if it
happens to work.

Everything else is REUSED UNCHANGED from part4/part8 (SIUO-167 samples, corruptions
+ severities: zoom sev3 / snow sev3 / glass sev5, 4096-token Qwen-family budget,
jsonl schema, GPT-4o R/C judge) → directly comparable to the Qwen2.5-VL row.

**Decoding caveat (verified on the smoke test):** pure greedy makes Qwen3-VL loop
(the Instruct model repeated one block to the token cap). We keep greedy +
`repetition_penalty=1.1`, applied identically to Instruct and Thinking, so the
E1 contrast is unaffected; footnote the absolute HR vs the pure-greedy Qwen2.5-VL
row. Re-run the idx-1002 smoke item after `git pull` to confirm the loop is gone
before launching the 8 jobs.

---

## 0. Isolated env — do NOT touch the working `REU` env

`REU` runs Qwen2.5-VL/Mllama and produced every existing number. Qwen3-VL needs
`transformers>=4.57` (`Qwen3VLForConditionalGeneration`). Build a separate env:

```bash
conda create -y -n qwen3vl python=3.11
conda activate qwen3vl
pip install "transformers>=4.57" accelerate qwen-vl-utils pillow
pip install torch --index-url https://download.pytorch.org/whl/cu121   # match cluster CUDA
python -c "from transformers import Qwen3VLForConditionalGeneration; print('class OK')"
```
If the class import fails, the driver falls back to `AutoModelForImageTextToText`
(also fine) — but if BOTH fail, transformers is too old; upgrade before continuing.

## 1. Download weights ONCE on the login node (compute nodes are offline)

```bash
conda activate qwen3vl
HF_HUB_ENABLE_HF_TRANSFER=0 python - <<'PY'
from huggingface_hub import snapshot_download
for r in ["Qwen/Qwen3-VL-8B-Instruct", "Qwen/Qwen3-VL-8B-Thinking"]:
    snapshot_download(r, max_workers=1)   # max_workers=1 dodges the login-node proc cap
    print("cached", r)
PY
```

## 2. SMOKE TEST — the correctness gate (2 items each, GPU, nothing written)

```bash
# inside an srun GPU session, qwen3vl env, offline vars set:
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
cd ~/llava_cot_eval/experiments/part13
python run_inference_qwen3.py --model qwen3_vl_thinking --condition clean --debug_n 2
python run_inference_qwen3.py --model qwen3_vl_instruct --condition clean --debug_n 2
# optional probe (only if curious whether same-weights suppression works here):
python run_inference_qwen3.py --model qwen3_vl_thinking_nothink --condition clean --debug_n 2
```
**Paste all outputs to Claude. CONFIRM before any full run:**
1. `thinking` shows a `<think>…</think>` block, then a final answer.
2. `instruct` is a direct answer with **no** `<think>` block.
3. Neither response ends mid-sentence (if it does, the 4096 budget truncated the
   verdict → raise `QWEN_MAX_NEW_TOKENS`; this is the MSR-2048 lesson).
4. (probe) whether `thinking_nothink` actually dropped the `<think>` block.

## 3. Full runs — 2 models × 4 conditions = 8 jobs (SIUO-167 each)

```bash
for M in qwen3_vl_instruct qwen3_vl_thinking; do
  for C in clean zoom_blur snow glass_blur; do
    sbatch --gres=gpu:nvidia_h100_pcie:1 --mem=80G --time=06:00:00 --exclude=evc42,evc44 \
      --job-name="p13_${M}_${C}" \
      --wrap="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0; \
              conda run -n qwen3vl python ~/llava_cot_eval/experiments/part13/run_inference_qwen3.py \
              --model ${M} --condition ${C}"
  done
done
```
Outputs: `~/experiments/part13/results/siuo_<condition>_<model>_responses.jsonl`.

## 4. Judge — reuse the GPT-4o R/C judge (needs OPENAI credits + PYTHONUNBUFFERED)

Point `judging/eval_hr_table1_siuo_jsonl_R_C.py` at the 8 part13 jsonls exactly like
`run_part8_ext_judge.sh` (one `--responses` per model/condition; per-item cache
resumes). **Scan the CSV for any `0.0,0.0` rows before use** — that's the
out-of-credits poison signature; if present, the API failed mid-run, fix credits
and re-judge (the fail-loud guard should abort now instead of poisoning).

## 5. What to report

HR_R and HR_C for `instruct` and `thinking` × 4 conditions, plus **Δ(clean→corrupt)**
per model. Headline tests for the paper:
- Baseline: is `thinking` any safer than `instruct` on clean? (expect: no / worse.)
- Robustness: does `thinking` degrade **less** under corruption than `instruct`?
  (R1 hinted reasoning buffers the *marginal* effect; does a frontier VL agree?)
- Leakage: is HR_R ≥ HR_C for `thinking`, as with our other reasoners?

Send Claude the judged CSV and it builds the new table + prose for the reasoning
section.
