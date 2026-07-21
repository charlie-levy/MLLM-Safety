# Part 14 — decoupling across models (the missing utility half of Table 1)

**Goal:** the safety–capability decoupling is the paper's stated central claim but is
measured on **one** checkpoint (safety-tuned LLaVA-CoT, ten corruptions). The safety
half already exists for six configurations × three corruptions (`tab:siuo-main`); the
**utility** half does not, so no other model can be plotted on the decoupling figure.
This part fills exactly that hole: ScienceQA-250 accuracy under the same ImageNet-C
corruptions, for the same six configurations.

**No API credits.** Grading uses the local LLaMA-3-8B utility judge, on GPU.

Everything is reused, nothing is new science:

| piece | reused from |
|---|---|
| SQA items + prompts | `datasets/scienceqa_250.json` (the set part1 uses) |
| corruption + severity | `corruption_lib.apply_corruption` / `severity_for` |
| grading | `judge_sqa_utility_hf.Judge` (LLaMA-3-8B, CORRECT/INCORRECT) |
| generation per family | `run_eval` (2048) / LlamaV staged 4-turn / `qwen_models` (4096) |
| output schema | identical to `part1/run_sqa.py` jsonl |

---

## 0. Environment

The working `REU` env — **not** the part13 `qwen3vl` env (these are the *old* Qwen2.5
models, which `REU` already runs correctly).

```bash
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
```

## 1. SMOKE TEST — the correctness gate (3 items, nothing written)

**This loads models, so it MUST be inside a GPU session** — on a login node the
`RLIMIT_NPROC ~100` cap makes the torch import hang or segfault, with no error:

```bash
srun --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42,evc44 --time=00:40:00 --pty bash
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh && conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
```

Then run one model per **family**, because the three generation paths differ (expect
2–4 min of silence while BOTH the VLM and the LLaMA grader load):

```bash
cd ~/llava_cot_eval/experiments/part14_decoupling
python run_sqa_multimodel.py --model qwen2_5_vl --corruption clean     --debug_n 3
python run_sqa_multimodel.py --model llamav_o1  --corruption zoom_blur --debug_n 3
python run_sqa_multimodel.py --model llava_cot  --corruption clean     --debug_n 3
```

**Confirm before any full run:**
1. Each response actually answers the question (LlamaV-o1 in particular must not stop
   at a planning preamble — that is what the staged pipeline exists to prevent).
2. `predicted=` is a letter A–D, not empty, for most items.
3. `judge_correct=` is `True`/`False`, not `None` (`None` means the judge failed to
   parse — if it is mostly `None`, stop and report, the accuracy would be garbage).

## 2. Full runs — 6 models × 4 conditions = 24 jobs (250 items each)

Reasoning models first: they are the ones with no utility data at all, and they carry
the novel part of the claim.

```bash
for M in qwen2_5_vl r1_onevision llamav_o1 llava_cot base_llama r1_onevision_nothink; do
  for C in clean zoom_blur snow glass_blur; do
    sbatch --gres=gpu:nvidia_h100_pcie:1 --mem=80G --time=06:00:00 --exclude=evc42,evc44 \
      --job-name="p14_${M}_${C}" \
      --wrap="export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0; \
              conda run -n REU python ~/llava_cot_eval/experiments/part14_decoupling/run_sqa_multimodel.py \
              --model ${M} --corruption ${C}"
  done
done
```
Start with the first three models (12 jobs) if you would rather stage it — those are
the ones with no utility data at all. The last three can follow.

Outputs: `~/experiments/part14/results/sqa_<corruption>_<model>.jsonl`.

**Progress check** (want 250 lines each, 24 files):
```bash
for f in ~/experiments/part14/results/sqa_*.jsonl; do printf "%5s  %s\n" "$(wc -l < "$f")" "$(basename "$f")"; done
```

## 3. What to report

Per model × corruption: accuracy = mean(`correct`). Then, for each of the 18
model×corruption cells in `tab:siuo-main`, pair Δaccuracy (here) against ΔHR_C (there)
and redraw `fig:decoupling` with **every model on it** rather than one.

The claim to test, model by model: are there cells with **zero or positive** utility
change that still show a **positive** safety change? On the single model measured so
far, JPEG/frost/fog were exactly those cells. If that pattern reproduces across
models, the central claim stops resting on one checkpoint. If it does *not* reproduce
— if utility and safety fall together for the reasoning models — that is a real result
too and the claim must be narrowed to say so.
