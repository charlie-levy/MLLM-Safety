# Part 7 — Module-ablation: where does corruption-robust TIS live (LLM vs Vision Encoder)?

Re-train LLaVA-CoT with Think-in-Safety on a **75% zoom_blur(sev2) / 25% clean** copy of the
TIS dataset, three ways — changing **only which tower gets the LoRA adapter**, everything else
identical to your existing `train_llava_cot_tis_lora.yaml` recipe (LoRA r8/α16/dropout0.05,
lr1e-5, 1.5ep, cosine, warmup0.1, bs1×ga8, bf16, template mllama).

| # | Checkpoint | `freeze_vision_tower` | `freeze_language_model` | LoRA trains |
|---|---|---|---|---|
| 1 | `saves/llava_cot_tis_zoomblur75_llm_lora`    | `true`  | `false` | LLM only (= your existing TIS setup) |
| 2 | `saves/llava_cot_tis_zoomblur75_vision_lora` | `false` | `true`  | Vision encoder only |
| 3 | `saves/llava_cot_tis_zoomblur75_both_lora`   | `false` | `false` | LLM + Vision encoder |

`freeze_multi_modal_projector: true` in all three (projector held constant → the only variable
is LLM-LoRA vs ViT-LoRA).

## Run order

Precondition: `prep_tis_data.py` already ran, so `~/LLaMA-Factory/data/think_in_safety.json`
and its images exist.

```bash
mkdir -p /home/ch169788/experiments/part7/logs
P=~/llava_cot_eval/experiments/part7_module_ablation

# 1) Build the corrupted dataset copy (login node, REU env) + PROOF corruption is applied.
conda activate REU
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
python3 $P/prep/prep_tis_zoomblur75.py          # writes think_in_safety_zoomblur75{,.json,_manifest.csv,_PROOF.png}
# scp the *_PROOF.png locally to SEE the blur; check the printed MAD proof (zoom>>0, clean==0).

# 2) PROVE freezing for all 3 configs BEFORE training (GPU, llamafactory env).
sbatch $P/jobs/verify_freezing.sbatch           # must print PASS ✅ for all 3

# 3) Train (GPU, llamafactory env) — only after step 2 passes.
sbatch $P/jobs/finetune_llm.sbatch
sbatch $P/jobs/finetune_vision.sbatch
sbatch $P/jobs/finetune_both.sbatch
```

Adapters land in `~/LLaMA-Factory/saves/llava_cot_tis_zoomblur75_{llm,vision,both}_lora/`.

## Then (evals — built separately)
- **SIUO** zoom_blur(sev2) whole-set responses for each checkpoint (reuses Part 6 `run_inference`).
- **ScienceQA** (full test split) + **MathVista** (testmini) utility, judged with local Llama-3-8B.
All responses pulled locally under `judge_responses/results/part7_*`.
