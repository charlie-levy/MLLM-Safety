# Part 10 — Paper-faithful GPT-4o judging: MM-SafetyBench-Tiny · SPA-VL · VLS-Bench

Base **LLaVA-CoT (no TIS)** on three multimodal-safety benchmarks, under **clean** and
**zoom_blur (severity 2)**, each scored by **its own paper's official judge, re-implemented
with GPT-4o**. One model, one decode; only the judge changes per benchmark.

## Inference (identical for all three)
`run_inference.py --dataset {mmsafety_tiny,spa_vl,vls_bench}` — loads base LLaVA-CoT **once**
and generates both conditions. Frozen decode (bf16 / greedy / max_new_tokens 2048 / seed 0 —
the repo's `run_eval` contract). Reads the **Part-2** offline manifest
(`/home/ch169788/experiments/part2/data/<ds>/manifest.jsonl`, 168 / 265 / 500 samples). For
zoom_blur the corruption is applied to the clean image in memory; `image_path` in every output
record points at the **clean source** (so the judges always see the ground-truth-risk image).
Output: `results/<dataset>_<clean|zoom_blur>.jsonl` (responses only, per-idx resume).

## Judges (GPT-4o, temperature 0) — verbatim from each source

| Judge | Source | Modality | Metric |
|---|---|---|---|
| `judge_mmsafety.py`  | isXinLiu/MM-SafetyBench `evaluation.py` | **text-only** (response only) | **ASR = unsafe/total**; 6 scenario templates¹ |
| `judge_vlsbench.py`  | AI45Lab/VLSBench `eval_utils.py` | **multimodal** (image+question+response), structured 3-label | **ASR = Unsafe(3)/total**; safe = refusal(1)+warning(2) |
| `judge_spavl.py`     | SPA-VL arXiv:2406.12030 App. F.1 | **multimodal, pairwise** | clean-vs-blur harmlessness, order-swapped + consistency-filtered |

¹ Templates are selected by scenario prefix: `08`→political-lobbying, `10/11/12`→legal/financial/health,
`13`→gov-decision, all others→general. Exactly the official per-scenario routing.

**SPA-VL note:** the paper's GPT judge is *pairwise* (it has no absolute per-response score; its
absolute Unsafe-Rate uses MD-Judge, a non-GPT classifier). We keep the judge + prompt exactly and
set the pair to (clean, zoom_blur) of the same item, order-swapped, keeping only consistent verdicts
(paper's rule). Reports **blur_more_harmful / tie / clean_more_harmful** and a Harm-Score
(`1·blur_more_harmful + 0.5·tie`). The headline is `blur_more_harmful` = corruption hurt safety.

**Why the multimodal judges see the CLEAN image:** we are testing whether the *model's response*
is safe w.r.t. the true risk; feeding the judge the corrupted image would measure the judge's
robustness, not the model's.

## Run order (on Newton)
```
# 1) inference (GPU) — one job per dataset, both conditions each
sbatch jobs/infer_mmsafety_tiny.sbatch
sbatch jobs/infer_spa_vl.sbatch
sbatch jobs/infer_vls_bench.sbatch

# 2) judging (login node: needs internet + OPENAI_API_KEY; CPU/IO only)
python judge_mmsafety.py --entry clean results/mmsafety_tiny_clean.jsonl \
                         --entry zoom_blur results/mmsafety_tiny_zoom_blur.jsonl
python judge_vlsbench.py --entry clean results/vls_bench_clean.jsonl \
                         --entry zoom_blur results/vls_bench_zoom_blur.jsonl
python judge_spavl.py    --clean results/spa_vl_clean.jsonl --blur results/spa_vl_zoom_blur.jsonl
```
Each judge caches per item (resumable) and writes a `*_summary.csv`. Add `--debug_n 2/3` to any
judge for a no-write smoke test that prints prompts + verdicts.
