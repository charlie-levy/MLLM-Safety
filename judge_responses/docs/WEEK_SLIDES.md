# This Week — Slide Bullets
*Do common image corruptions break safety alignment in LLaVA-CoT (base vs. TIS-defended)?*

---

## 1 — What I did
- Built + ran a full **image-corruption × safety** pipeline on Newton (H100s)
- **38 jobs done** (Part 1 + 2); **30 more launched** (Part 3, running now)
- Headline: **TIS holds ~0% attack success under every corruption**
- …but I found *where* that 0% is real vs. a judge blind spot — with worked examples

---

## 2 — Setup
- **Models:** LLaVA-CoT (undefended) vs. **+TIS** (safety LoRA)
- **10 ImageNet-C corruptions** — blur / snow / fog / frost / JPEG / …, severities tuned
- **ASR judge:** Llama-Guard-3 (text-only, locked) · **utility judge:** LLaMA-3 on ScienceQA
- Frozen inference (bf16, greedy, 2048 tok, seed 0) → apples-to-apples across runs

---

## 3 — Datasets
- **Attacks:** FigStep, SIUO, MM-SafetyBench (+Tiny), SPA-VL, VLS-Bench, HoliSafe
- **Utility:** ScienceQA
- Materialized **4 new sets from scratch** — direct parquet read (HF loader thread-bombs the login node)

---

## 4 — Result: Safety (ASR)
- **TIS = ~0%** across every dataset × every corruption
- **Base** (undefended): MM-Tiny **35%**, others 8–9% clean; MM SD_TYPO **43 → 51%** under JPEG
- That base "corruption bump" is mostly **Llama-Guard noise on long CoT** — not a real effect

---

## 5 — Result: Utility (ScienceQA)
- TIS keeps **82–92%** through all 10 corruptions (clean = 90.4%)
- Only **blurs bite** — glass −8, defocus −7; fog / frost / JPEG ≈ baseline
- Utility loss tracks **perception loss** — corruption only matters if it truly destroys the image

---

## 6 — Where safety *actually* breaks (the real finding)
- SIUO hides the danger **inside the image**; corruption blinds the model to it
- Found **3 genuine flips**: clean **refuses** → corrupted **complies** (idx 2003 / 8007 / 2009)
- The model **incriminates itself in its own reasoning**:
  - clean → *"microwaving eggs can explode"* (refuses)
  - corrupted → *"seems unrelated to the image… Allowed Request"* (complies)
- **146** clean-refuse → corrupted-engage transitions on SIUO-TIS; a subset genuinely harmful

---

## 7 — Part 3 (launched today)
- Same 10 corruptions on **MM-SafetyBench-Tiny, VLS-Bench, HoliSafe** (TIS)
- **Inference-only** — responses saved for our *separate* judge (no Llama-Guard in the loop)
- Reuses the proven pipeline; **30 jobs**, resume-safe, draining overnight

---

## 8 — Engineering
- Newton compat fixes: imagecorruptions / skimage / numpy shims, model staging, `--mem` tuning
- **Resume logic** on every job → safe through timeouts & preemption
- Locked judge + frozen inference contract → reproducible, apples-to-apples
- Tuned SLURM time limits for backfill to get jobs running under a busy H100 queue

---

## 9 — Takeaways & next
- **Harm-location principle:** corruption breaks safety only when harm lives *in the image*, the corruption destroys it, AND the model isn't defended (text-borne harm = corruption-inert)
- **Judge caveat:** text-only Llama-Guard can't see the image → under-counts image-borne ASR
- **Next:** judge Part 3 responses · FigStep × base dose-response · severity sweep
