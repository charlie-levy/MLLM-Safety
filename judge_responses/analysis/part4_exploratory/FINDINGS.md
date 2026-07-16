# Part 4 — Exploratory analysis: how does image corruption change VLM *behavior*?

**Data:** 16 response files = 4 models × 4 image conditions, SIUO (167 prompts each), 2,672 responses total.
**Models:** `base_llama` (Llama-3.2-11B, base) · `llava_cot` (LLaVA-CoT, reasoning) · `qwen2_5_vl` (Qwen2.5-VL-7B, base) · `r1_onevision` (R1-Onevision-7B, reasoning).
**Conditions:** `clean`, `zoom_blur` (sev3), `snow` (sev3), `glass_blur` (sev5).

**Hard rules honored:** no ASR, no external judge, no safety classification, no assumed labels. Every number here is a **surface / stylometric feature** computed directly from the text the models already wrote. `refusal_like` and `perception_failure` are *linguistic surface flags*, **not** safety verdicts.

> The goal was discovery of unexpected *behavioral* changes under visual corruption — not benchmarking.

---

## How to reproduce (scripts in this folder)

| Step | Script | Produces |
|---|---|---|
| 1. per-response metrics + pivots | `build_master.py` | `outputs/master_responses.csv`, `outputs/tables/{cell_means,pivot_*,exact_duplicate_answers}.csv` |
| 2. per-idx drift clean→corrupted | `consistency.py` | `outputs/tables/{drift_per_response,drift_summary,most_changed_examples}.csv` |
| 3. unsupervised clustering | `cluster.py` | `outputs/tables/{cluster_top_terms,cluster_composition}.csv`, `outputs/pca_coords.csv` |
| 4. figures | `plots.py` | `outputs/figures/01..10_*.png` |
| shared metric library | `lib.py` | (loaders, reasoning/answer split, lexicons) |

Run order: `python3 build_master.py && python3 consistency.py && python3 cluster.py && python3 plots.py`.

---

## The 12 analyses run (and where to look)

1. **Answer/reasoning length distributions** per model×condition → `pivot_answer_words.csv`, `pivot_reasoning_words.csv`, fig 01/02.
2. **Verbosity drift** clean→corrupted → fig 03.
3. **Reasoning vs final-answer length** (collapse check) → fig 04, `pivot_reasoning_words.csv`.
4. **Discourse connectors** (because/therefore/thus/first/then…) density → `pivot_connector_density.csv`, fig 09.
5. **Uncertainty/hedging lexicon** density → `pivot_uncertainty_density.csv`, fig 01.
6. **Refusal-like surface phrasing** rate → `pivot_refusal_like.csv`, fig 01.
7. **Lexical diversity** (TTR + length-controlled MATTR) and **bigram-repeat / mode-collapse proxy** → `cell_means.csv`, fig 01.
8. **Per-idx content drift** clean→corrupted: token Jaccard + TF-IDF cosine + stable/partial/changed buckets → `drift_summary.csv`, fig 05/06.
9. **Structural stability**: first-sentence change rate; reasoning-scaffold intact rate; empty-answer rate → `drift_summary.csv`, `pivot_structure_intact.csv`, `pivot_answer_empty.csv`, fig 10.
10. **Two-failure-mode map**: overt perception-failure vs silent semantic drift, per corruption → fig 07.
11. **Unsupervised clustering** (TF-IDF + KMeans k=10 + LSA-2D) of answers; cluster composition by model/condition → fig 08, cluster tables.
12. **Stylistic fingerprints** (radar of normalized metrics) → fig 09.

---

## Headline findings (most surprising first)

### F1 — Corruption induces **two qualitatively different failure modes**, and ImageNet-C *severity does not predict which* ⭐ most publication-worthy
- `glass_blur` (severity 5) → **overt perceptual failure**: highest `perception_failure` (base_llama 0.06, llava_cot 0.04) — the model says some variant of "I can't make it out."
- `zoom_blur` (severity **3**) → **silent semantic drift**: the *lowest* clean↔corrupted content similarity (cosine: base_llama 0.38, llava_cot 0.43) yet **~0 perception failure** — the model stays confident and just answers about a *different* image.
- So the milder-severity corruption causes the *larger* behavioral change. `snow` is the gentlest on both axes.
- See `07_two_failure_modes.png`: glass_blur points sit high (perceptual failure), zoom_blur points sit far right (content drift). They occupy different corners.

### F2 — **No length collapse — corruption mildly *inflates* output** (incl. reasoning)
- Most models get *longer* under corruption, not shorter: llava_cot answer 96→108 words (snow, +12%), qwen 218→229, r1 116→122. Only base_llama dips slightly (175→168).
- **Reasoning does not collapse**: llava_cot reasoning 116→130 words (+12%) under corruption; r1_onevision ~360 throughout. If anything, a degraded image makes the CoT model *deliberate more*. (fig 03/04)

### F3 — **Family identity dominates style far more than corruption does**
- Refusal-like phrasing is a **family-level prior**: Llama family 20–27% vs Qwen family 1–4%, essentially flat across conditions (fig 01 bottom-left).
- Unsupervised: the "deflect to a professional / cite illegality / give guidance" cluster (#8) is **75% Llama-family**; the **degenerate/garbled cluster (#1) is 100% `base_llama`** (repeated "feature extraction", language-switching to Chinese fragments).
- LSA scatter (fig 08): answers separate cleanly **by model**, not by condition.

### F4 — **Reasoning did not buy robustness — in the Qwen pair it *reduced* output stability**
- Qwen **base** (`qwen2_5_vl`) is the **most corruption-stable** model: cosine 0.54–0.58, only ~12% of prompts "changed", length ratio ~1.1.
- Adding reasoning (`r1_onevision`) made the *same family* the **least stable**: cosine 0.40–0.43, 22–29% "changed", and by far the most "partial" drift.
- Llama base vs reasoning are both mid-stability. So within a family, the reasoning variant was *not* more visually robust. (fig 05/06)

### F5 — **Corruption rewrites answers silently and idiosyncratically — there is no "failure template"**
- Every KMeans cluster sits at the baseline 0.75 corrupt-share (no cluster "fills up" only when corrupted); clusters are topical (train, statue, food, driving…), not corruption-induced.
- Yet per-prompt drift is large (20–37% of prompts "changed"). Conclusion: corruption changes *what* a model says prompt-by-prompt **without** collapsing into recognizable boilerplate — the drift is distributed and semantic, so it would be **invisible to a template/keyword detector**. (fig 08)

### F6 — **The opening sentence is the least stable part of the response**
- First-sentence change rate is 74–98% across *all* model×corruption cells — even for the most content-stable model (qwen, 74–80%). The first sentence flips even when overall content barely moves (fig 10). Hypothesis: the opening is where visual grounding is injected; later sentences are more template-driven.

### F7 — **LLaVA-CoT keeps its scaffold *more* when the image is corrupted**
- `structure_intact` (did the `<SUMMARY>/<CAPTION>/<REASONING>/<CONCLUSION>` scaffold appear): clean **0.77** → corrupted **0.85–0.87**. On clean images it more often answers directly; degraded images push it back onto the full CoT structure. `r1_onevision` keeps `<think>` ~100% regardless.

### F8 — **Mild hedging rise under the two blur corruptions**
- Uncertainty density rises most under `zoom_blur`/`glass_blur` (base_llama 0.22→0.48–0.51 per 100 words; qwen 0.42→0.54), least under `snow` — weak but consistent, and tracks F1's "blur is more perceptually confusing."

---

## Hypotheses (for later, testable work)

- **H1 (safety-relevant).** *Silent drift* (zoom_blur) is plausibly more dangerous than *overt perception failure* (glass_blur): the model stays fluent and confident while its visual grounding has shifted. Worth checking against your separate safety judge: does ASR rise more under the *silent-drift* corruption than the *overt-failure* one?
- **H2.** Reasoning traces act as a **length amplifier under uncertainty**, not an error-catching mechanism — they inflate rather than self-correct (F2 + F4).
- **H3.** Refusal/deflection phrasing is an **alignment-tuning prior at the family level**, nearly independent of the pixels (F3).
- **H4.** Failure *mode* is set by corruption **geometry, not severity**: zoom_blur's global radial warp preserves a plausible-but-wrong scene (confident drift), while glass_blur's local pixel scramble destroys legibility (abstention). Severity ranking is the wrong axis (F1).
- **H5.** Visual grounding is concentrated in the **first sentence**; the tail of the response is comparatively prompt/template-driven (F6).

---

## What's most publication-worthy

1. **F1 — two dissociated failure modes; severity ⊥ drift.** A clean, novel *behavioral* phenomenon with a mechanistic story (H4), and it reframes "robustness" as multi-modal rather than a single degradation curve.
2. **F5 — corruption as a silent, untemplated rewrite.** Motivates *why* a safety study can't rely on perception-failure / keyword detection: the dangerous case leaves no surface signature.
3. **F4 — reasoning ≠ visual robustness.** Directly relevant to the reasoning-vs-base framing of the whole project; the Qwen pair shows reasoning can *reduce* output stability under corruption.

F2, F3, F6, F7 are strong supporting results / good figures.

---

## Caveats (so nobody over-reads this)

- All metrics are **surface text features**, not meaning or safety. "Drift" = lexical/TF-IDF change, a proxy for semantic change.
- `perception_failure` is the pipeline's **heuristic** flag; treat rates as relative, not absolute.
- The `mean_len_ratio` column in `drift_summary.csv` is **right-skewed** (a few very short clean answers blow up the ratio), so length-change claims here lean on the *mean answer length* and *cosine*, not that column.
- LSA explained variance is low (typical for sparse TF-IDF); the scatter is for *qualitative* separation only.
- n = 167 prompts per cell; differences of ~1–2% (e.g. perception_failure) are a handful of responses — directional, not definitive.
