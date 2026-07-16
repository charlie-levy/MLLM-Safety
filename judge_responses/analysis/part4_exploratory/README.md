# part4_exploratory

Exploratory, **label-free** behavioral analysis of the Part-4 SIUO responses
(4 models × 4 image corruptions × 167 prompts). Discovers how visual corruption
changes *how* the models write — **no ASR, no judge, no safety labels.**

**Start here:** [`FINDINGS.md`](FINDINGS.md) — the write-up (metrics, surprising findings, hypotheses, what's publication-worthy).

## Layout
```
lib.py              shared loaders + surface-metric functions + reasoning/answer splitter
build_master.py     -> per-response metrics, model×condition pivots
consistency.py      -> per-idx drift clean→corrupted (jaccard, TF-IDF cosine, buckets)
cluster.py          -> unsupervised TF-IDF/KMeans clusters + LSA-2D coords
plots.py            -> outputs/figures/01..10_*.png
outputs/
  master_responses.csv      one row per (model, condition, idx) + all metrics
  pca_coords.csv            LSA 2D coords per response
  tables/                   pivots, cell means, drift summaries, cluster tables
  figures/                  10 PNGs (titles state the takeaway)
```

## Reproduce
```bash
cd analysis/part4_exploratory
python3 build_master.py && python3 consistency.py && python3 cluster.py && python3 plots.py
```
Needs: pandas, numpy, scikit-learn, matplotlib (all present in the local env).

## Figure index
| file | shows |
|---|---|
| 01_metric_heatmaps | length / reasoning / perception-fail / refusal-like / uncertainty / repeat, model×condition |
| 02_answer_length_violins | answer-length *distributions* (spread, not just means) |
| 03_length_trajectory | verbosity drift clean→corrupted |
| 04_reasoning_vs_answer | reasoning length does not collapse (it inflates) |
| 05_drift_cosine_heatmap | content drift; zoom_blur moves answers most |
| 06_drift_buckets | stable/partial/changed per model×corruption |
| 07_two_failure_modes | ⭐ perception-failure vs silent semantic drift |
| 08_lsa_scatter | answers cluster by model, not by corruption |
| 09_style_radar | per-model stylistic fingerprints |
| 10_first_sentence_change | the opening sentence is the least stable part |
```
