#!/usr/bin/env python3
"""
build_master.py — build the per-response master table + per-cell aggregates.

Outputs (to outputs/ and outputs/tables/):
  master_responses.csv        one row per (model, condition, idx) with all metrics
  cell_means.csv              model x condition means of every numeric metric
  pivot_<metric>.csv          4x4 model x condition pivots for headline metrics
  exact_duplicate_answers.csv per-cell count of identical answer strings (template/collapse)
"""
import os, csv
import pandas as pd
import lib

lib.ensure_dirs()
rows = lib.load_records()
df = pd.DataFrame(rows)
print("loaded %d responses (%d cells x %d)" % (len(df), df.groupby(['model', 'condition']).ngroups,
                                               len(df) // df.groupby(['model', 'condition']).ngroups))

# ---- master table (drop the big raw text to keep csv light; keep a short preview) ----
master = df.drop(columns=["response"]).copy()
master.to_csv(os.path.join(lib.OUT, "master_responses.csv"), index=False)

NUMERIC = ["full_words", "answer_words", "reasoning_words", "reasoning_ratio",
           "answer_sentences", "ttr_answer", "mattr_answer", "bigram_repeat",
           "connectors", "connector_density", "uncertainty", "uncertainty_density",
           "has_uncertainty", "refusal_like", "structure_intact", "answer_empty",
           "perception_failure", "resp_chars"]

# ---- per-cell means ----
cell = df.groupby(["model", "condition"])[NUMERIC].mean().reset_index()
cell.to_csv(os.path.join(lib.TABLES, "cell_means.csv"), index=False)

# ---- headline pivots (model x condition) ----
def pivot(metric, agg="mean"):
    p = df.pivot_table(index="model", columns="condition", values=metric, aggfunc=agg)
    p = p.reindex(index=lib.MODELS, columns=lib.CONDITIONS)
    p.to_csv(os.path.join(lib.TABLES, "pivot_%s.csv" % metric))
    return p

for m in ["answer_words", "reasoning_words", "uncertainty_density", "refusal_like",
          "connector_density", "mattr_answer", "bigram_repeat", "perception_failure",
          "answer_empty", "structure_intact"]:
    pivot(m)

# ---- exact-duplicate answers per cell (mode-collapse / template proxy) ----
dup_rows = []
for (model, cond), g in df.groupby(["model", "condition"]):
    ans = g["response"].apply(lambda r: lib.strip_tags(lib.split_reasoning_answer(model, r)[1]).strip().lower())
    n = len(ans)
    uniq = ans.nunique()
    vc = ans.value_counts()
    top_n = int(vc.iloc[0]) if len(vc) else 0
    dup_rows.append(dict(model=model, condition=cond, n=n, unique_answers=uniq,
                         duplicate_rate=round(1 - uniq / n, 4), max_identical=top_n))
pd.DataFrame(dup_rows).to_csv(os.path.join(lib.TABLES, "exact_duplicate_answers.csv"), index=False)

# ---- console summary of the headline pivots ----
pd.set_option("display.width", 140)
pd.set_option("display.float_format", lambda v: "%.2f" % v)
print("\n=== mean ANSWER words (model x condition) ===")
print(pivot("answer_words").round(1))
print("\n=== mean REASONING words (reasoning models only meaningful) ===")
print(pivot("reasoning_words").round(1))
print("\n=== uncertainty density (per 100 answer words) ===")
print(pivot("uncertainty_density").round(2))
print("\n=== refusal-like phrasing rate (surface marker, NOT safety) ===")
print(pivot("refusal_like").round(3))
print("\n=== perception_failure rate (pipeline flag) ===")
print(pivot("perception_failure").round(3))
print("\n=== answer_empty rate (reasoning ran but produced no final answer) ===")
print(pivot("answer_empty").round(3))
print("\n=== structure_intact rate (reasoning models kept their tag scaffold) ===")
print(pivot("structure_intact").round(3))
print("\n=== bigram repeat (mode-collapse proxy) ===")
print(pivot("bigram_repeat").round(3))
print("\nWROTE: master_responses.csv + tables/ (cell_means, pivots, exact_duplicate_answers)")
