#!/usr/bin/env python3
"""
consistency.py — per-idx response stability from clean -> each corruption.

For every (model, idx) we compare the CLEAN answer to the answer under each
corruption and measure how much the text moved:
  * jaccard       token-set overlap (1 = identical vocabulary, 0 = disjoint)
  * tfidf_cosine  cosine similarity of TF-IDF vectors (semantic-drift proxy,
                  vectorizer fit PER MODEL so style is held constant)
  * len_ratio     corrupted answer words / clean answer words
  * first_changed did the first sentence change (normalized)
  * bucket        stable (cos>=.6) / partial (.3-.6) / changed (<.3)

Outputs:
  tables/drift_per_response.csv      one row per (model, condition!=clean, idx)
  tables/drift_summary.csv           mean drift + bucket mix per model x corruption
  tables/most_changed_examples.csv   the lowest-similarity idx per model (for eyeballing)
"""
import os, re
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import lib

lib.ensure_dirs()
recs = lib.load_records()
df = pd.DataFrame(recs)
df["answer"] = [lib.strip_tags(lib.split_reasoning_answer(m, r)[1])
                for m, r in zip(df["model"], df["response"])]


def norm_first_sentence(t):
    s = re.split(r"[.!?]+", t.strip())
    return re.sub(r"\s+", " ", (s[0] if s else "").lower()).strip()


def jacc(a, b):
    A, B = set(lib.words(a)), set(lib.words(b))
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B) if (A | B) else 0.0


rows = []
examples = []
for model in lib.MODELS:
    sub = df[df["model"] == model]
    # per-model TF-IDF so cosine reflects content drift, not cross-model style
    vec = TfidfVectorizer(min_df=2, stop_words="english")
    X = vec.fit_transform(sub["answer"].fillna(""))
    sub = sub.assign(_row=np.arange(len(sub)))
    by_cond = {c: sub[sub["condition"] == c].set_index("idx") for c in lib.CONDITIONS}
    clean = by_cond["clean"]
    for cond in lib.CONDITIONS:
        if cond == "clean":
            continue
        cur = by_cond[cond]
        for idx in clean.index:
            if idx not in cur.index:
                continue
            a0, a1 = clean.loc[idx, "answer"], cur.loc[idx, "answer"]
            cos = float(cosine_similarity(X[clean.loc[idx, "_row"]], X[cur.loc[idx, "_row"]])[0, 0])
            w0 = max(1, len(lib.words(a0)))
            w1 = len(lib.words(a1))
            bucket = "stable" if cos >= 0.6 else ("partial" if cos >= 0.3 else "changed")
            rows.append(dict(model=model, condition=cond, idx=idx,
                             jaccard=round(jacc(a0, a1), 4), tfidf_cosine=round(cos, 4),
                             len_ratio=round(w1 / w0, 3),
                             first_changed=int(norm_first_sentence(a0) != norm_first_sentence(a1)),
                             bucket=bucket,
                             category=clean.loc[idx, "category"]))

drift = pd.DataFrame(rows)
drift.to_csv(os.path.join(lib.TABLES, "drift_per_response.csv"), index=False)

# ---- summary per model x corruption ----
summ = drift.groupby(["model", "condition"]).agg(
    mean_jaccard=("jaccard", "mean"),
    mean_cosine=("tfidf_cosine", "mean"),
    mean_len_ratio=("len_ratio", "mean"),
    first_sentence_change_rate=("first_changed", "mean"),
    stable=("bucket", lambda s: (s == "stable").mean()),
    partial=("bucket", lambda s: (s == "partial").mean()),
    changed=("bucket", lambda s: (s == "changed").mean()),
).round(3).reset_index()
summ.to_csv(os.path.join(lib.TABLES, "drift_summary.csv"), index=False)

# ---- most-changed examples (lowest cosine) per model ----
for model in lib.MODELS:
    g = drift[drift["model"] == model].nsmallest(5, "tfidf_cosine")
    for _, r in g.iterrows():
        examples.append(r)
pd.DataFrame(examples).to_csv(os.path.join(lib.TABLES, "most_changed_examples.csv"), index=False)

pd.set_option("display.width", 160)
pd.set_option("display.float_format", lambda v: "%.3f" % v)
print("=== DRIFT clean -> corruption (per-idx, content similarity) ===")
print(summ.to_string(index=False))
print("\nReading: mean_cosine ~1 = response barely moves when image is corrupted;")
print("'changed' column = fraction of prompts whose answer content substantially flipped.")
print("\nWROTE: tables/drift_per_response.csv, drift_summary.csv, most_changed_examples.csv")
