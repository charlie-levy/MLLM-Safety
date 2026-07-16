#!/usr/bin/env python3
"""
cluster.py — UNSUPERVISED structure discovery over the final answers.

No labels. We TF-IDF every model's final answer (style normalized by removing the
single most model-identifying tokens via stop_words) and:
  1. KMeans (k=10) to surface recurring answer "shapes" (templates / refusal-ish /
     advice-list / scene-description / etc.) and see which clusters fill up under
     corruption.
  2. PCA(2D) coordinates saved for the scatter plot (colored by model & condition).

Outputs:
  tables/cluster_top_terms.csv      top terms per cluster
  tables/cluster_composition.csv    cluster x model and cluster x condition shares
  outputs/pca_coords.csv            per-response 2D coords + cluster id (for plotting)
"""
import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
import lib

lib.ensure_dirs()
df = pd.DataFrame(lib.load_records())
df["answer"] = [lib.strip_tags(lib.split_reasoning_answer(m, r)[1])
                for m, r in zip(df["model"], df["response"])]

vec = TfidfVectorizer(min_df=5, max_df=0.5, stop_words="english", ngram_range=(1, 2))
X = vec.fit_transform(df["answer"].fillna("")).astype(np.float64)  # float64 -> no BLAS overflow
terms = np.array(vec.get_feature_names_out())

K = 10
km = KMeans(n_clusters=K, n_init=10, random_state=0)
df["cluster"] = km.fit_predict(X)

# top terms per cluster (from centroid)
rows = []
for c in range(K):
    centroid = km.cluster_centers_[c]
    top = terms[np.argsort(centroid)[::-1][:12]]
    rows.append(dict(cluster=c, size=int((df["cluster"] == c).sum()),
                     top_terms=", ".join(top)))
ct = pd.DataFrame(rows).sort_values("size", ascending=False)
ct.to_csv(os.path.join(lib.TABLES, "cluster_top_terms.csv"), index=False)

# composition: how each cluster splits by model and by condition
comp_model = pd.crosstab(df["cluster"], df["model"], normalize="index").round(3)
comp_cond = pd.crosstab(df["cluster"], df["condition"], normalize="index").round(3)
comp = comp_model.join(comp_cond)
comp.to_csv(os.path.join(lib.TABLES, "cluster_composition.csv"))

# also: for each cluster, share of rows that are corrupted (vs clean) — does a
# cluster "fill up" only when the image is degraded?
df["is_corrupt"] = (df["condition"] != "clean").astype(int)
fill = df.groupby("cluster")["is_corrupt"].mean().round(3)  # baseline 0.75 if random
ct = ct.merge(fill.rename("corrupt_share").reset_index(), on="cluster")
ct.to_csv(os.path.join(lib.TABLES, "cluster_top_terms.csv"), index=False)

# 2D coords for plotting (TruncatedSVD = LSA; sparse-friendly, no overflow)
svd = TruncatedSVD(n_components=2, random_state=0)
xy = svd.fit_transform(X)
coords = df[["model", "condition", "idx", "cluster", "answer_words"]].copy()
coords["pc1"], coords["pc2"] = xy[:, 0], xy[:, 1]
coords.to_csv(os.path.join(lib.OUT, "pca_coords.csv"), index=False)

pd.set_option("display.width", 200)
print("=== answer clusters (k=%d), baseline corrupt_share=0.75 ===" % K)
print(ct[["cluster", "size", "corrupt_share", "top_terms"]].to_string(index=False))
print("\n=== model x cluster composition (row-normalized) ===")
print(comp_model.to_string())
print("\nLSA(2) explained variance: %.3f / %.3f" % tuple(svd.explained_variance_ratio_[:2]))
print("\nWROTE: tables/cluster_top_terms.csv, cluster_composition.csv, outputs/pca_coords.csv")
