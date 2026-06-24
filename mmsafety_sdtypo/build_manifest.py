#!/usr/bin/env python
"""
build_manifest.py — read MM-SafetyBench SD_TYPO (Sex + Physical_Harm) parquet
files DIRECTLY (the `datasets` library's split-generation thread-bombs the login
node's RLIMIT_NPROC, so we bypass it), concatenate to 253 samples, verify counts,
write samples_manifest.json, and print the first 5 questions per subset.

Run with threads capped:
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python3 mmsafety_sdtypo/build_manifest.py
"""
import os
import json
import glob
import argparse
import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import pyarrow.parquet as pq   # noqa: E402

SUBSETS = [("Sex", 109), ("Physical_Harm", 144)]   # order fixes idx: Sex 0-108, PHarm 109-252
HARM_KW = ["kill", "bomb", "weapon", "gun", "drug", "hack", "steal", "explos", "attack",
           "rape", "abuse", "suicide", "poison", "murder", "assault", "traffick", "counterfeit"]


def find_parquet(sub):
    pats = glob.glob(os.path.expanduser("~/.cache/huggingface/**/%s/SD_TYPO.parquet" % sub),
                     recursive=True)
    if not pats:
        raise FileNotFoundError("no SD_TYPO.parquet cached for %s (hf download it first)" % sub)
    return pats[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default="/home/ch169788/mmsafety_sdtypo")
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    samples = []
    for sub, exp in SUBSETS:
        t = pq.read_table(find_parquet(sub))
        assert t.num_rows == exp, "Expected %d %s rows, got %d" % (exp, sub, t.num_rows)
        ids = t.column("id").to_pylist()
        qs = t.column("question").to_pylist()
        base = len(samples)
        for i, (sid, q) in enumerate(zip(ids, qs)):
            samples.append({"idx": base + i, "subset": sub, "id": sid, "question": q})
        print("[%s] %d rows -> idx %d..%d" % (sub, t.num_rows, base, len(samples) - 1))

    assert len(samples) == 253, "Expected 253 total, got %d" % len(samples)
    path = os.path.join(args.out_dir, "samples_manifest.json")
    with open(path, "w") as f:
        json.dump(samples, f, indent=2)
    print("\nwrote %s  (%d samples)\n" % (path, len(samples)))

    for sub, _ in SUBSETS:
        print("=== first 5 questions: %s ===" % sub)
        for s in [x for x in samples if x["subset"] == sub][:5]:
            flag = [k for k in HARM_KW if k in s["question"].lower()]
            tag = ("   <<FLAG harmful kw: %s>>" % ",".join(flag)) if flag else ""
            print("  idx %d: %s%s" % (s["idx"], s["question"][:140], tag))
        print()


if __name__ == "__main__":
    main()