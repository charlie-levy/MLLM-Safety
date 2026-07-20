#!/usr/bin/env python3
"""Join the filled human audit sheet with the GPT-4o judge cache and report
Cohen's kappa + a confusion matrix on the final-answer (conclusion) verdict.
Run on Newton (or wherever the judge caches live).

  python compute_judge_kappa.py REU_WACV/audit/judge_audit_sheet.csv \
      --cache 'results/**/siuo_R_C_judge_cache.json'

Merges every cache matched by --cache (glob, recursive). Cache keys are
'{model}|siuo|{idx}|conclusion' -> {"harmful": bool, ...}. If a sheet row's
model name doesn't key the cache, pass --map 'sheetname=judgename,...'.
No numpy/sklearn needed (login-node safe)."""
import argparse, csv, json, glob, sys
from collections import Counter


def load_caches(patterns):
    cache = {}
    files = []
    for p in patterns:
        files += glob.glob(p, recursive=True)
    for f in files:
        cache.update(json.load(open(f)))
    return cache, files


def kappa(pairs):
    """Cohen's kappa for binary labels. pairs = list of (human, judge) in {0,1}."""
    n = len(pairs)
    po = sum(1 for h, j in pairs if h == j) / n
    ph1 = sum(h for h, _ in pairs) / n
    pj1 = sum(j for _, j in pairs) / n
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    return (po - pe) / (1 - pe) if pe != 1 else 1.0, po


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("sheet")
    ap.add_argument("--cache", action="append", required=True)
    ap.add_argument("--map", default="")
    args = ap.parse_args()

    remap = dict(kv.split("=") for kv in args.map.split(",") if "=" in kv)
    cache, files = load_caches(args.cache)
    print(f"merged {len(cache)} cache entries from {len(files)} file(s)")

    rows = list(csv.DictReader(open(args.sheet)))
    labeled = [r for r in rows if r["human_harmful"].strip() in ("0", "1")]
    if not labeled:
        sys.exit("No filled rows: fill the human_harmful column (0/1) first.")

    pairs, missing = [], []
    for r in labeled:
        m = remap.get(r["model"], r["model"])
        key = f"{m}|siuo|{r['idx']}|conclusion"
        if key not in cache:
            missing.append(key); continue
        pairs.append((int(r["human_harmful"]), int(bool(cache[key]["harmful"]))))

    if missing:
        print(f"\nWARNING: {len(missing)} rows had no cache key, e.g. {missing[:3]}")
        print("cache model names present:",
              sorted({k.split('|')[0] for k in cache}))
        print("-> fix with --map 'sheetname=judgename,...' and re-run\n")

    k, po = kappa(pairs)
    cm = Counter((h, j) for h, j in pairs)  # (human, judge)
    print(f"n labeled & matched = {len(pairs)}")
    print(f"raw agreement       = {po:.3f}")
    print(f"Cohen's kappa       = {k:.3f}")
    print("confusion (human,judge): "
          f"both-harmful={cm[(1,1)]}  both-safe={cm[(0,0)]}  "
          f"human-harm/judge-safe={cm[(1,0)]}  human-safe/judge-harm={cm[(0,1)]}")


if __name__ == "__main__":
    main()
