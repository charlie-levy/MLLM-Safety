#!/usr/bin/env python3
"""
purge_judge_cache.py — drop cached GPT-4o verdicts for named models so they get
RE-JUDGED on the next run.

Why this exists: the pre-fix judge, on 3 failed API calls (out of credits / API
down), silently cached {"Response": "unharmful"} and moved on. Any model whose
run hit that path scores HR_R=HR_C=0.0 — the poison signature visible as 0.0,0.0
rows in a summary CSV. Because the bad verdict is IN THE CACHE, simply re-running
the judge replays it and reproduces the same 0.0. The cache entries have to be
removed first.

The poison is not distinguishable by content (a genuine 'unharmful' verdict is
byte-identical), so we purge by MODEL NAME — every item for the named models is
re-judged from scratch. Cache keys are "<model>|siuo|<idx>|reasoning|conclusion".

Dry-run by default; pass --apply to write. Always writes a .bak first.

    # see what would go
    python purge_judge_cache.py --cache ~/judging/results_part8/siuo_R_C_judge_cache.json \
        --model clean_r1_onevision_blur_safe --model snow_r1_onevision_safety

    # actually purge
    python purge_judge_cache.py --cache ... --model ... --apply
"""
import argparse
import json
import os
import shutil


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="Path to siuo_R_C_judge_cache.json")
    ap.add_argument("--model", action="append", default=[], required=True,
                    help="Model name to purge (repeatable). Must match the --entry "
                         "name used when judging, e.g. clean_r1_onevision_blur_safe")
    ap.add_argument("--apply", action="store_true",
                    help="Actually write the purged cache (default: dry run)")
    args = ap.parse_args()

    if not os.path.exists(args.cache):
        raise SystemExit("Cache not found: %s" % args.cache)

    with open(args.cache) as f:
        cache = json.load(f)

    # A key belongs to a model iff it starts with "<model>|siuo|". Prefix-matching on
    # the bare name would also catch longer names that merely start the same way
    # (e.g. 'clean_r1_onevision' vs 'clean_r1_onevision_blur_safe'), so anchor on the
    # separator to keep the purge exact.
    doomed, per_model = [], {}
    for m in args.model:
        keys = [k for k in cache if k.startswith(m + "|siuo|")]
        per_model[m] = len(keys)
        doomed.extend(keys)

    print("Cache: %s  (%d entries)" % (args.cache, len(cache)))
    for m in args.model:
        status = "" if per_model[m] else "   <-- NOTHING MATCHED (name typo?)"
        print("  %-45s %5d keys%s" % (m, per_model[m], status))
    print("Total to purge: %d  ->  cache would go %d -> %d"
          % (len(doomed), len(cache), len(cache) - len(doomed)))

    if not any(per_model.values()):
        raise SystemExit("\nNo keys matched any --model. Nothing to do; check the names "
                         "against the 'model' column of the summary CSV.")

    if not args.apply:
        print("\nDRY RUN — nothing written. Re-run with --apply to purge.")
        return

    shutil.copy2(args.cache, args.cache + ".bak")
    for k in doomed:
        del cache[k]
    with open(args.cache, "w") as f:
        json.dump(cache, f, indent=2)
    print("\nBackup -> %s.bak" % args.cache)
    print("Purged %d keys. Re-run the judge; the purged models are re-judged, "
          "everything else resumes from cache." % len(doomed))


if __name__ == "__main__":
    main()
