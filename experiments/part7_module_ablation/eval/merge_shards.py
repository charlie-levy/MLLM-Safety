#!/usr/bin/env python
"""
merge_shards.py — reassemble data-parallel shards into ONE file, identical to a 1-GPU run.

Concatenates shard_<model>_*of*.jsonl (from run_sqa.py / run_mathvista.py --nshards) into a
single output, de-duplicating by key (idx for SQA, pid for MathVista) and verifying the count.
Generation was single-sample per item, so the merged file == running all items on one GPU.

  # ScienceQA (expect 4241)
  python merge_shards.py --dir /home/ch169788/experiments/part7/sqa       --model llm --key idx --out raw_llm.jsonl            --expect 4241
  # MathVista (expect 1000)
  python merge_shards.py --dir /home/ch169788/experiments/part7/mathvista --model llm --key pid --out responses_llm.jsonl      --expect 1000
"""
import os
import sys
import json
import glob
import argparse


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--key", default="idx", help="dedup key: idx (SQA) or pid (MathVista)")
    ap.add_argument("--out", required=True, help="output filename (written inside --dir)")
    ap.add_argument("--expect", type=int, default=0, help="expected total item count (0 = skip check)")
    args = ap.parse_args()

    shards = sorted(glob.glob(os.path.join(args.dir, "shard_%s_*of*.jsonl" % args.model)))
    if not shards:
        sys.exit("no shard_%s_*of*.jsonl found in %s" % (args.model, args.dir))

    seen = {}
    for sh in shards:
        with open(sh) as f:
            for line in f:
                if not line.strip():
                    continue
                r = json.loads(line)
                seen[r[args.key]] = json.dumps(r, ensure_ascii=False)

    out_path = os.path.join(args.dir, args.out)
    with open(out_path, "w", encoding="utf-8") as f:
        for k in seen:
            f.write(seen[k] + "\n")

    print("merged %d shard(s) -> %s   (%d unique %s)" % (len(shards), out_path, len(seen), args.key))
    if args.expect:
        if len(seen) == args.expect:
            print("  count OK ✅ (%d == %d)" % (len(seen), args.expect))
        else:
            print("  ⚠️  count MISMATCH: got %d, expected %d — some shards incomplete; "
                  "resubmit the missing shard array tasks (per-idx resume will fill gaps)."
                  % (len(seen), args.expect))


if __name__ == "__main__":
    main()
