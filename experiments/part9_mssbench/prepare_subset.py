#!/usr/bin/env python
"""
prepare_subset.py — build a small, reproducible, class-stratified subset of the
MSSBench instruction-following ('if') evaluation, and write it as a flat manifest.

Follows the official protocol (eric-ai-lab/MSSBench, utils/infer_on_data.py) EXACTLY:
  chat:      each query is run once with safe_image_path and once with unsafe_image_path
             (SAME query, two contexts).
  embodied:  each (safe_instruction[i], unsafe_instruction[i]) pair -> safe image + safe
             instr, unsafe image + unsafe instr.

Sampling: for each stratum (task x category) draw `--per_category` DISTINCT records
(fixed --seed), and one random sub-item (query / instruction index) per record. Each
sampled record yields a safe+unsafe PAIR (2 manifest items) so the judge can compute
safe-accuracy and unsafe-accuracy on the same item. Balanced safe/unsafe by construction.

Manifest line (one per eval item):
  uid, pair_id, task, category, context(safe|unsafe), image(<task>/<file>), text, record_idx, sub_idx

  python prepare_subset.py --data_root <mssbench_dir> --out mss_subset.jsonl --per_category 8
"""
import os
import json
import random
import argparse
import collections

CHAT_CATS = ["property", "harmful", "offensive", "illegal"]
EMBODIED_CATS = ["property", "harmful"]


def build_pairs(data):
    """Flatten combined.json into (task, category, record_idx, sub_idx, safe/unsafe image+text)."""
    pairs = []
    for ri, d in enumerate(data["chat"]):
        for qi, query in enumerate(d["queries"]):
            pairs.append(dict(
                task="chat", category=d["Type"], record_idx=ri, sub_idx=qi,
                safe_image="chat/" + d["safe_image_path"],
                unsafe_image="chat/" + d["unsafe_image_path"],
                safe_text=query, unsafe_text=query,   # chat: SAME query, two contexts
            ))
    for ri, d in enumerate(data["embodied"]):
        n = min(len(d["safe_instructions"]), len(d["unsafe_instructions"]))
        for si in range(n):
            pairs.append(dict(
                task="embodied", category=d["category"], record_idx=ri, sub_idx=si,
                safe_image="embodied/" + d["safe"],
                unsafe_image="embodied/" + d["unsafe"],
                safe_text=d["safe_instructions"][si],
                unsafe_text=d["unsafe_instructions"][si],
            ))
    return pairs


def sample_subset(pairs, per_category, seed):
    rng = random.Random(seed)
    by_stratum = collections.defaultdict(lambda: collections.defaultdict(list))  # (task,cat)->record_idx->[pairs]
    for p in pairs:
        by_stratum[(p["task"], p["category"])][p["record_idx"]].append(p)

    chosen = []
    for (task, cat), recs in sorted(by_stratum.items()):
        rec_ids = sorted(recs)
        rng.shuffle(rec_ids)
        for ri in rec_ids[:per_category]:
            chosen.append(rng.choice(recs[ri]))   # one random sub-item per distinct record
    return chosen


def to_items(pair):
    """A pair -> two manifest items (safe context, unsafe context)."""
    base = dict(task=pair["task"], category=pair["category"],
                record_idx=pair["record_idx"], sub_idx=pair["sub_idx"])
    pid = "%s-%s-r%d-s%d" % (pair["task"], pair["category"], pair["record_idx"], pair["sub_idx"])
    out = []
    for ctx in ("safe", "unsafe"):
        out.append(dict(
            uid="%s-%s" % (pid, ctx), pair_id=pid, context=ctx,
            image=pair["%s_image" % ctx], text=pair["%s_text" % ctx], **base))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/home/ch169788/experiments/part9/data",
                    help="dir containing combined.json + chat/ + embodied/")
    ap.add_argument("--out", default="/home/ch169788/experiments/part9/mss_subset.jsonl")
    ap.add_argument("--per_category", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    with open(os.path.join(args.data_root, "combined.json")) as f:
        data = json.load(f)

    pairs = build_pairs(data)
    chosen = sample_subset(pairs, args.per_category, args.seed)
    items = [it for p in chosen for it in to_items(p)]

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    # report
    strat = collections.Counter((it["task"], it["category"]) for it in items)
    ctx = collections.Counter(it["context"] for it in items)
    print("MSSBench 'if' subset -> %s" % args.out)
    print("  pairs=%d  items=%d  (per_category=%d, seed=%d)" % (len(chosen), len(items), args.per_category, args.seed))
    print("  by stratum (items):")
    for k in sorted(strat):
        print("    %-9s %-10s : %d" % (k[0], k[1], strat[k]))
    print("  context balance: %s" % dict(ctx))


if __name__ == "__main__":
    main()
