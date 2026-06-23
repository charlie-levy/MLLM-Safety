#!/usr/bin/env python
"""
interactive_eval.py — load a model ONCE, then run many (dataset/corrupt/pct)
conditions on demand with NO reload between them. For fast interactive PREVIEW
exploration only.

It imports run_eval.py and calls its FROZEN functions verbatim (same load,
generate_one, CORRUPTORS, load_samples, seed) — the ONLY difference is the model
stays resident in memory, so after the one-time load every condition runs
instantly. The full sweep still runs through the frozen run_eval.py sbatch path,
so apples-to-apples is untouched.

  python code/interactive_eval.py --model llava_cot_tis
  > siuo gaussian_blur 10 8        # <dataset> <corrupt> <pct> <n>  -> runs now
  > siuo gaussian_blur 5 8
  > figstep none 0 8
  > quit

Output JSON per condition: results/preview/<dataset>_<model>_<cond>.json
Corrupted images (when pct>0): results/preview/img_<model>_<dataset>_<cond>/
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as RE   # reuse every frozen setting (load / generate_one / corruptors / sampling)


def run_condition(model, processor, dataset, corrupt, pct, n, seed, outdir):
    cond = "clean" if (corrupt == "none" or pct == 0) else "%s%g" % (corrupt, pct)
    corrupt_fn = None if cond == "clean" else RE.CORRUPTORS[corrupt]
    samples = RE.load_samples(dataset, n, seed)

    imgdir = os.path.join(outdir, "img_%s_%s_%s" % (model_tag, dataset, cond))
    if corrupt_fn is not None:
        os.makedirs(imgdir, exist_ok=True)

    print("-" * 70)
    print("  %s | %s | %s | n=%d" % (model_tag, dataset, cond, len(samples)), flush=True)
    print("-" * 70, flush=True)

    records = []
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        image = s["image"]
        if corrupt_fn is not None and image is not None:
            image = corrupt_fn(image, pct)
            image.save(os.path.join(imgdir, "%s.png" % idx))
        resp = RE.generate_one(model, processor, image, s["prompt"])
        records.append({
            "idx": idx, "model": model_tag, "dataset": dataset,
            "corrupt": corrupt, "pct": pct, "condition": cond,
            "category": meta.get("category", ""), "prompt": s["prompt"],
            "image_path": meta.get("image_path", ""), "full_response": resp,
        })
        flat = resp.replace("\n", " ")
        print("[%2d/%d] idx=%s%s" % (i + 1, len(samples), idx,
                                     "   [DEAD?]" if RE.is_dead(resp) else ""), flush=True)
        print("        HEAD: %s" % flat[:150], flush=True)
        print("        TAIL: %s" % flat[-150:], flush=True)

    out = os.path.join(outdir, "%s_%s_%s.json" % (dataset, model_tag, cond))
    with open(out, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    ndead = sum(RE.is_dead(r["full_response"]) for r in records)
    print("\nwrote %s  (%d responses, %d flagged DEAD?)%s\n" % (
        out, len(records), ndead,
        "   <-- severity too high, step DOWN" if ndead else ""), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["base_llama", "llava_cot", "llava_cot_tis"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--outdir", default="results/preview")
    args = ap.parse_args()

    global model_tag
    model_tag = args.model
    os.makedirs(args.outdir, exist_ok=True)

    print("loading %s once (this is the only load)..." % args.model, flush=True)
    model, processor = RE.load(args.model)
    print("\nREADY. enter:  <dataset> <corrupt> <pct> <n>   (or 'quit')", flush=True)
    print("  datasets : figstep | beavertails | siuo")
    print("  corrupt  : none | %s" % " | ".join(sorted(RE.CORRUPTORS)))
    print("  example  : siuo gaussian_blur 10 8\n", flush=True)

    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        if not line:
            continue
        if line.lower() in ("quit", "exit", "q"):
            break
        parts = line.split()
        if len(parts) != 4:
            print("usage: <dataset> <corrupt> <pct> <n>   e.g.  siuo gaussian_blur 10 8")
            continue
        dataset, corrupt, pct_s, n_s = parts
        if dataset not in ("figstep", "beavertails", "siuo"):
            print("bad dataset: %s" % dataset); continue
        if corrupt != "none" and corrupt not in RE.CORRUPTORS:
            print("bad corrupt: %s" % corrupt); continue
        try:
            pct, n = float(pct_s), int(n_s)
        except ValueError:
            print("pct must be a number, n an int"); continue
        try:
            run_condition(model, processor, dataset, corrupt, pct, n, args.seed, args.outdir)
        except Exception as e:
            print("ERROR running condition: %s" % e, flush=True)

    print("bye.")


if __name__ == "__main__":
    main()
