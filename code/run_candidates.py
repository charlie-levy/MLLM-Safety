#!/usr/bin/env python
"""
run_candidates.py — small batch corruption sweep. Loads the model ONCE, applies a
graded corruption ladder to every sample, SAVES the corrupted image + full
response, prints the corruption parameter. Non-interactive; built to run fast then
be pulled local for inspection.

Reuses run_eval.py's FROZEN functions (load / load_samples / generate_one /
CORRUPTORS) verbatim — generation is identical to the main pipeline; only the
graded sweep differs.

Input EITHER:
  --dataset figstep|beavertails|siuo  --n 10     (frozen loaders, random seed=0 subset)
  --items <candidates.json>                       (custom set: idx/prompt/image_path)

  python code/run_candidates.py --model llava_cot_tis --dataset figstep --n 10 \
      --sweep none:0,noise:10,noise:20,noise:30 --out results/fignoise/figstep

Output per condition:
  <out>/<tag>_<model>_<cond>.json   full responses
  <out>/img_<cond>/<idx>.png        the corrupted image actually fed to the model
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as RE                       # frozen load / load_samples / generate_one / CORRUPTORS
from PIL import Image                        # noqa: E402
from blur_utils import blur_radius           # noqa: E402


def load_from_items(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    items = list(d.values()) if isinstance(d, dict) else d
    return [(it["idx"], it["prompt"], None, it.get("image_path", "")) for it in items]


def load_from_dataset(dataset, n, seed):
    samples = RE.load_samples(dataset, n, seed)   # frozen loader, same seed=0 subset
    out = []
    for i, s in enumerate(samples):
        m = s.get("metadata", {})
        idx = m.get("idx") or str(i)
        out.append((idx, s["prompt"], s.get("image"), m.get("image_path", "")))
    return out


def param_desc(corrupt, pct, img):
    if corrupt == "gaussian_blur":
        return "blur radius=%.1f px" % blur_radius(img, pct)
    return "%s %g" % (corrupt, pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["base_llama", "llava_cot", "llava_cot_tis"])
    ap.add_argument("--dataset", default="", help="figstep|beavertails|siuo (frozen loader)")
    ap.add_argument("--items", default="", help="custom candidate JSON instead of --dataset")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sweep", default="none:0,noise:10,noise:20,noise:30",
                    help="comma list of corrupt:pct, graded mild->moderate")
    ap.add_argument("--tag", default="", help="filename prefix (defaults to dataset name)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conds = [(t.split(":")[0].strip(), float(t.split(":")[1])) for t in args.sweep.split(",")]
    if args.items:
        base, tag = load_from_items(args.items), (args.tag or "items")
    else:
        base, tag = load_from_dataset(args.dataset, args.n, args.seed), (args.tag or args.dataset)

    os.makedirs(args.out, exist_ok=True)
    print("loading %s once..." % args.model, flush=True)
    model, processor = RE.load(args.model)
    print("ready. %d samples x %d conditions (tag=%s)\n" % (len(base), len(conds), tag), flush=True)

    for corrupt, pct in conds:
        cond = "clean" if (corrupt == "none" or pct == 0) else "%s%g" % (corrupt, pct)
        fn = None if cond == "clean" else RE.CORRUPTORS[corrupt]
        imgdir = os.path.join(args.out, "img_" + cond)
        os.makedirs(imgdir, exist_ok=True)

        print("=" * 72)
        print("  CONDITION = %s   | model=%s  corrupt=%s  pct=%g  | n=%d"
              % (cond, args.model, corrupt, pct, len(base)))
        print("=" * 72, flush=True)

        recs = []
        for idx, prompt, preimg, ip in base:
            img = preimg
            if img is None and ip and os.path.exists(ip):
                img = Image.open(ip).convert("RGB")
            pdesc = ""
            if fn is not None and img is not None:
                pdesc = param_desc(corrupt, pct, img)
                img = fn(img, pct)
                img.save(os.path.join(imgdir, "%s.png" % idx))
            resp = RE.generate_one(model, processor, img, prompt)
            recs.append({
                "idx": idx, "model": args.model, "dataset": tag,
                "corrupt": corrupt, "pct": pct, "condition": cond,
                "prompt": prompt, "image_path": ip, "full_response": resp,
            })
            tag2 = "   [DEAD?]" if RE.is_dead(resp) else ""
            print("[%s] %s%s" % (idx, pdesc, tag2), flush=True)
            print("     ...%s" % resp.replace("\n", " ")[-240:], flush=True)

        out = os.path.join(args.out, "%s_%s_%s.json" % (tag, args.model, cond))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        ndead = sum(RE.is_dead(r["full_response"]) for r in recs)
        print("\nwrote %s  (%d responses, %d DEAD?)%s\n"
              % (out, len(recs), ndead, "   <-- severity too high, step DOWN" if ndead else ""),
              flush=True)


if __name__ == "__main__":
    main()