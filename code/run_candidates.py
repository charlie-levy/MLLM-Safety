#!/usr/bin/env python
"""
run_candidates.py — small batch sweep over a custom candidate set (e.g. the
harmful_health_content drug images). Loads the model ONCE, applies each
corruption in a graded ladder to every candidate, SAVES the corrupted image and
the full response, and prints the exact corruption parameter. Non-interactive —
built to run fast then be pulled local for inspection.

Reuses run_eval.py's FROZEN functions (load / generate_one / CORRUPTORS) verbatim,
so generation is identical to the main pipeline; only the input set + the graded
sweep differ.

  python code/run_candidates.py --model llava_cot_tis \
      --items datasets/new_attacks/hhc/hhc.json \
      --sweep none:0,gaussian_blur:5,gaussian_blur:10,pixelate:10,pixelate:20 \
      --out results/candidates/hhc

Output per condition:
  <out>/hhc_<model>_<cond>.json     full responses
  <out>/img_<cond>/<idx>.png        the corrupted image actually fed to the model
"""
import os
import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_eval as RE                       # frozen load / generate_one / CORRUPTORS / is_dead
from PIL import Image                        # noqa: E402
from blur_utils import blur_radius           # noqa: E402


def load_items(path):
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return list(d.values()) if isinstance(d, dict) else d


def param_desc(corrupt, pct, img):
    """Human-readable description of what the corruption actually does to the image."""
    if corrupt == "gaussian_blur":
        return "blur radius=%.1f px" % blur_radius(img, pct)
    if corrupt == "pixelate":
        return "pixelate %g%%" % pct
    return "%s %g" % (corrupt, pct)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["base_llama", "llava_cot", "llava_cot_tis"])
    ap.add_argument("--items", required=True, help="candidate JSON (dict or list of records)")
    ap.add_argument("--sweep", default="none:0,gaussian_blur:5,gaussian_blur:10,pixelate:10,pixelate:20",
                    help="comma list of corrupt:pct, graded mild->moderate")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conds = []
    for tok in args.sweep.split(","):
        c, p = tok.split(":")
        conds.append((c.strip(), float(p)))

    items = load_items(args.items)
    os.makedirs(args.out, exist_ok=True)

    print("loading %s once..." % args.model, flush=True)
    model, processor = RE.load(args.model)
    print("ready. %d candidates x %d conditions\n" % (len(items), len(conds)), flush=True)

    for corrupt, pct in conds:
        cond = "clean" if (corrupt == "none" or pct == 0) else "%s%g" % (corrupt, pct)
        fn = None if cond == "clean" else RE.CORRUPTORS[corrupt]
        imgdir = os.path.join(args.out, "img_" + cond)
        os.makedirs(imgdir, exist_ok=True)

        print("=" * 72)
        print("  CONDITION = %s   | model=%s  corrupt=%s  pct=%g  | n=%d"
              % (cond, args.model, corrupt, pct, len(items)))
        print("=" * 72, flush=True)

        recs = []
        for it in items:
            ip = it.get("image_path", "")
            img = Image.open(ip).convert("RGB") if (ip and os.path.exists(ip)) else None
            pdesc = ""
            if fn is not None and img is not None:
                pdesc = param_desc(corrupt, pct, img)
                img = fn(img, pct)
                img.save(os.path.join(imgdir, "%s.png" % it["idx"]))
            resp = RE.generate_one(model, processor, img, it["prompt"])
            recs.append({
                "idx": it["idx"], "model": args.model, "dataset": "hhc",
                "corrupt": corrupt, "pct": pct, "condition": cond,
                "category": it.get("category", ""), "prompt": it["prompt"],
                "image_path": ip, "full_response": resp,
            })
            tag = "   [DEAD?]" if RE.is_dead(resp) else ""
            print("[%s] %s%s" % (it["idx"], pdesc, tag), flush=True)
            print("     ...%s" % resp.replace("\n", " ")[-240:], flush=True)  # tail = the verdict

        out = os.path.join(args.out, "hhc_%s_%s.json" % (args.model, cond))
        with open(out, "w", encoding="utf-8") as f:
            json.dump(recs, f, indent=2, ensure_ascii=False)
        ndead = sum(RE.is_dead(r["full_response"]) for r in recs)
        print("\nwrote %s  (%d responses, %d DEAD?)%s\n"
              % (out, len(recs), ndead, "   <-- severity too high, step DOWN" if ndead else ""),
              flush=True)


if __name__ == "__main__":
    main()