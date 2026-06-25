#!/usr/bin/env python
"""
run_inference.py — Part 2 driver: LLaVA-CoT Base OR Base+TIS on CLEAN images of
one materialized dataset, scored inline by the LOCKED text-only Llama-Guard-3
judge. No corruption.

Reads the local manifest produced by prepare_datasets.py (offline — no HF calls
at job time), reuses the repo's frozen loaders + the locked judge:
  * model      -> run_eval.load("llava_cot" | "llava_cot_tis")  (bf16, greedy, 2048)
  * judge      -> judge_grid_guards.LlamaGuardTextJudge (text-only (prompt,response))

JSONL append + per-idx resume.

  python run_inference.py --model base --dataset spa_vl
  python run_inference.py --model tis  --dataset vls_bench --debug_n 3
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

import run_eval as RE                                          # noqa: E402  (chdir's to REPO)
from judge_grid_guards import LlamaGuardTextJudge, LLAMAGUARD_MODEL  # noqa: E402
from corruption_lib import is_perception_failure              # noqa: E402
from PIL import Image                                          # noqa: E402

MODEL_MAP = {"base": "llava_cot", "tis": "llava_cot_tis"}
DATA_ROOT = "/home/ch169788/experiments/part2/data"
DATASETS = ["mmsafety_tiny", "spa_vl", "vls_bench", "holisafe"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "tis"])
    ap.add_argument("--dataset", required=True, choices=DATASETS)
    ap.add_argument("--data_root", default=DATA_ROOT)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part2/results")
    ap.add_argument("--debug_n", type=int, default=0)
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    manifest = os.path.join(args.data_root, args.dataset, "manifest.jsonl")
    if not os.path.exists(manifest):
        sys.exit("missing manifest %s — run prepare_datasets.py --dataset %s first"
                 % (manifest, args.dataset))
    recs = [json.loads(l) for l in open(manifest) if l.strip()]
    if debug:
        recs = recs[:args.debug_n]

    out_path = os.path.join(args.output_dir, "%s_%s_clean.jsonl" % (args.dataset, args.model))
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)
    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("=" * 78, flush=True)
    print("  Part2 | model=%s(%s) dataset=%s | %d samples%s"
          % (args.model, MODEL_MAP[args.model], args.dataset, len(recs),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL_MAP[args.model])
    judge = LlamaGuardTextJudge(LLAMAGUARD_MODEL, batch_size=1)

    n_done = n_unsafe = 0
    for r in recs:
        idx = r["idx"]
        if idx in written:
            continue
        image = Image.open(r["image_path"]).convert("RGB")   # CLEAN, no corruption
        prompt = r["prompt"]

        resp = RE.generate_one(model, processor, image, prompt)
        _, raw, fields = judge.judge_batch([(prompt, resp)])[0]
        label = fields["verdict"]

        out = {
            "idx": idx,
            "dataset": args.dataset,
            "model": args.model,
            "category": r.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "llama_guard_raw": raw,
            "llama_guard_label": label,
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1
        n_unsafe += (label == "unsafe")

        if debug:
            print("\n----- idx=%s [%s] -----" % (idx, out["category"]))
            print("PROMPT:", prompt[:200])
            print("RESPONSE:\n", resp)
            print("LG_RAW: %r -> %s" % (raw, label))
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  %d judged (running ASR %.1f%%)" % (n_done, 100.0 * n_unsafe / n_done), flush=True)

    if debug:
        print("\n[DEBUG] %d/%d unsafe — confirm responses + LG parsing above. Nothing written."
              % (n_unsafe, n_done), flush=True)
    else:
        print("\nDONE -> %s  (%d new, %d unsafe this run)" % (out_path, n_done, n_unsafe), flush=True)


if __name__ == "__main__":
    main()