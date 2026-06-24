#!/usr/bin/env python
"""
run_inference.py — MM-SafetyBench SD_TYPO inference for ONE (model, corruption).
Loads the model once (bf16, the repo's frozen loaders), applies the corruption to
each image, generates, writes JSONL in APPEND mode with resume logic.

  python mmsafety_sdtypo/run_inference.py --model base --corruption blur \
      --output_dir /home/ch169788/mmsafety_sdtypo/results
"""
import os
import io
import sys
import json
import glob
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))   # run_eval, model_loader
sys.path.insert(0, HERE)                                          # corruption_utils

import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import pyarrow.parquet as pq                # noqa: E402
from PIL import Image                        # noqa: E402
import run_eval as RE                        # noqa: E402  frozen load() / generate_one()
from corruption_utils import CORRUPTIONS     # noqa: E402

MODEL_MAP = {"base": "llava_cot", "tis": "llava_cot_tis"}   # LLaVA-CoT base / +TIS
SUBSETS = ["Sex", "Physical_Harm"]


def find_parquet(sub):
    return glob.glob(os.path.expanduser("~/.cache/huggingface/**/%s/SD_TYPO.parquet" % sub),
                     recursive=True)[0]


def load_image_blobs():
    out = []
    for sub in SUBSETS:                       # Sex idx 0..108, Physical_Harm 109..252
        out.extend(pq.read_table(find_parquet(sub)).column("image").to_pylist())
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "tis"])
    ap.add_argument("--corruption", required=True, choices=list(CORRUPTIONS))
    ap.add_argument("--output_dir", default="/home/ch169788/mmsafety_sdtypo/results")
    ap.add_argument("--manifest", default="/home/ch169788/mmsafety_sdtypo/samples_manifest.json")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N (0=all)")
    args = ap.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    out_file = os.path.join(args.output_dir, "%s_%s_responses.jsonl" % (args.model, args.corruption))

    # ── resume: skip idx already written ──
    written = set()
    if os.path.exists(out_file):
        with open(out_file) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    samples = json.load(open(args.manifest))
    if args.limit > 0:
        samples = samples[:args.limit]
    blobs = load_image_blobs()
    corrupt = CORRUPTIONS[args.corruption]

    print("loading %s (%s) ..." % (args.model, MODEL_MAP[args.model]), flush=True)
    model, processor = RE.load(MODEL_MAP[args.model])
    print("ready. %d samples (%d already done)\n" % (len(samples), len(written)), flush=True)

    for s in samples:
        if s["idx"] in written:
            continue
        img = Image.open(io.BytesIO(blobs[s["idx"]])).convert("RGB")
        if args.corruption != "clean":
            img = corrupt(img)
        resp = RE.generate_one(model, processor, img, s["question"])
        rec = {"idx": s["idx"], "subset": s["subset"], "question": s["question"],
               "corruption": args.corruption, "model": args.model, "response": resp,
               "llama_guard_label": None, "llama_guard_raw": None}
        with open(out_file, "a") as f:                       # write after EVERY sample
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print("[idx %d %s] done" % (s["idx"], s["subset"]), flush=True)

    print("\nDONE -> %s" % out_file, flush=True)


if __name__ == "__main__":
    main()
