#!/usr/bin/env python
"""
run_mathvista.py — Part 7 MathVista testmini inference for a module-ablation checkpoint.

Prompt = the dataset's built-in `query` field (the paper's exact prompt) + the image.
Generation reuses run_eval.generate_one (frozen greedy 2048-tok). Carries every field the
official extractor/scorer needs into the output. Extraction + scoring are separate steps
(extract_mathvista.py -> score_mathvista.py) using the OFFICIAL MathVista code.

Reads offline files from prep_mathvista.py. Output: responses_<model>.jsonl (per-pid resume).

  python run_mathvista.py --model llm
  python run_mathvista.py --model vision --debug_n 3
"""
import os
import sys
import json
import argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "code"))

import run_eval as RE                                          # noqa: E402
from model_loader import load_model_and_processor             # noqa: E402

SAVES = "/home/ch169788/LLaMA-Factory/saves"
CKPTS = {
    "llm":    os.path.join(SAVES, "llava_cot_tis_zoomblur75_llm_lora"),
    "vision": os.path.join(SAVES, "llava_cot_tis_zoomblur75_vision_lora"),
    "both":   os.path.join(SAVES, "llava_cot_tis_zoomblur75_both_lora"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CKPTS))
    ap.add_argument("--lora_path", default="")
    ap.add_argument("--data_dir", default="/home/ch169788/experiments/part7/data")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part7/mathvista")
    ap.add_argument("--nshards", type=int, default=1, help="data-parallel: split testmini across this many GPUs")
    ap.add_argument("--shard", type=int, default=0, help="which shard [0, nshards) this job handles")
    ap.add_argument("--debug_n", type=int, default=0)
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    lora_path = args.lora_path or CKPTS[args.model]
    if not os.path.isdir(lora_path):
        raise SystemExit("adapter dir not found: %s" % lora_path)

    recs = json.load(open(os.path.join(args.data_dir, "mathvista_testmini.json")))
    if debug:
        recs = recs[:args.debug_n]
    elif args.nshards > 1:
        # data-parallel: disjoint stride, SAME single-sample generation; merge_shards.py
        # reassembles responses_<model>.jsonl (byte-identical to a 1-GPU run).
        recs = [r for i, r in enumerate(recs) if i % args.nshards == args.shard]

    if args.nshards > 1 and not debug:
        out_path = os.path.join(args.output_dir, "shard_%s_%dof%d.jsonl" % (args.model, args.shard, args.nshards))
    else:
        out_path = os.path.join(args.output_dir, "responses_%s.jsonl" % args.model)
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)
    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["pid"])
                except Exception:
                    pass

    print("=" * 80, flush=True)
    print("  Part7 MathVista testmini | model=%s | adapter=%s | %d items%s"
          % (args.model, lora_path, len(recs), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    model, processor, tag = load_model_and_processor(lora_path=lora_path)
    processor.tokenizer.padding_side = "left"
    model.eval()

    n_done = 0
    for rec in recs:
        pid = rec["pid"]
        if pid in written:
            continue
        image = Image.open(os.path.join(args.data_dir, rec["image_file"])).convert("RGB")
        resp = RE.generate_one(model, processor, image, rec["query"])

        out = {
            "pid": pid,
            "model": args.model,
            "lora_path": lora_path,
            "query": rec["query"],
            "response": resp,
            "answer": rec["answer"],
            "question_type": rec["question_type"],
            "answer_type": rec["answer_type"],
            "precision": rec.get("precision"),
            "choices": rec.get("choices"),
            "metadata": rec.get("metadata", {}),
        }
        n_done += 1
        if debug:
            print("\n--- pid=%s (%s/%s) gold=%s ---" % (pid, rec["question_type"], rec["answer_type"], rec["answer"]))
            print("RESPONSE:\n", resp)
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(out, ensure_ascii=False) + "\n")
            if n_done % 50 == 0:
                print("  %d generated" % n_done, flush=True)

    print("\nDONE -> %s  (%d new)" % (out_path, n_done) if not debug
          else "\n[DEBUG] %d printed; nothing written." % n_done, flush=True)


if __name__ == "__main__":
    main()
