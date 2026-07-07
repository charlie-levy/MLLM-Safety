#!/usr/bin/env python
"""
run_siuo.py — Part 7 SIUO eval for the 3 module-ablation checkpoints.

BYTE-FAITHFUL to experiments/part6/run_inference.py (same SIUO samples, same
corruption_lib zoom_blur at severity 2, same PEFT adapter load, same frozen greedy
2048-tok generation via run_eval.generate_one, same output schema + per-idx resume)
so these results are directly comparable to the Part 6 SIUO cells. The ONLY changes:
the 3 checkpoints below and the output_dir.

Checkpoints (LoRA adapters over the SAME base Xkev/Llama-3.2V-11B-cot):
    llm     -> saves/llava_cot_tis_zoomblur75_llm_lora     (LoRA on LLM only)
    vision  -> saves/llava_cot_tis_zoomblur75_vision_lora  (LoRA on vision encoder only)
    both    -> saves/llava_cot_tis_zoomblur75_both_lora    (LoRA on both)

Output: siuo_zoom_blur_<model>_responses.jsonl   (JSONL append + per-idx resume)

  python run_siuo.py --model llm
  python run_siuo.py --model vision --debug_n 2
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
# .../experiments/part7_module_ablation/eval -> repo root is 3 dirs up
REPO = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

import run_eval as RE                                                       # noqa: E402
from model_loader import load_model_and_processor                          # noqa: E402
from dataset_loader import load_new_attack                                 # noqa: E402
from corruption_lib import apply_corruption, is_perception_failure         # noqa: E402

SAVES = "/home/ch169788/LLaMA-Factory/saves"
CKPTS = {
    "llm":    os.path.join(SAVES, "llava_cot_tis_zoomblur75_llm_lora"),
    "vision": os.path.join(SAVES, "llava_cot_tis_zoomblur75_vision_lora"),
    "both":   os.path.join(SAVES, "llava_cot_tis_zoomblur75_both_lora"),
}
CONDITION = "zoom_blur"
SEVERITY = 2                 # identical to Part 6 SIUO zoom_blur cell


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CKPTS))
    ap.add_argument("--lora_path", default="", help="override adapter dir (else CKPTS[model])")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part7/siuo")
    ap.add_argument("--debug_n", type=int, default=0)
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    lora_path = args.lora_path or CKPTS[args.model]
    if not os.path.isdir(lora_path):
        raise SystemExit("adapter dir not found: %s  (has the training job finished?)" % lora_path)

    out_path = os.path.join(args.output_dir, "siuo_%s_%s_responses.jsonl" % (CONDITION, args.model))
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

    samples = load_new_attack("siuo")
    if debug:
        samples = samples[:args.debug_n]

    print("=" * 80, flush=True)
    print("  Part7 SIUO | cond=%s(sev%d) model=%s | adapter=%s | %d samples%s  [NO JUDGE]"
          % (CONDITION, SEVERITY, args.model, lora_path, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    model, processor, tag = load_model_and_processor(lora_path=lora_path)
    processor.tokenizer.padding_side = "left"
    model.eval()

    def generate(img, prompt):
        return RE.generate_one(model, processor, img, prompt)

    n_done = 0
    for i, s in enumerate(samples):
        meta = s.get("metadata", {})
        idx = meta.get("idx") or str(i)
        if idx in written:
            continue
        prompt = s["prompt"]
        image = s["image"]
        if image is not None:
            image = apply_corruption(image, CONDITION, severity=SEVERITY)

        resp = generate(image, prompt)

        rec = {
            "idx": idx,
            "model": args.model,
            "dataset": "siuo",
            "condition": CONDITION,
            "severity": SEVERITY,
            "lora_path": lora_path,
            "category": meta.get("category", ""),
            "prompt": prompt,
            "response": resp,
            "image_path": meta.get("image_path", ""),
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1

        if debug:
            print("\n----- idx=%s [%s] -----" % (idx, rec["category"]))
            print("RESPONSE:\n", resp)
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 10 == 0:
                print("  %d generated" % n_done, flush=True)

    if debug:
        print("\n[DEBUG] %d responses printed; nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
