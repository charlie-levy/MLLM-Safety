#!/usr/bin/env python
"""
run_inference.py — Part 5: evaluate the two TIS-fine-tuned LLaVA-CoT checkpoints
on SIUO under image corruptions. Generates model responses ONLY — NO judging.

Two checkpoints (LoRA adapters over the SAME base Xkev/Llama-3.2V-11B-cot):
    tis_lora     -> saves/llava_cot_tis_lora          (Think-in-Safety, CLEAN images)
    tis_corrupt  -> saves/llava_cot_tis_corrupt_lora  (TIS + RANDOM image corruptions)

Apples-to-apples with Part 4 SIUO: SAME SIUO samples, SAME corruption_lib
(zoom_blur sev3 / glass_blur sev5 — identical call to Parts 1/3/4), SAME frozen
Llama generation (code/run_eval.generate_one; greedy, 2048 tok). The ONLY thing
that differs vs Part-4 `llava_cot` is the LoRA adapter attached to the identical base.

Loading: code/model_loader.load_model_and_processor(lora_path=<ckpt>) — the repo's
existing PEFT-adapter path (base Xkev/Llama-3.2V-11B-cot + adapter), the same one
used for `llava_cot_tis`. Generation REUSES run_eval.generate_one UNCHANGED.

Output:  siuo_<condition>_<model>_responses.jsonl   (JSONL append + per-idx resume)

  python run_inference.py --model tis_lora    --condition zoom_blur
  python run_inference.py --model tis_corrupt --condition glass_blur --debug_n 2
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                     # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval, model_loader, dataset_loader
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

# importing run_eval chdir's to REPO root and wires up the tested Llama loaders/gen
import run_eval as RE                                                       # noqa: E402
from model_loader import load_model_and_processor                          # noqa: E402
from dataset_loader import load_new_attack                                 # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402

# LoRA adapters produced by experiments/tis_finetune (LLaMA-Factory output_dir roots).
CKPTS = {
    "tis_lora":    "/home/ch169788/LLaMA-Factory/saves/llava_cot_tis_lora",
    "tis_corrupt": "/home/ch169788/LLaMA-Factory/saves/llava_cot_tis_corrupt_lora",
}
CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]   # experiment uses zoom_blur + glass_blur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=sorted(CKPTS))
    ap.add_argument("--condition", required=True, choices=CONDITIONS)
    ap.add_argument("--lora_path", default="", help="override adapter dir (else CKPTS[model])")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part5/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response (writes nothing)")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    is_clean = (args.condition == "clean")
    sev = 0 if is_clean else severity_for(args.condition)
    lora_path = args.lora_path or CKPTS[args.model]
    if not os.path.isdir(lora_path):
        raise SystemExit("adapter dir not found: %s  (has the training job finished?)" % lora_path)

    out_path = os.path.join(args.output_dir, "siuo_%s_%s_responses.jsonl" % (args.condition, args.model))
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    # resume: collect already-written idx
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
    print("  Part5 INFER | siuo cond=%s(sev%d) model=%s | adapter=%s | %d samples%s  [NO JUDGE]"
          % (args.condition, sev, args.model, lora_path, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    # base Xkev/Llama-3.2V-11B-cot + this LoRA adapter (PEFT-wrapped), then reuse
    # run_eval's frozen greedy 2048-tok generation (identical to Part-4 llava_cot).
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
        if (not is_clean) and image is not None:
            image = apply_corruption(image, args.condition, severity=sev)

        resp = generate(image, prompt)

        rec = {
            "idx": idx,
            "model": args.model,
            "dataset": "siuo",
            "condition": args.condition,
            "severity": sev,
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
            print("PROMPT:", prompt[:200])
            print("RESPONSE:\n", resp)
            print("perception_failure=%s" % rec["perception_failure"])
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 10 == 0:
                print("  %d generated" % n_done, flush=True)

    if debug:
        print("\n[DEBUG] %d responses printed above — confirm they look right. Nothing written." % n_done, flush=True)
    else:
        print("\nDONE -> %s  (%d new responses this run)" % (out_path, n_done), flush=True)


if __name__ == "__main__":
    main()
