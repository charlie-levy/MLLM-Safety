#!/usr/bin/env python
"""
run_holisafe_ssu.py — HoliSafe SI+ST->U (SSU) inference under the four image
conditions of the SIUO Table-1 figure, for the 5 target VLMs. Responses ONLY —
judged afterwards by the SAME GPT-4o R/C Conclusion judge as SIUO.

IDENTICAL engine to Part 4's run_inference.py — same model dispatch (Llama pair
via run_eval 2048tok; llamav_o1 staged; Qwen pair via qwen_models 4096tok), same
greedy decoding, and the SAME corruption severities as the attached SIUO figure
(corruption_lib.severity_for: zoom_blur sev3, snow sev3, glass_blur sev5). The
ONLY change is the dataset: the HoliSafe SSU manifest instead of load_new_attack.

One model load per job -> loops all requested conditions (like the Part 11 pilot).

Dataset: experiments/part2/data/holisafe_ssu/manifest.jsonl  (n=476;
    fields {idx, dataset, category, prompt, image_path}).
Output:  holisafe_ssu_<condition>_<model>_responses.jsonl   (JSONL append + resume)

  python run_holisafe_ssu.py --model llava_cot    --conditions clean,zoom_blur,snow,glass_blur
  python run_holisafe_ssu.py --model r1_onevision --conditions glass_blur --debug_n 2
"""
import os
import sys
import json
import argparse
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                       # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "experiments", "part4"))     # qwen_models, llamav_models
sys.path.insert(0, os.path.join(REPO, "code"))                     # run_eval
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))    # corruption_lib

# importing run_eval chdir's to REPO root and wires up the tested Llama loaders
import run_eval as RE                                                       # noqa: E402
from corruption_lib import apply_corruption, severity_for, is_perception_failure  # noqa: E402

LLAMA_MODELS = ["llava_cot", "base_llama"]      # run_eval path (UNCHANGED, 2048 tok)
LLAMAV_MODELS = ["llamav_o1"]                   # Mllama reasoning fine-tune; staged loader
QWEN_MODELS = ["qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]  # qwen_models path (4096 tok)
ALL_MODELS = LLAMA_MODELS + LLAMAV_MODELS + QWEN_MODELS
CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
QWEN_MAX_NEW_TOKENS = 4096
MANIFEST = "/home/ch169788/experiments/part2/data/holisafe_ssu/manifest.jsonl"


def load_manifest(path):
    recs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                recs.append(json.loads(line))
    return recs


def gen_for_condition(generate, recs, condition, model_name, out_dir, debug):
    is_clean = (condition == "clean")
    sev = 0 if is_clean else severity_for(condition)
    out_path = os.path.join(out_dir, "holisafe_ssu_%s_%s_responses.jsonl" % (condition, model_name))

    written = set()
    if not debug and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    written.add(json.loads(line)["idx"])
                except Exception:
                    pass

    print("\n" + "-" * 80, flush=True)
    print("  condition=%s(sev%d) model=%s -> %s  (%d already done)"
          % (condition, sev, model_name, os.path.basename(out_path), len(written)), flush=True)
    print("-" * 80, flush=True)

    n_done = 0
    for m in recs:
        idx = m["idx"]
        if idx in written:
            continue
        image = Image.open(m["image_path"]).convert("RGB")
        if not is_clean:
            image = apply_corruption(image, condition, severity=sev)

        resp = generate(image, m["prompt"])

        rec = {
            "idx": idx,
            "model": model_name,
            "dataset": "holisafe_ssu",
            "condition": condition,
            "severity": sev,
            "category": m.get("category", ""),
            "prompt": m["prompt"],
            "response": resp,
            "image_path": m["image_path"],
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1

        if debug:
            print("\n----- idx=%s [%s] cond=%s -----" % (idx, rec["category"], condition))
            print("PROMPT:", m["prompt"][:200])
            print("RESPONSE:\n", resp)
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 25 == 0:
                print("  [%s] %d generated" % (condition, n_done), flush=True)

    print("  DONE condition=%s (%d new)" % (condition, n_done), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=ALL_MODELS)
    ap.add_argument("--conditions", default=",".join(CONDITIONS),
                    help="comma list from clean,zoom_blur,snow,glass_blur")
    ap.add_argument("--manifest", default=MANIFEST)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/holisafe_ssu/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N items per condition and PRINT (nothing written)")
    args = ap.parse_args()

    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for c in conditions:
        if c not in CONDITIONS:
            ap.error("unknown condition %r (choose from %s)" % (c, CONDITIONS))
    debug = args.debug_n and args.debug_n > 0

    recs = load_manifest(args.manifest)
    if debug:
        recs = recs[:args.debug_n]
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    fam = "Qwen/advisor(4096)" if args.model in QWEN_MODELS else "Llama/run_eval(2048)"
    print("=" * 80, flush=True)
    print("  HoliSafe-SSU INFER | model=%s [%s] | conditions=%s | %d items%s  [NO JUDGE]"
          % (args.model, fam, conditions, len(recs), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    # Load via the correct path; expose a uniform generate(img, prompt) closure.
    if args.model in LLAMA_MODELS:
        model, processor = RE.load(args.model)
        def generate(img, prompt):
            return RE.generate_one(model, processor, img, prompt)
    elif args.model in LLAMAV_MODELS:
        from llamav_models import load_llamav_o1, generate_llamav_staged
        model, processor = load_llamav_o1()
        def generate(img, prompt):
            return generate_llamav_staged(model, processor, img, prompt)
    else:
        from qwen_models import load_qwen, generate_one_qwen
        model, processor = load_qwen(args.model)
        no_think = args.model.endswith("_nothink")
        def generate(img, prompt):
            return generate_one_qwen(model, processor, img, prompt,
                                     max_new_tokens=QWEN_MAX_NEW_TOKENS, no_think=no_think)

    for condition in conditions:
        gen_for_condition(generate, recs, condition, args.model, args.output_dir, debug)

    if debug:
        print("\n[DEBUG] responses printed above — nothing written.", flush=True)
    else:
        print("\nALL DONE model=%s conditions=%s" % (args.model, conditions), flush=True)


if __name__ == "__main__":
    main()
