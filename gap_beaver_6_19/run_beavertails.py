#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
run_beavertails.py — ONE parameterized driver to fill the BeaverTails-V ASR grid.

Reuses the EXACT proven pipeline that produced the existing cells (verified: the
scorer reproduces base/clean 76.95, base/blur20 77.80, base+TIS/clean 21.36 to the
decimal). Nothing about the method is re-implemented:
  * samples  : dataset_loader.load_new_attack("beavertails")  -> the same 1180
               prepared (image, per-sample prompt) pairs the existing cells used.
  * generate : evaluator._run_batch (same apply_chat_template -> processor ->
               model.generate -> decode path, greedy, MAX_NEW_TOKENS=1024).
  * blur     : blur_utils.blur_image(img, pct)    (same as the blur20 cell)
  * noise    : noise_utils.noisy_image(img, pct)  (the project-standard noise used
               in every FigStep/ORR/SQA noise sweep; no attack dataset had noise yet)
  * refusal  : metrics.is_refusal  (string-match; non-refusal of a harmful prompt
               = attack success).  ASR = n_non_refusals / n.

Perturbation is applied IN-MEMORY per image (originals never touched). Incremental
save every 50 images + RESUME (skips already-done idx), so a preempt loses <=50.

  python gap_beaver_6_19/run_beavertails.py --model base --condition blur40
  python gap_beaver_6_19/run_beavertails.py --model tis  --condition noise20
  python gap_beaver_6_19/run_beavertails.py --model base --condition noise20 --pilot

Output (list of 1180 entries): outputs/results_beavertails_<model>_<condition>.json
Each entry: idx, image_path, prompt, model_response, conclusion, is_refusal,
            attack_success, condition, model, category
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
os.chdir(REPO)
sys.path.insert(0, HERE)          # self-contained: proven modules live beside this

import torch                                          # noqa: E402
from tqdm import tqdm                                 # noqa: E402
from model_loader import load_model_and_processor     # noqa: E402
from dataset_loader import load_new_attack            # noqa: E402
from evaluator import Evaluator                       # noqa: E402
from metrics import is_refusal, _extract_conclusion   # noqa: E402

# condition -> (corruption_type for evaluator, percent). Every corruption_type is a
# branch already in evaluator._apply_corruption, using the project-standard *_utils
# function (so the perturbation matches the example strips). 0% = clean baseline.
COND = {
    "clean":          (None,                 0),
    "noise20":        ("gaussian_noise_pct", 20),
    "noise40":        ("gaussian_noise_pct", 40),
    "blur20":         ("gaussian_blur_pct",  20),
    "blur40":         ("gaussian_blur_pct",  40),
    # --- additional corruption families (motion blur / jpeg / pixelate) ---
    "motion_blur20":  ("motion_blur_pct",    20),
    "motion_blur40":  ("motion_blur_pct",    40),
    "jpeg20":         ("jpeg_pct",           20),
    "jpeg40":         ("jpeg_pct",           40),
    "pixelate20":     ("pixelate_pct",       20),
    "pixelate40":     ("pixelate_pct",       40),
}
SAVE_EVERY = 50


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["base", "tis"])
    ap.add_argument("--condition", required=True, choices=list(COND))
    ap.add_argument("--pilot", action="store_true", help="first 5 images only + save a sample png")
    ap.add_argument("--output_file", default=None)
    args = ap.parse_args()

    corr, pct = COND[args.condition]
    use_tis = (args.model == "tis")
    suffix = "_pilot" if args.pilot else ""
    out_path = args.output_file or os.path.join(
        HERE, "outputs", "results_beavertails_%s_%s%s.json" % (args.model, args.condition, suffix))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    print("=" * 80)
    print("  BeaverTails-V | model=%s | condition=%s%s" % (args.model, args.condition,
                                                           "  (PILOT)" if args.pilot else ""))
    print("=" * 80)

    samples = load_new_attack("beavertails")          # 1180, the SAME prepared set
    if args.pilot:
        samples = samples[:5]
    n = len(samples)
    print("[run] %d samples" % n)

    # resume / early-exit: if this cell is already complete, skip the model load
    records = json.load(open(out_path, encoding="utf-8")) if os.path.exists(out_path) else []
    done = {str(e["idx"]) for e in records}
    if all(str(s["metadata"]["idx"]) in done for s in samples):
        nonref = sum(1 for e in records if not e["is_refusal"])
        asr = 100.0 * nonref / n if n else 0.0
        print("[run] already complete (%d entries) — skipping model load." % len(records))
        print("ASR (%s/%s) = %.2f%%  (%d/%d non-refusals)  [cached]"
              % (args.model, args.condition, asr, nonref, n))
        return

    model, processor, _ = load_model_and_processor(use_tis=use_tis)
    ev = Evaluator(model, processor, corruption_type=corr, corruption_severity=pct)
    gen_kwargs = ev._get_gen_kwargs()
    print("[run] model loaded (%s)" % args.model)

    # pilot: save the (perturbed) first image so the corruption can be eyeballed
    if args.pilot and samples and samples[0]["image"] is not None:
        img0 = samples[0]["image"]
        shown = ev._apply_corruption(img0, corr, pct) if corr else img0
        png = os.path.join(HERE, "pilot_%s_sample.png" % args.condition)
        shown.convert("RGB").save(png)
        print("[run] saved sample image -> %s" % png)

    with torch.inference_mode():
        pbar = tqdm(total=n, initial=len(records), desc="%s/%s" % (args.model, args.condition), unit="img")
        for s in samples:
            idx = str(s["metadata"]["idx"])
            if idx in done:
                continue
            res = ev._run_batch([s], gen_kwargs)[0]    # corrupt in-memory + generate (proven path)
            resp = res["response"]
            ref = bool(is_refusal(resp))
            records.append({
                "idx":            idx,
                "image_path":     s["metadata"].get("image_path", ""),
                "prompt":         res["prompt"],
                "model_response": resp,
                "conclusion":     (_extract_conclusion(resp) or "").strip(),
                "is_refusal":     ref,
                "attack_success": int(not ref),        # harmful prompt not refused
                "condition":      args.condition,
                "model":          args.model,
                "category":       s["metadata"].get("category", ""),
            })
            done.add(idx)
            pbar.update(1)
            if len(records) % SAVE_EVERY == 0:
                json.dump(records, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
        pbar.close()

    json.dump(records, open(out_path, "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    assert len(records) == n, "expected %d entries, wrote %d" % (n, len(records))

    nonref = sum(1 for e in records if not e["is_refusal"])
    asr = 100.0 * nonref / n if n else 0.0
    print("\n" + "=" * 80)
    print("ASR (%s/%s) = %.2f%%   (%d/%d non-refusals)" % (args.model, args.condition, asr, nonref, n))
    print("Saved: %s  (%d entries)" % (out_path, len(records)))
    print("=" * 80)


if __name__ == "__main__":
    main()
