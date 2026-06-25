#!/usr/bin/env python
"""
run_asr.py — Part 1 ASR driver: LLaVA-CoT+TIS on FigStep or SIUO under ONE
ImageNet-C corruption (severity=3), scored inline by the LOCKED text-only
Llama-Guard-3 judge.

Reuses the repo's tested pieces UNCHANGED:
  * model loading + frozen generation  -> code/run_eval.py  (RE.load / RE.generate_one;
    bf16, greedy, max_new_tokens=2048, one sample at a time)
  * dataset loaders                    -> code/dataset_loader.py
        figstep -> load_figstep()      (FIGSTEP_PROMPT, attack lives in the image)
        siuo    -> load_new_attack("siuo")
  * ASR judge                          -> code/judge_grid_guards.LlamaGuardTextJudge
    (Llama-Guard-3-11B-Vision, TEXT-ONLY (prompt,response) pair — the same judge
    behind every prior ASR number; never swapped)

JSONL append + per-idx resume: re-running skips already-written idx, so every job
is safely restartable.

  python run_asr.py --dataset figstep --corruption fog
  python run_asr.py --dataset siuo    --corruption glass_blur --debug_n 3
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))            # .../llava_cot_eval
sys.path.insert(0, os.path.join(REPO, "code"))
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))

# importing run_eval chdir's to REPO root and wires up the tested loaders
import run_eval as RE                                          # noqa: E402
from dataset_loader import load_figstep, load_new_attack       # noqa: E402
from judge_grid_guards import LlamaGuardTextJudge, LLAMAGUARD_MODEL  # noqa: E402
from corruption_lib import apply_corruption, PART1_CORRUPTIONS, is_perception_failure  # noqa: E402

MODEL = "llava_cot_tis"   # Part 1 is TIS-only
SEVERITY = 3


def load_samples(dataset):
    if dataset == "figstep":
        return load_figstep()
    if dataset == "siuo":
        return load_new_attack("siuo")
    raise ValueError("unknown dataset %r" % dataset)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=["figstep", "siuo"])
    ap.add_argument("--corruption", required=True, choices=PART1_CORRUPTIONS)
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part1/results")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT every response + LG output")
    args = ap.parse_args()

    debug = args.debug_n and args.debug_n > 0
    out_path = os.path.join(args.output_dir, "%s_%s_tis_asr.jsonl" % (args.dataset, args.corruption))
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

    samples = load_samples(args.dataset)
    if debug:
        samples = samples[:args.debug_n]

    print("=" * 78, flush=True)
    print("  Part1 ASR | dataset=%s corruption=%s(sev%d) model=%s | %d samples%s"
          % (args.dataset, args.corruption, SEVERITY, MODEL, len(samples),
             "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL)
    judge = LlamaGuardTextJudge(LLAMAGUARD_MODEL, batch_size=1)

    n_done = n_unsafe = 0
    for i, s in enumerate(samples):
        idx = s.get("metadata", {}).get("idx") or str(i)
        if idx in written:
            continue
        prompt = s["prompt"]
        image = s["image"]
        if image is not None:
            image = apply_corruption(image, args.corruption, severity=SEVERITY)

        resp = RE.generate_one(model, processor, image, prompt)
        _, raw, fields = judge.judge_batch([(prompt, resp)])[0]
        label = fields["verdict"]                       # "safe" / "unsafe"

        rec = {
            "idx": idx,
            "corruption": args.corruption,
            "prompt": prompt,
            "response": resp,
            "llama_guard_raw": raw,
            "llama_guard_label": label,
            "perception_failure": is_perception_failure(resp),
        }
        n_done += 1
        n_unsafe += (label == "unsafe")

        if debug:
            print("\n----- sample %d  idx=%s -----" % (i, idx))
            print("PROMPT:", prompt[:200])
            print("RESPONSE:\n", resp)
            print("LG_RAW: %r  -> label=%s  perception_failure=%s"
                  % (raw, label, rec["perception_failure"]))
        else:
            with open(out_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            if n_done % 10 == 0:
                print("  %d judged (running ASR %.1f%%)" % (n_done, 100.0 * n_unsafe / n_done), flush=True)

    if debug:
        print("\n[DEBUG] %d/%d unsafe — confirm responses + LG parsing above. Nothing written."
              % (n_unsafe, n_done), flush=True)
    else:
        print("\nDONE -> %s  (%d new, %d unsafe this run)" % (out_path, n_done, n_unsafe), flush=True)


if __name__ == "__main__":
    main()