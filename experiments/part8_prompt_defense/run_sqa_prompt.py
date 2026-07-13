#!/usr/bin/env python
"""
run_sqa_prompt.py — Part 8 UTILITY companion: ScienceQA accuracy for the
prompt-based defense. Answers "does the safety / blur_safe system prompt cost
utility?" — the utility counterpart to the Part-8 SIUO safety numbers.

Base LLaVA-CoT (NOT TIS) on ScienceQA-250, SAME frozen decode as Part 4/8, with
the Part-8 safety instruction folded into the user turn (prompts.py, exactly like
run_inference.py). Graded inline by the SAME LLaMA-3-8B judge run_sqa.py uses
(judge_sqa_utility_hf.Judge -> CORRECT/INCORRECT).

Loads the VLM + judge ONCE, then loops the requested prompts x conditions and
prints an accuracy table. Per-idx JSONL resume (re-run to continue).

  # clean, all three prompts (the money comparison: none vs safety vs blur_safe):
  python run_sqa_prompt.py --prompts none,safety,blur_safe --conditions clean
  # sanity check first (prints 2 responses, writes nothing):
  python run_sqa_prompt.py --prompts safety --conditions clean --debug_n 2
  # widen to corruptions to match the Part-8 SIUO grid:
  python run_sqa_prompt.py --prompts none,safety,blur_safe --conditions clean,zoom_blur,snow,glass_blur

Output: <output_dir>/sqa_<condition>_llava_cot_<prompt>.jsonl  (field: correct)
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
sys.path.insert(0, HERE)                                         # prompts
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval, judge, metrics
sys.path.insert(0, os.path.join(REPO, "experiments", "common"))  # corruption_lib

import run_eval as RE                                            # noqa: E402  (chdir's to REPO)
from judge_sqa_utility_hf import Judge, DEFAULT_MODEL            # noqa: E402
from metrics import extract_answer_letter                        # noqa: E402
from PIL import Image                                            # noqa: E402
from corruption_lib import apply_corruption, severity_for        # noqa: E402
from prompts import SYSTEM_PROMPTS                               # noqa: E402

MODEL = "llava_cot"                          # base LLaVA-CoT + prompt (NOT TIS)
SQA_JSON = "datasets/scienceqa_250.json"     # relative to REPO (RE import chdir'd us there)


# identical to run_inference.py: the safety instruction is PREPENDED to the user
# prompt (Llama-3.2-Vision forbids a system message when an image is present).
def fold_prompt(system, prompt):
    return prompt if system is None else "%s\n\n%s" % (system, prompt)


def split_question_choices(prompt):
    if "Choices:" in prompt:
        q, rest = prompt.split("Choices:", 1)
        return q.replace("Question:", "").strip(), rest.replace("Answer:", "").strip()
    return prompt.strip(), ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="none,safety,blur_safe",
                    help="comma list from prompts.py (none/safety/blur_safe/perceive)")
    ap.add_argument("--conditions", default="clean",
                    help="comma list: clean,zoom_blur,snow,glass_blur")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part8/results_sqa")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N samples and PRINT responses (nothing written)")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    conditions = [c.strip() for c in args.conditions.split(",") if c.strip()]
    for p in prompts:
        assert p in SYSTEM_PROMPTS, "unknown prompt %r (have %s)" % (p, list(SYSTEM_PROMPTS))
    debug = args.debug_n and args.debug_n > 0
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    with open(SQA_JSON) as f:
        data = json.load(f)
    items = [data[k] for k in sorted(data.keys(), key=lambda x: int(x))]
    if debug:
        items = items[:args.debug_n]

    print("=" * 78, flush=True)
    print("  Part8 SQA-utility | model=%s | prompts=%s | conditions=%s | %d samples%s"
          % (MODEL, prompts, conditions, len(items), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 78, flush=True)

    model, processor = RE.load(MODEL)
    judge = Judge(DEFAULT_MODEL)
    judge.model_id = DEFAULT_MODEL

    summary = {}
    for cond in conditions:
        is_clean = (cond == "clean")
        sev = None if is_clean else severity_for(cond)
        for prm in prompts:
            system = SYSTEM_PROMPTS[prm]
            out_path = os.path.join(args.output_dir, "sqa_%s_llava_cot_%s.jsonl" % (cond, prm))

            written = set()
            n_done = n_correct = 0
            if not debug and os.path.exists(out_path):
                with open(out_path) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            written.add(r["idx"])
                            n_done += 1
                            n_correct += (r.get("correct") is True)
                        except Exception:
                            pass

            for i, it in enumerate(items):
                idx = it.get("idx", i)
                if idx in written:
                    continue
                image = Image.open(it["image_path"]).convert("RGB")
                if not is_clean:
                    image = apply_corruption(image, cond, severity=sev)

                resp = RE.generate_one(model, processor, image, fold_prompt(system, it["prompt"]))
                verdict = judge.ask(it["prompt"], it["label"], resp)   # judge sees ORIGINAL Q+choices
                pred = extract_answer_letter(resp)
                question, choices = split_question_choices(it["prompt"])

                rec = {
                    "idx": idx, "condition": cond, "prompt_type": prm,
                    "question": question, "choices": choices,
                    "ground_truth": it["label"], "response": resp,
                    "predicted_answer": pred,
                    "correct": bool(verdict) if verdict is not None else None,
                }
                n_done += 1
                n_correct += (verdict is True)

                if debug:
                    print("\n-- cond=%s prompt=%s idx=%s pred=%s correct=%s"
                          % (cond, prm, idx, pred, rec["correct"]))
                    print("RESP:", resp[:300])
                else:
                    with open(out_path, "a", encoding="utf-8") as f:
                        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    if n_done % 25 == 0:
                        print("  [%s/%s] %d graded (acc %.1f%%)"
                              % (cond, prm, n_done, 100.0 * n_correct / max(n_done, 1)), flush=True)

            acc = 100.0 * n_correct / n_done if n_done else 0.0
            summary[(cond, prm)] = (acc, n_correct, n_done)
            print("  DONE %s / %s -> acc %.1f%% (%d/%d)" % (cond, prm, acc, n_correct, n_done), flush=True)

    print("\n" + "=" * 60)
    print("  ScienceQA UTILITY — prompt-based defense (LLaMA-3 judge)")
    print("=" * 60)
    print("  %-11s %-11s %8s" % ("condition", "prompt", "acc%"))
    for (cond, prm), (acc, c, n) in summary.items():
        print("  %-11s %-11s %7.1f%%   (%d/%d)" % (cond, prm, acc, c, n))


if __name__ == "__main__":
    main()
