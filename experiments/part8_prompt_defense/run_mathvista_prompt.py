#!/usr/bin/env python
"""
run_mathvista_prompt.py — Part 8 UTILITY companion #2: MathVista testmini accuracy for the
prompt-based defense. The MathVista analogue of run_sqa_prompt.py — "does the safety /
blur_safe system prompt cost utility?" — now on a MATH benchmark, so the prompt-defense
utility story spans two datasets (ScienceQA + MathVista), mirroring the module-ablation
utility numbers we already have on both.

Base LLaVA-CoT (NOT TIS) on MathVista-testmini (1000), SAME frozen decode as Part 4/7, with
the Part-8 safety instruction folded into the user turn (prompts.py, exactly like
run_sqa_prompt.py / run_inference.py — Llama-3.2-Vision forbids a system message with an image).

Writes responses in the EXACT schema run_mathvista.py emits, so the OFFICIAL two-step pipeline
scores it byte-for-byte the same way (extract_mathvista.py -> score_mathvista.py). The record
stores the ORIGINAL `query` (what the extractor reads); generation uses the FOLDED query (what
the model sees). Per-pid resume. Output: <output_dir>/responses_<prompt>.jsonl

  # all three prompts (money comparison none vs safety vs blur_safe), one prompt per run for arrays:
  python run_mathvista_prompt.py --prompts none,safety,blur_safe --output_dir ~/experiments/part8/mathvista
  python run_mathvista_prompt.py --prompts safety --output_dir ~/experiments/part8/mathvista
  python run_mathvista_prompt.py --prompts blur_safe --debug_n 2          # prints, writes nothing
Then (identical to the module-ablation scoring, just a different dir):
  python ../part7_module_ablation/eval/extract_mathvista.py --dir ~/experiments/part8/mathvista --mathvista_repo ~/MathVista
  python ../part7_module_ablation/eval/score_mathvista.py   --dir ~/experiments/part8/mathvista --mathvista_repo ~/MathVista
"""
import os
import sys
import json
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))                    # .../llava_cot_eval
sys.path.insert(0, HERE)                                         # prompts
sys.path.insert(0, os.path.join(REPO, "code"))                   # run_eval

import run_eval as RE                                            # noqa: E402  (chdir's to REPO)
from PIL import Image                                            # noqa: E402
from prompts import SYSTEM_PROMPTS                               # noqa: E402

MODEL = "llava_cot"                          # base LLaVA-CoT + prompt (NOT TIS), same as run_sqa_prompt.py


# identical to run_sqa_prompt.py / run_inference.py: safety instruction PREPENDED to the user turn.
def fold_prompt(system, prompt):
    return prompt if system is None else "%s\n\n%s" % (system, prompt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompts", default="none,safety,blur_safe",
                    help="comma list from prompts.py (none/safety/blur_safe/perceive)")
    ap.add_argument("--data_dir", default="/home/ch169788/experiments/part7/data",
                    help="dir with mathvista_testmini.json + images (Part 7's prepped offline set)")
    ap.add_argument("--output_dir", default="/home/ch169788/experiments/part8/mathvista")
    ap.add_argument("--debug_n", type=int, default=0,
                    help=">0: run only N items and PRINT responses (nothing written)")
    args = ap.parse_args()

    prompts = [p.strip() for p in args.prompts.split(",") if p.strip()]
    for p in prompts:
        assert p in SYSTEM_PROMPTS, "unknown prompt %r (have %s)" % (p, list(SYSTEM_PROMPTS))
    debug = args.debug_n and args.debug_n > 0
    if not debug:
        os.makedirs(args.output_dir, exist_ok=True)

    recs = json.load(open(os.path.join(args.data_dir, "mathvista_testmini.json")))
    if debug:
        recs = recs[:args.debug_n]

    print("=" * 80, flush=True)
    print("  Part8 MathVista-utility | model=%s | prompts=%s | %d items%s"
          % (MODEL, prompts, len(recs), "  [DEBUG]" if debug else ""), flush=True)
    print("=" * 80, flush=True)

    model, processor = RE.load(MODEL)
    processor.tokenizer.padding_side = "left"          # match run_mathvista.py exactly
    model.eval()

    for prm in prompts:
        system = SYSTEM_PROMPTS[prm]
        out_path = os.path.join(args.output_dir, "responses_%s.jsonl" % prm)

        written = set()
        if not debug and os.path.exists(out_path):
            with open(out_path) as f:
                for line in f:
                    try:
                        written.add(json.loads(line)["pid"])
                    except Exception:
                        pass

        print("\n----- prompt=%s -> %s (%d already done) -----"
              % (prm, os.path.basename(out_path), len(written)), flush=True)
        n_done = 0
        for rec in recs:
            pid = rec["pid"]
            if pid in written:
                continue
            image = Image.open(os.path.join(args.data_dir, rec["image_file"])).convert("RGB")
            resp = RE.generate_one(model, processor, image, fold_prompt(system, rec["query"]))

            out = {
                "pid": pid,
                "model": MODEL,
                "prompt_type": prm,
                "query": rec["query"],                 # ORIGINAL query (what the extractor reads)
                "response": resp,                      # generated from the FOLDED query
                "answer": rec["answer"],
                "question_type": rec["question_type"],
                "answer_type": rec["answer_type"],
                "precision": rec.get("precision"),
                "choices": rec.get("choices"),
                "metadata": rec.get("metadata", {}),
            }
            n_done += 1
            if debug:
                print("\n--- pid=%s prompt=%s (%s/%s) gold=%s ---"
                      % (pid, prm, rec["question_type"], rec["answer_type"], rec["answer"]))
                print("RESPONSE:\n", resp[:400])
            else:
                with open(out_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(out, ensure_ascii=False) + "\n")
                if n_done % 50 == 0:
                    print("  [%s] %d generated" % (prm, n_done), flush=True)

        print("  DONE prompt=%s (%d new)" % (prm, n_done) if not debug
              else "  [DEBUG] %d printed for prompt=%s; nothing written." % (n_done, prm), flush=True)


if __name__ == "__main__":
    main()
