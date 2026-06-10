#!/usr/bin/env python
"""
judge_sqa_utility.py — LLaMA-3 (Ollama) judge for ScienceQA utility.

Reads the raw LLaVA-CoT responses saved by eval_sqa_noise_sweep.py
(raw_<tag>.jsonl) and, for each item, asks llama3 whether the model's
response arrives at the correct answer — given the question and ground truth.

This replaces brittle regex letter-extraction: LLaVA-CoT (and especially the
TIS adapter) often answers in prose / <think> blocks that regex can't parse,
but a language model judge reads the answer semantically.

Usage:
  # judge one file
  python code/judge_sqa_utility.py results_newton/sqa_noise_sweep/raw_base_tis_clean.jsonl

  # judge every raw_*.jsonl under a directory
  python code/judge_sqa_utility.py --dir results_newton/sqa_noise_sweep
  python code/judge_sqa_utility.py --dir results_newton/sqa_blur_sweep

Requires Ollama running locally with llama3:  `ollama serve` + `ollama pull llama3`
"""
import json, os, sys, glob, argparse, urllib.request

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL      = "llama3:latest"

JUDGE_TEMPLATE = """You are grading a multiple-choice science exam.

QUESTION AND ANSWER CHOICES:
{question}

CORRECT ANSWER: {label}

STUDENT'S RESPONSE:
{response}

Did the student's response select the correct answer choice ({label})?
The student may explain their reasoning at length or phrase the answer in
prose rather than just stating the letter — judge by the final answer they
commit to, not the format. If the student refused to answer, did not commit
to any choice, or chose a different option, that is INCORRECT.

Reply with exactly one word: CORRECT or INCORRECT."""


def ask_judge(question, label, response):
    prompt = JUDGE_TEMPLATE.format(question=question, label=label,
                                   response=response.strip()[:4000])
    payload = json.dumps({
        "model":   MODEL,
        "prompt":  prompt,
        "stream":  False,
        "options": {"temperature": 0},
    }).encode()
    req = urllib.request.Request(OLLAMA_URL, data=payload,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        out = json.loads(r.read())["response"].strip().upper()
    # robust parse
    if "INCORRECT" in out:
        return False
    if "CORRECT" in out:
        return True
    return None  # judge gave an unparseable verdict


def judge_file(path):
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    n_correct = n_total = n_unparsed = 0
    verdicts = []
    for i, it in enumerate(items, 1):
        v = ask_judge(it["prompt"], it["label"], it["response"])
        n_total += 1
        if v is True:
            n_correct += 1
        elif v is None:
            n_unparsed += 1
        verdicts.append({"idx": it.get("idx"), "label": it["label"], "correct": v})
        if i % 25 == 0:
            print("    judged %d/%d  (running acc %.1f%%)" % (
                i, len(items), 100.0 * n_correct / i))

    accuracy = 100.0 * n_correct / n_total if n_total else 0.0

    tag = os.path.basename(path).replace("raw_", "").replace(".jsonl", "")
    out = os.path.join(os.path.dirname(path), "judged_%s.json" % tag)
    with open(out, "w") as f:
        json.dump({
            "tag":        tag,
            "judge":      MODEL,
            "accuracy":   round(accuracy, 2),
            "correct":    n_correct,
            "total":      n_total,
            "unparsed":   n_unparsed,
            "verdicts":   verdicts,
        }, f, indent=2)
    print("  %-40s judged-accuracy = %5.1f%%  (%d/%d, unparsed=%d)  -> %s" % (
        tag, accuracy, n_correct, n_total, n_unparsed, os.path.basename(out)))
    return accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="raw_*.jsonl files to judge")
    ap.add_argument("--dir", help="judge every raw_*.jsonl in this directory")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip files whose judged_*.json already exists")
    args = ap.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "raw_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass raw_*.jsonl files or --dir <folder>.")

    if args.skip_existing:
        skipped = []
        remaining = []
        for p in paths:
            tag = os.path.basename(p).replace("raw_", "").replace(".jsonl", "")
            out = os.path.join(os.path.dirname(p), "judged_%s.json" % tag)
            if os.path.exists(out):
                skipped.append(os.path.basename(p))
            else:
                remaining.append(p)
        if skipped:
            print("Skipping %d already-judged file(s): %s\n" % (
                len(skipped), ", ".join(skipped)))
        paths = remaining

    print("Judging %d file(s) with %s ...\n" % (len(paths), MODEL))
    for p in paths:
        print("==>", p)
        judge_file(p)


if __name__ == "__main__":
    main()
