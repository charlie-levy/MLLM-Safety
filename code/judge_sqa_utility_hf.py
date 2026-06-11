#!/usr/bin/env python
"""
judge_sqa_utility_hf.py — GPU (transformers) LLaMA-3 judge for ScienceQA utility.

Newton/HPC version of judge_sqa_utility.py: instead of hitting a local Ollama
REST server, it loads the judge model directly on the GPU with transformers.
Same prompt, same output format (judged_<tag>.json), so plot_results.py and
generate_experiments_log.py work unchanged.

Default judge model is meta-llama/Meta-Llama-3-8B-Instruct — the same weights
Ollama serves as "llama3:latest", so judged accuracies stay comparable to the
Base numbers already scored locally.

Usage (on a GPU node):
  python code/judge_sqa_utility_hf.py --dir results/sqa_noise_sweep --skip-existing
  python code/judge_sqa_utility_hf.py --dir results/sqa_blur_sweep  --skip-existing
  python code/judge_sqa_utility_hf.py results/sqa_noise_sweep/raw_base_tis_clean.jsonl

Notes:
  - The judge model is gated on HuggingFace. Make sure it is cached or you are
    logged in (huggingface-cli login) — ideally pre-download on the login node.
  - Override the judge with --model <hf_id> (e.g. Qwen/Qwen2.5-7B-Instruct).
"""
import json, os, sys, glob, argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL = "NousResearch/Meta-Llama-3-8B-Instruct"  # ungated mirror, identical weights to Ollama llama3

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


class Judge:
    def __init__(self, model_id):
        print("[judge] loading %s ..." % model_id)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto",
        )
        self.model.eval()
        print("[judge] ready.")

    @torch.no_grad()
    def ask(self, question, label, response):
        prompt = JUDGE_TEMPLATE.format(
            question=question, label=label, response=response.strip()[:4000])
        msgs = [{"role": "user", "content": prompt}]
        ids = self.tok.apply_chat_template(
            msgs, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
        out = self.model.generate(
            ids, max_new_tokens=8, do_sample=False,
            pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).upper()
        if "INCORRECT" in text:
            return False
        if "CORRECT" in text:
            return True
        return None  # unparseable verdict


def judge_file(judge, path):
    with open(path) as f:
        items = [json.loads(line) for line in f if line.strip()]

    n_correct = n_total = n_unparsed = 0
    verdicts = []
    for i, it in enumerate(items, 1):
        v = judge.ask(it["prompt"], it["label"], it["response"])
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
            "tag":      tag,
            "judge":    judge.model_id,
            "accuracy": round(accuracy, 2),
            "correct":  n_correct,
            "total":    n_total,
            "unparsed": n_unparsed,
            "verdicts": verdicts,
        }, f, indent=2)
    print("  %-40s judged-accuracy = %5.1f%%  (%d/%d, unparsed=%d)  -> %s" % (
        tag, accuracy, n_correct, n_total, n_unparsed, os.path.basename(out)))
    return accuracy


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="*", help="raw_*.jsonl files to judge")
    ap.add_argument("--dir", help="judge every raw_*.jsonl in this directory")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="HF judge model id")
    ap.add_argument("--skip-existing", action="store_true",
                    help="skip files whose judged_*.json already exists")
    args = ap.parse_args()

    paths = list(args.files)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "raw_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass raw_*.jsonl files or --dir <folder>.")

    if args.skip_existing:
        skipped, remaining = [], []
        for p in paths:
            tag = os.path.basename(p).replace("raw_", "").replace(".jsonl", "")
            out = os.path.join(os.path.dirname(p), "judged_%s.json" % tag)
            (skipped if os.path.exists(out) else remaining).append(
                os.path.basename(p) if os.path.exists(out) else p)
        if skipped:
            print("Skipping %d already-judged file(s): %s\n" % (
                len(skipped), ", ".join(skipped)))
        paths = remaining

    if not paths:
        print("Nothing to judge — all files already have judged_*.json.")
        return

    judge = Judge(args.model)
    judge.model_id = args.model
    print("Judging %d file(s) with %s ...\n" % (len(paths), args.model))
    for p in paths:
        print("==>", p)
        judge_file(judge, p)


if __name__ == "__main__":
    main()
