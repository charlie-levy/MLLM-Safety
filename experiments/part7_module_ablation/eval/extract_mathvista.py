#!/usr/bin/env python
"""
extract_mathvista.py — official MathVista answer extraction, with the GPT-4 extractor
swapped for your local Llama-3-8B (the offline judge you chose).

The extraction LOGIC is the paper's extract_answer() reproduced exactly (empty-guard,
multi_choice-in-choices shortcut, integer/float cast shortcuts, then the LLM extraction
using the OFFICIAL demo_prompt imported from the cloned MathVista repo). ONLY the
`model.get_response()` call is replaced with a local Llama-3-8B chat completion.

Requires a shallow clone of the official repo for the verbatim demo_prompt:
    git clone --depth 1 https://github.com/lupantech/MathVista ~/MathVista

  conda activate REU
  python extract_mathvista.py --responses /home/ch169788/experiments/part7/mathvista/responses_llm.jsonl
  python extract_mathvista.py --dir /home/ch169788/experiments/part7/mathvista
"""
import os
import re
import sys
import json
import glob
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_JUDGE = "NousResearch/Meta-Llama-3-8B-Instruct"   # same judge family as judge_sqa_utility_hf.py


def load_demo_prompt(mathvista_repo):
    """Import the OFFICIAL few-shot extraction prompt verbatim (a plain string, no deps)."""
    evald = os.path.join(os.path.expanduser(mathvista_repo), "evaluation")
    if not os.path.isdir(evald):
        raise SystemExit("MathVista repo not found at %s (git clone --depth 1 "
                         "https://github.com/lupantech/MathVista ~/MathVista)" % mathvista_repo)
    sys.path.insert(0, evald)
    from prompts.ext_ans import demo_prompt          # noqa
    return demo_prompt


def create_test_prompt(demo_prompt, query, response):
    demo_prompt = demo_prompt.strip()
    test_prompt = f"{query}\n\n{response}"
    return f"{demo_prompt}\n\n{test_prompt}\n\nExtracted answer: "


class LlamaExtractor:
    def __init__(self, model_id):
        print("[extract] loading %s ..." % model_id, flush=True)
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, torch_dtype=torch.bfloat16, device_map="auto")
        self.model.eval()
        print("[extract] ready.", flush=True)

    @torch.no_grad()
    def get_response(self, user_prompt):
        msgs = [{"role": "user", "content": user_prompt}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {k: v.to(self.model.device) for k, v in enc.items()}
        n = enc["input_ids"].shape[1]
        out = self.model.generate(**enc, max_new_tokens=64, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, n:], skip_special_tokens=True).strip()
        return text.splitlines()[0].strip() if text else ""     # short answer = first line


def extract_answer(extractor, demo_prompt, response, problem, quick_extract=False):
    """Verbatim port of MathVista extract_answer(); LLM call -> local extractor."""
    question_type = problem['question_type']
    answer_type = problem['answer_type']
    choices = problem['choices']
    query = problem['query']
    pid = problem['pid']

    if response == "":
        return ""
    if question_type == 'multi_choice' and response in (choices or []):
        return response
    if answer_type == "integer":
        try:
            return str(int(response))
        except Exception:
            pass
    if answer_type == "float":
        try:
            return str(float(response))
        except Exception:
            pass
    if quick_extract:
        try:
            result = re.search(r'The answer is "(.*)"\.', response)
            if result:
                return result.group(1)
        except Exception:
            pass
    try:
        full_prompt = create_test_prompt(demo_prompt, query, response)
        return extractor.get_response(full_prompt)
    except Exception as e:
        print("  [extract] error pid=%s: %s" % (pid, e), flush=True)
    return ""


def extract_file(extractor, demo_prompt, path):
    items = [json.loads(l) for l in open(path) if l.strip()]
    out_path = path.replace("responses_", "extracted_")
    # resume
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done[r["pid"]] = r
            except Exception:
                pass
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for it in items:
            if it["pid"] in done and done[it["pid"]].get("extraction", "") != "":
                f.write(json.dumps(done[it["pid"]], ensure_ascii=False) + "\n")
                continue
            it["extraction"] = extract_answer(extractor, demo_prompt, it["response"], it)
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
            n += 1
            if n % 50 == 0:
                print("    extracted %d ..." % n, flush=True)
    print("  %s -> %s  (%d newly extracted)" % (os.path.basename(path), os.path.basename(out_path), n))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--responses", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--mathvista_repo", default="~/MathVista")
    ap.add_argument("--model", default=DEFAULT_JUDGE)
    args = ap.parse_args()

    paths = list(args.responses)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "responses_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass --responses or --dir.")

    demo_prompt = load_demo_prompt(args.mathvista_repo)
    extractor = LlamaExtractor(args.model)
    print("Extracting %d file(s) with %s ...\n" % (len(paths), args.model))
    for p in paths:
        print("==>", p)
        extract_file(extractor, demo_prompt, p)


if __name__ == "__main__":
    main()
