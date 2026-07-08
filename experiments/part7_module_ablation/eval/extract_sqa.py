#!/usr/bin/env python
"""
extract_sqa.py — recover each ScienceQA response's chosen option LETTER using the SAME
two-stage strategy the MathVista paper uses for answer extraction (rule-based shortcut
first, LLM few-shot extractor only for what the shortcut misses):

  STAGE 1  RULE (ScienceQA paper-exact):  re.compile(r'The answer is ([A-Z]).')
           If the response emits that line, take the letter. No LLM. Paper-exact.
  STAGE 2  LLM FALLBACK (MathVista-style): ONLY for responses that omit the letter (the
           model reasoned in prose). A Llama-3-8B few-shot extractor reads the question,
           options, and response and returns the single option letter it concluded.

Writes extracted_<model>.jsonl = raw record + 'pred_letter' + 'extract_via' (rule|llm|none).
Scoring stays ScienceQA-exact in score_sqa.py (get_pred_idx + accuracy) — this only replaces
the paper's *random* fallback with a principled LLM extraction for the non-compliant minority.

  conda activate REU
  python extract_sqa.py --dir /home/ch169788/experiments/part7/sqa        # all raw_*.jsonl
  python extract_sqa.py --raw /home/ch169788/experiments/part7/sqa/raw_llm.jsonl
"""
import os
import re
import sys
import json
import glob
import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_JUDGE = "NousResearch/Meta-Llama-3-8B-Instruct"   # same local judge as MathVista extraction
OPTIONS = ["A", "B", "C", "D", "E"]
RULE = re.compile(r'The answer is ([A-Z]).')              # ScienceQA paper-exact primary rule

# Few-shot extraction demo, in the style of MathVista's evaluation/prompts/ext_ans.py, but for
# multiple-choice letter selection. Shows the model how to map a prose conclusion to a letter.
DEMO = """Read the question, its options, and a model's response. Output ONLY the single option \
letter the response selects. If the response reasons to a conclusion without naming a letter, map \
that conclusion to the matching option and output its letter. Output just the letter, nothing else.

Question: Which property do these two objects have in common?
Options: (A) soft (B) yellow (C) sticky
Response: The lemon and the banana are both yellow, while their softness differs. They share the color yellow.
Extracted answer: B

Question: What is the capital of the country shown?
Options: (A) Paris (B) Rome (C) Madrid (D) Berlin
Response: This is the flag of Italy, and the capital of Italy is Rome.
Extracted answer: B

Question: Which sentence states a fact?
Options: (A) The painting is beautiful. (B) The painting was finished in 1889.
Response: A fact can be checked. Whether it is beautiful is an opinion; the year it was finished can be verified, so the second sentence is the fact.
Extracted answer: B"""

_SYS = ("You extract the chosen multiple-choice option letter from a model's response. "
        "Reply with ONLY a single capital letter (one of A, B, C, D, E) and nothing else.")


def first_valid_letter(text, k):
    """Return the first standalone A..(A+k-1) letter in text, else ''."""
    valid = set(OPTIONS[:k])
    m = re.search(r'\b([A-E])\b', text.upper())
    if m and m.group(1) in valid:
        return m.group(1)
    for ch in text.upper():          # fallback: first valid letter char
        if ch in valid:
            return ch
    return ""


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
    def letter(self, question, choices, response):
        k = len(choices)
        opts = " ".join("(%s) %s" % (OPTIONS[i], c) for i, c in enumerate(choices))
        user = "%s\n\nQuestion: %s\nOptions: %s\nResponse: %s\nExtracted answer: " % (
            DEMO, question, opts, response)
        msgs = [{"role": "system", "content": _SYS}, {"role": "user", "content": user}]
        enc = self.tok.apply_chat_template(msgs, add_generation_prompt=True,
                                           return_tensors="pt", return_dict=True)
        enc = {kk: v.to(self.model.device) for kk, v in enc.items()}
        n = enc["input_ids"].shape[1]
        out = self.model.generate(**enc, max_new_tokens=8, do_sample=False,
                                  pad_token_id=self.tok.eos_token_id)
        text = self.tok.decode(out[0, n:], skip_special_tokens=True)
        return first_valid_letter(text, k)


def extract_file(ex, path):
    rows = [json.loads(l) for l in open(path) if l.strip()]
    out_path = os.path.join(os.path.dirname(path),
                            "extracted_" + os.path.basename(path).replace("raw_", ""))
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done[r["idx"]] = r
            except Exception:
                pass
    n_rule = n_llm = n_none = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for r in rows:
            cached = done.get(r["idx"])
            if cached is not None and cached.get("extract_via"):
                f.write(json.dumps(cached, ensure_ascii=False) + "\n")
                via = cached["extract_via"]
                n_rule += via == "rule"; n_llm += via == "llm"; n_none += via == "none"
                continue
            k = len(r["choices"])
            mm = RULE.findall(r["response"])
            if len(mm) == 1 and mm[0] in OPTIONS[:k]:
                r["pred_letter"] = mm[0]; r["extract_via"] = "rule"; n_rule += 1
            else:
                lt = ex.letter(r["question"], r["choices"], r["response"])
                if lt:
                    r["pred_letter"] = lt; r["extract_via"] = "llm"; n_llm += 1
                else:
                    r["pred_letter"] = ""; r["extract_via"] = "none"; n_none += 1
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            if (n_rule + n_llm + n_none) % 500 == 0:
                print("    %d done (rule=%d llm=%d none=%d)" % (n_rule + n_llm + n_none, n_rule, n_llm, n_none), flush=True)
    print("  %s -> %s   rule=%d  llm=%d  none=%d"
          % (os.path.basename(path), os.path.basename(out_path), n_rule, n_llm, n_none), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", nargs="*", default=[])
    ap.add_argument("--dir")
    ap.add_argument("--model", default=DEFAULT_JUDGE)
    args = ap.parse_args()
    paths = list(args.raw)
    if args.dir:
        paths += sorted(glob.glob(os.path.join(args.dir, "raw_*.jsonl")))
    if not paths:
        sys.exit("No input. Pass --raw or --dir.")
    ex = LlamaExtractor(args.model)
    print("Extracting %d file(s) ...\n" % len(paths))
    for p in paths:
        print("==>", p, flush=True)
        extract_file(ex, p)


if __name__ == "__main__":
    main()
