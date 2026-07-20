#!/usr/bin/env python3
"""Build the human judge-validity audit sheet (paper P0: Cohen's kappa on the
GPT-4o SIUO judge). Stratified sample of SIUO responses balanced across the six
model configs and the four conditions (clean/glass/snow/zoom), drawn from the
local part4 response jsonls. One author then fills `human_harmful` (1=harmful
final answer, 0=safe/refusal) BLIND to the judge. The judge verdict is joined
back on Newton (see compute_judge_kappa.py) to compute kappa.
-> REU_WACV/audit/judge_audit_sheet.csv  (+ a filled copy stays out of git churn)
"""
import os, glob, json, csv, random, re

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.normpath(os.path.join(HERE, ".."))
SRC = os.path.join(REPO, "results", "part4_siuo_reasoning_vs_base")
OUTDIR = os.path.join(REPO, "REU_WACV", "audit")
os.makedirs(OUTDIR, exist_ok=True)
N_PER_CELL = 3          # 6 configs x 4 conditions x 3 = 72 items
SEED = 20260720


def final_answer(resp):
    """Best-effort extraction of the final-answer span the judge scores, so the
    human labels the same text. Falls back to the last ~1200 chars."""
    for tag in ("<CONCLUSION>", "### Safe Response", "Final Answer", "<OUTPUT>"):
        i = resp.rfind(tag)
        if i != -1:
            return resp[i:].strip()[:1500]
    return resp.strip()[-1200:]


def main():
    rng = random.Random(SEED)
    rows = []
    for f in sorted(glob.glob(os.path.join(SRC, "*", "siuo_*_responses.jsonl"))):
        recs = [json.loads(l) for l in open(f) if l.strip()]
        rng.shuffle(recs)
        for r in recs[:N_PER_CELL]:
            rows.append(dict(
                model=r["model"], condition=r["condition"], idx=r["idx"],
                category=r.get("category", ""),
                prompt=re.sub(r"\s+", " ", r["prompt"]).strip(),
                final_answer=re.sub(r"\s+", " ", final_answer(r["response"])),
                human_harmful="", notes=""))
    rng.shuffle(rows)  # de-correlate label order from model/condition
    out = os.path.join(OUTDIR, "judge_audit_sheet.csv")
    with open(out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["model", "condition", "idx", "category",
                                           "prompt", "final_answer",
                                           "human_harmful", "notes"])
        w.writeheader(); w.writerows(rows)
    print(f"wrote {len(rows)} items -> {out}")
    print("Fill `human_harmful` (1=harmful, 0=safe) blind, then run compute_judge_kappa.py on Newton.")


if __name__ == "__main__":
    main()
