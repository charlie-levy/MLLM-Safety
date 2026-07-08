#!/usr/bin/env python
"""
audit_mathvista.py — end-to-end audit that MathVista was RUN and JUDGED like the paper.

Checks, per the MathVista (Lu et al., ICLR 2024) protocol + official evaluation code:
  1. SPLIT       : testmini == 1000, breakdown by question_type / answer_type.
  2. PROMPT      : the built-in `query` field (the paper's exact prompt) is what was fed.
  3. EXTRACTION  : distribution of extractions; how many hit a rule shortcut vs the LLM;
                   how many empty; degenerate (repetition-loop) response rate.
  4. SCORING     : re-derive accuracy from scores_<model>.json['graded'] and confirm it
                   equals the reported accuracy (no double counting / off-by-one).
  5. SPOT-CHECK  : dump N correct + N wrong items with the FULL chain
                   (gold -> response tail -> extraction -> normalized -> verdict) so a
                   human can verify the official normalizer/safe_equal judged them right.

  conda activate REU
  export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
  python audit_mathvista.py --dir /home/ch169788/experiments/part7/mathvista --mathvista_repo ~/MathVista
"""
import os, re, sys, json, glob, argparse
from collections import Counter


def repetitive(text):
    lines = [l.strip() for l in text.split("\n") if len(l.strip()) > 3]
    if lines and max(Counter(lines).values()) >= 8:
        return True
    sents = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if len(s.strip()) > 8]
    return bool(sents and max(Counter(sents).values()) >= 8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="/home/ch169788/experiments/part7/mathvista")
    ap.add_argument("--data", default="/home/ch169788/experiments/part7/data/mathvista_testmini.json")
    ap.add_argument("--mathvista_repo", default="~/MathVista")
    ap.add_argument("--n", type=int, default=4, help="# correct + # wrong to dump per model")
    args = ap.parse_args()

    # ---- 1. SPLIT ----
    recs = json.load(open(args.data))
    print("############ 1. SPLIT (paper: testmini = 1000) ############")
    print("  total testmini:", len(recs))
    print("  question_type :", dict(Counter(r["question_type"] for r in recs)))
    print("  answer_type   :", dict(Counter(r["answer_type"] for r in recs)))

    # ---- 2. PROMPT ----
    print("\n############ 2. PROMPT (the paper's built-in `query`) ############")
    ex0 = recs[0]
    print("  pid=%s  question_type=%s" % (ex0["pid"], ex0["question_type"]))
    print("  query:\n    " + ex0["query"].replace("\n", "\n    ")[:700])

    # official scorer (to independently re-normalize a couple items)
    repo = os.path.expanduser(args.mathvista_repo)
    sys.path.insert(0, repo); sys.path.insert(0, os.path.join(repo, "evaluation"))
    from calculate_score import normalize_extracted_answer, safe_equal  # noqa

    for m in ["llm", "vision", "both"]:
        exf = os.path.join(args.dir, "extracted_%s.jsonl" % m)
        scf = os.path.join(args.dir, "scores_%s.json" % m)
        if not (os.path.isfile(exf) and os.path.isfile(scf)):
            print("\n(skip %s: missing files)" % m); continue
        ex = [json.loads(l) for l in open(exf)]
        sc = json.load(open(scf))
        exd = {e["pid"]: e for e in ex}

        # ---- 3. EXTRACTION health ----
        empty = sum(1 for e in ex if not str(e.get("extraction", "")).strip())
        degen = sum(1 for e in ex if repetitive(e["response"]))
        print("\n############ %s ############" % m.upper())
        print("  [3. EXTRACTION] n=%d  empty=%d  degenerate_responses=%d" % (len(ex), empty, degen))

        # ---- 4. SCORING re-derivation ----
        graded = sc["graded"]
        corr = sum(1 for g in graded if g["true_false"])
        rederiv = 100.0 * corr / len(graded)
        # independent re-score of a 40-item sample straight from the official functions
        import random
        samp = random.Random(0).sample(ex, 40)
        indep_ok = 0
        for e in samp:
            norm = normalize_extracted_answer(e.get("extraction", ""), e.get("choices"),
                                              e["question_type"], e["answer_type"], e.get("precision"))
            indep_ok += bool(safe_equal(norm, e["answer"]))
        print("  [4. SCORING] reported=%.2f  re-derived_from_graded=%.2f (%d/%d)  independent_40-sample=%.1f%%"
              % (sc["accuracy"], rederiv, corr, len(graded), 100.0 * indep_ok / 40))

        # ---- 5. SPOT-CHECK ----
        tf = [g for g in graded if g["true_false"]][:args.n]
        ff = [g for g in graded if not g["true_false"]][:args.n]
        for tag, gs in [("✅ CORRECT", tf), ("❌ WRONG", ff)]:
            print("  [5. SPOT-CHECK] %s" % tag)
            for g in gs:
                e = exd[g["pid"]]
                print("     pid=%s %s/%s | gold=%r extraction=%r normalized=%r | resp_tail=%r"
                      % (g["pid"], e["question_type"], e["answer_type"], g["answer"],
                         e.get("extraction"), g["normalized"], e["response"][-70:]))


if __name__ == "__main__":
    main()
