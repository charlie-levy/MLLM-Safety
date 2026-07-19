#!/usr/bin/env python3
"""Paired significance tests for SIUO clean-vs-corruption HR_C deltas.
For each model with per-sample GPT-4o judged files, pairs items by idx and runs
an exact McNemar test (binomial on discordant pairs) on judge_conclusion_harmful.
Currently local per-sample data covers LlamaV-o1 (restaged, 4 conditions);
point --dir at the full results_part4 pull to cover all six configurations."""
import json
import os
import sys
from math import comb

PULL = "/Users/charlielevy/Desktop/REU/mllm_audit_pull/newton/judging/results_part4_llamav_restaged"


def exact_mcnemar(b, c):
    """Two-sided exact McNemar: b = clean-safe->corr-harmful, c = clean-harmful->corr-safe."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(0, k + 1)) / 2**n * 2
    return min(1.0, p)


def load_verdicts(path):
    d = json.load(open(path))
    assert len(d) == 167, f"{path}: {len(d)} items"
    return {r["idx"]: bool(r["judge_conclusion_harmful"]) for r in d}


def main():
    base = sys.argv[1] if len(sys.argv) > 1 else PULL
    files = sorted(f for f in os.listdir(base) if f.endswith("_siuo_R_C_judged.json"))
    # group: {model: {condition: file}}
    grid = {}
    for f in files:
        stem = f.replace("_siuo_R_C_judged.json", "")
        model, cond = stem.rsplit("-", 1) if "-" in stem else (stem, "?")
        grid.setdefault(model, {})[cond] = os.path.join(base, f)

    for model, conds in grid.items():
        if "clean" not in conds:
            continue
        clean = load_verdicts(conds["clean"])
        hr_clean = 100 * sum(clean.values()) / len(clean)
        print(f"\n{model}  (clean HR_C {hr_clean:.1f}%, n={len(clean)})")
        print(f"  {'condition':<12}{'HR_C':>7}{'delta':>8}{'b(+)':>6}{'c(-)':>6}{'p':>9}")
        for cond, path in sorted(conds.items()):
            if cond == "clean":
                continue
            corr = load_verdicts(path)
            common = sorted(set(clean) & set(corr))
            assert len(common) == 167
            b = sum(1 for i in common if not clean[i] and corr[i])
            c = sum(1 for i in common if clean[i] and not corr[i])
            hr = 100 * sum(corr.values()) / len(corr)
            p = exact_mcnemar(b, c)
            sig = " *" if p < 0.05 else ""
            print(f"  {cond:<12}{hr:>7.1f}{hr - hr_clean:>+8.1f}{b:>6}{c:>6}{p:>9.3f}{sig}")


if __name__ == "__main__":
    main()
