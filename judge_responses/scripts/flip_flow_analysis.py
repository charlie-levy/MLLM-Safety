#!/usr/bin/env python3
"""
flip_flow_analysis.py — gross vs net item flips under corruption (no API calls).

THE QUESTION THIS ANSWERS. The paper's safety effects are net changes of a few
points on n=167, and it says so plainly ("a 4-point delta is ~7 flipped items").
A reviewer's immediate question is what those 7 items are made of. Two worlds
produce the same +4:

  SYSTEMATIC   9 items flip safe->harmful, 2 flip back.   Corruption removes safety.
  NOISY        28 items flip safe->harmful, 21 flip back. Corruption jitters the
               verdict and the imbalance is close to chance.

Net deltas cannot tell these apart; the discordant pair counts (b, c) can, and they
are already available from the saved per-item judgements. This also yields the exact
McNemar test the paper cites for two cells, computed uniformly for every cell.

INPUT: the judged JSONs written by eval_hr_table1_siuo_jsonl_R_C.py, named
<model>_siuo_R_C_judged.json, each a list of per-item records carrying `idx` and
`judge_conclusion_harmful` / `judge_reasoning_harmful`. Entry names are
<condition>_<model>, so clean and corrupted cells pair up by model.

    python flip_flow_analysis.py --dir ~/judging/results_part13
    python flip_flow_analysis.py --dir ~/judging/results_part8 --metric reasoning

Login-node safe: stdlib only, no numpy/torch.
"""
import argparse
import glob
import json
import os
from math import comb

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]


def mcnemar_exact_two_sided(b, c):
    """Exact McNemar: P(|X - n/2| >= |b - n/2|) for X ~ Binom(n=b+c, 0.5)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return min(1.0, 2 * tail)


def load_cell(path, metric):
    key = "judge_%s_harmful" % metric
    out = {}
    for rec in json.load(open(path)):
        v = rec.get(key)
        if v is None:                      # unjudged (empty response) -> excluded
            continue
        out[str(rec.get("idx"))] = bool(v)
    return out


def split_name(stem):
    """'<condition>_<model>' -> (condition, model); longest matching prefix wins."""
    for c in sorted(CONDITIONS, key=len, reverse=True):
        if stem.startswith(c + "_"):
            return c, stem[len(c) + 1:]
    return None, stem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, help="dir of *_siuo_R_C_judged.json")
    ap.add_argument("--metric", default="conclusion", choices=["conclusion", "reasoning"])
    args = ap.parse_args()

    cells = {}
    for f in sorted(glob.glob(os.path.join(os.path.expanduser(args.dir),
                                            "*_siuo_R_C_judged.json"))):
        stem = os.path.basename(f)[:-len("_siuo_R_C_judged.json")]
        cond, model = split_name(stem)
        if cond is None:
            print("  [skip] cannot parse condition from %s" % stem)
            continue
        cells[(model, cond)] = load_cell(f, args.metric)

    models = sorted({m for m, _ in cells})
    if not models:
        raise SystemExit("No judged JSONs found in %s" % args.dir)

    print("=" * 92)
    print("GROSS vs NET FLIPS under corruption   (metric: %s)" % args.metric)
    print("b = safe->harmful (corruption hurt)   c = harmful->safe (corruption helped)")
    print("net = b - c, the change the tables report.  p = exact McNemar, two-sided.")
    print("=" * 92)
    print("%-26s %-11s %5s %5s %5s %6s %8s %9s" %
          ("model", "corruption", "n", "b", "c", "net", "net %", "McNemar p"))

    tot_b = tot_c = 0
    for m in models:
        base = cells.get((m, "clean"))
        if not base:
            print("  [skip] %s has no clean cell to pair against" % m)
            continue
        for cond in CONDITIONS[1:]:
            cur = cells.get((m, cond))
            if not cur:
                continue
            shared = sorted(set(base) & set(cur))
            b = sum(1 for i in shared if not base[i] and cur[i])
            c = sum(1 for i in shared if base[i] and not cur[i])
            n = len(shared)
            p = mcnemar_exact_two_sided(b, c)
            tot_b += b
            tot_c += c
            star = " *" if p < 0.05 else ""
            print("%-26s %-11s %5d %5d %5d %+6d %+7.1f%% %9.3f%s"
                  % (m, cond, n, b, c, b - c, (b - c) / n * 100 if n else 0, p, star))

    print("-" * 92)
    n_tot = tot_b + tot_c
    p_all = mcnemar_exact_two_sided(tot_b, tot_c)
    print("POOLED over all cells: b=%d  c=%d  net=%+d   of %d total flips, %.0f%% went "
          "safe->harmful\n  exact McNemar p=%.3g"
          % (tot_b, tot_c, tot_b - tot_c, n_tot,
             (tot_b / n_tot * 100) if n_tot else 0, p_all))
    print("""
HOW TO READ IT. A high flip volume with a near-even split (b ~ c) means corruption is
mostly jittering the judge's verdict and the net effect rides on a small imbalance --
report the net with that caveat. A modest flip volume that is strongly one-directional
(b >> c) means corruption is systematically removing safety, and the net understates
nothing. The pooled row is the sharpest single number for the paper.""")


if __name__ == "__main__":
    main()
