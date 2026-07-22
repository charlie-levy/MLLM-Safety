#!/usr/bin/env python3
"""build_e1_and_prompt_tables.py — derive the two new result tables from the judge CSVs.

Exists so that no number reaches the .tex by hand arithmetic. Reads the two summary
CSVs written by the GPT-4o R/C judge and emits (a) the deltas and (b) LaTeX table
bodies to paste. Stdlib only; login-node safe.

  E1  (part13)  Qwen3-VL-8B Instruct vs Thinking, SIUO-167. A MATCHED PAIR: same
                family and scale, differing in reasoning post-training. This is a
                cleaner test of "does reasoning training help safety?" than
                R1-Onevision native-vs-no-think, which suppresses thinking at
                inference on a single reasoning-trained checkpoint.

  P   (part8-ext)  The corruption-aware prompt defense on Qwen2.5-VL and
                R1-Onevision -- the experiment sec:defenses:prompt names as the
                highest-value missing one. The no-prompt column is transcribed from
                tab:siuo-main (see make_decoupling_multimodel.py for why that table
                cannot currently be recomputed); the safety-vs-blur_safe comparison
                is WITHIN a single judge run and does not depend on it.
"""
import csv
import os

HERE = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.normpath(os.path.join(HERE, "..", "..", "results_newton", "judging"))
CONDS = ["clean", "glass_blur", "snow", "zoom_blur"]
PRETTY = {"clean": "clean", "glass_blur": "glass blur", "snow": "snow", "zoom_blur": "zoom blur"}

# no-prompt HR_C, transcribed from tab:siuo-main in sec/4_results.tex
NOPROMPT = {
    "qwen2_5_vl":   {"clean": 70.1, "glass_blur": 73.0, "snow": 73.0, "zoom_blur": 74.2},
    "r1_onevision": {"clean": 83.8, "glass_blur": 85.6, "snow": 84.4, "zoom_blur": 86.8},
}


def read(name):
    """-> {row_label: (HR_R, HR_C)}"""
    out = {}
    with open(os.path.join(CSV_DIR, name)) as f:
        for row in csv.DictReader(f):
            out[row["model"]] = (float(row["HR_R"]), float(row["HR_C"]))
    return out


def strip_cond(label):
    """'zoom_blur_qwen3_vl_thinking' -> ('zoom_blur', 'qwen3_vl_thinking').
    Longest condition first so 'blur' cannot shadow 'zoom_blur'/'glass_blur'."""
    for c in sorted(CONDS, key=len, reverse=True):
        if label.startswith(c + "_"):
            return c, label[len(c) + 1:]
    raise SystemExit("unparseable row label: %s" % label)


def regroup(raw):
    """-> {variant: {condition: (HR_R, HR_C)}}"""
    out = {}
    for label, vals in raw.items():
        c, v = strip_cond(label)
        out.setdefault(v, {})[c] = vals
    return out


def e1():
    g = regroup(read("part13_hr_summary.csv"))
    ins, thk = g["qwen3_vl_instruct"], g["qwen3_vl_thinking"]

    print("=" * 78)
    print("E1  Qwen3-VL-8B  Instruct vs Thinking   (SIUO 167, GPT-4o judge, HR %)")
    print("=" * 78)
    print("%-11s %8s %8s | %8s %8s | %10s" %
          ("condition", "ins HR_R", "ins HR_C", "thk HR_R", "thk HR_C", "thk-ins HR_C"))
    for c in CONDS:
        print("%-11s %8.1f %8.1f | %8.1f %8.1f | %+10.1f" %
              (PRETTY[c], ins[c][0], ins[c][1], thk[c][0], thk[c][1], thk[c][1] - ins[c][1]))

    print("\ndelta from CLEAN (corruption sensitivity):")
    print("%-11s %9s %9s %9s %9s" % ("condition", "ins dR", "ins dC", "thk dR", "thk dC"))
    for c in CONDS[1:]:
        print("%-11s %+9.1f %+9.1f %+9.1f %+9.1f" %
              (PRETTY[c], ins[c][0] - ins["clean"][0], ins[c][1] - ins["clean"][1],
               thk[c][0] - thk["clean"][0], thk[c][1] - thk["clean"][1]))
    iw = max(ins[c][1] - ins["clean"][1] for c in CONDS[1:])
    tw = max(thk[c][1] - thk["clean"][1] for c in CONDS[1:])
    print("\nworst-case HR_C degradation:  Instruct %+.1f   Thinking %+.1f" % (iw, tw))

    # The R>C gap is the reasoning-leakage surface; it should exist only where there
    # IS a separate trace. Instruct emits no <think>, so its R and C score the same
    # text and differ only by judge noise -- which is itself a check on the fix to
    # extract_reasoning for the </think>-only Qwen3-VL-Thinking format.
    print("\nHR_R - HR_C  (leakage gap; large only where a separate trace exists):")
    for c in CONDS:
        print("  %-11s Instruct %+5.1f    Thinking %+5.1f"
              % (PRETTY[c], ins[c][0] - ins[c][1], thk[c][0] - thk[c][1]))

    print("\n--- LaTeX body ---")
    for c in CONDS:
        print("    %-11s & %.1f & %.1f & %.1f & %.1f & $%+.1f$ \\\\" %
              (PRETTY[c], ins[c][0], ins[c][1], thk[c][0], thk[c][1], thk[c][1] - ins[c][1]))


def prompt():
    g = regroup(read("part8_ext_hr_summary.csv"))
    print("\n" * 2 + "=" * 78)
    print("P  corruption-aware prompt on two more models  (SIUO 167, HR_C %)")
    print("=" * 78)

    worse = total = 0
    for m, mname in [("qwen2_5_vl", "Qwen2.5-VL"), ("r1_onevision", "R1-Onevision")]:
        safety, blur = g[m + "_safety"], g[m + "_blur_safe"]
        print("\n%s" % mname)
        print("  %-11s %10s %8s %10s | %s" %
              ("condition", "no prompt", "safety", "blur-safe", "blur-safe minus safety"))
        for c in CONDS:
            d = blur[c][1] - safety[c][1]
            total += 1
            worse += d > 0
            print("  %-11s %10.1f %8.1f %10.1f | %+6.1f  %s" %
                  (PRETTY[c], NOPROMPT[m][c], safety[c][1], blur[c][1], d,
                   "blur-safe WORSE" if d > 0 else "blur-safe better"))

    print("\n" + "-" * 78)
    print("blur-safe is WORSE than the generic safety prompt in %d of %d cells." % (worse, total))
    print("""On LLaVA-CoT (tab:prompting) blur-safe beat the safety prompt in all 4 cells by
5-8 points. The corruption-aware component therefore does NOT generalise: it is
specific to the one model it was developed on. Note what does survive -- BOTH
prompts still beat no prompt in every cell, so generic safety prompting works and
only the corruption-awareness fails to transfer.""")

    print("\n--- LaTeX body (no prompt / safety / blur-safe, per model) ---")
    for m, mname in [("qwen2_5_vl", "Qwen2.5-VL"), ("r1_onevision", "R1-Onevision")]:
        safety, blur = g[m + "_safety"], g[m + "_blur_safe"]
        print("    \\multicolumn{4}{l}{\\emph{%s}} \\\\" % mname)
        for c in CONDS:
            best_s = "\\textbf{%.1f}" % safety[c][1] if safety[c][1] < blur[c][1] else "%.1f" % safety[c][1]
            best_b = "\\textbf{%.1f}" % blur[c][1] if blur[c][1] < safety[c][1] else "%.1f" % blur[c][1]
            print("    \\quad %-11s & %.1f & %s & %s \\\\" % (PRETTY[c], NOPROMPT[m][c], best_s, best_b))


if __name__ == "__main__":
    e1()
    prompt()
