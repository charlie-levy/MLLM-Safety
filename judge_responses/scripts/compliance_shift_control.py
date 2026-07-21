#!/usr/bin/env python3
"""
compliance_shift_control.py — the benign control for the corruption/safety claim.

THE OBJECTION THIS ANSWERS. Every safety number in the paper is a harmful-rate on
harmful-intent items. If corruption simply made models answer MORE and refuse LESS
across the board, harmful rate would rise for a boring reason -- a global shift in
refusal propensity -- and nothing about safety-relevant visual evidence would be
implicated. Section 4.3 rules out PERCEPTION failure (the model does not go blind);
it does not rule out a COMPLIANCE shift. This script tests that directly, on benign
items where there is no harm signal to disconnect.

DATA (all pre-existing, no compute): over-refusal rate (ORR) on XSTest-safe (250) and
a benign MM-SafetyBench-style suite (428) for four LLaVA-CoT configurations, clean vs
gaussian blur and gaussian noise at severities 1-5.
    results_newton/_archive_legacy/orr/orr_<model>.json      (clean baselines)
    results_newton/orr_{blur,noise}_sweep/orr_<model>_<corruption>_sev<N>.json

READ THE SIGN CAREFULLY. ORR is an over-refusal rate on BENIGN prompts, so ORR DOWN =
more compliant. The confound predicts corruption drives ORR down by roughly the amount
it drives harmful rate up. Run from the repo root:

    python judge_responses/scripts/compliance_shift_control.py
"""
import json
import glob
import os

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
LEGACY = os.path.join(REPO, "results_newton", "_archive_legacy", "orr")
SWEEPS = [os.path.join(REPO, "results_newton", d) for d in ("orr_blur_sweep", "orr_noise_sweep")]

# Safety-side effects to compare against, from the paper's own tables. SIUO HR_C under
# zoom blur (tab:siuo-main) and the ten-corruption ASR mean (sec:results:decoupling).
SAFETY_REF = {
    "base":     ("SIUO HR_C, zoom blur (Tab.1)", 68.3, 74.8),
    "base_tis": ("SIUO ASR, zoom blur (Sec.5.1)", 25.1, 33.5),
}


def _split(d):
    """(xstest, mmsa) over-refusal rates. THESE MUST NOT BE AVERAGED -- see main()."""
    return d.get("xstest", {}).get("orr_pct"), d.get("mmsa_combined", {}).get("orr_pct")


def load_all():
    models = {}
    for f in glob.glob(os.path.join(LEGACY, "orr_*.json")):
        d = json.load(open(f))
        models.setdefault(d["model"], {})["clean"] = _split(d)
    for sweep in SWEEPS:
        for f in glob.glob(os.path.join(sweep, "orr_*.json")):
            d = json.load(open(f))
            key = "%s_sev%d" % (d["noise_type"], d["severity"])
            models.setdefault(d["model"], {})[key] = _split(d)
    return models


def main():
    models = load_all()
    if not models:
        raise SystemExit("No ORR files found -- run from the repo root.")

    print("=" * 84)
    print("BENIGN CONTROL: does corruption shift general compliance?")
    print("=" * 84)
    print("""CRITICAL SPLIT. The two benign suites differ in WHERE the request lives, and
averaging them (the published avg_orr_pct) mixes two different mechanisms:

  XSTest-250  TEXT-ON-IMAGE. dataset_loader.load_xstest typesets the benign request
              INTO the PNG; the model must read it off the image. Blur destroys the
              rendered text, so refusal collapses for a LEGIBILITY reason -- the same
              mechanism as typographic MM-SafetyBench. This is NOT a compliance test.
  MMSA-428    NATURAL IMAGES with a text prompt. Nothing to render illegible, so a
              change here IS a change in how compliant the model is. THIS is the
              valid control for the confound.

ORR down = refuses benign requests less = more compliant.""")

    summary = {}
    for m in sorted(models):
        clean = models[m].get("clean")
        if not clean or clean[0] is None:
            continue
        print("\n%s   clean: XSTest %.1f%%   MMSA %.1f%%" % (m, clean[0], clean[1]))
        for fam in ("gaussian_blur", "gaussian_noise"):
            sevs = sorted(k for k in models[m] if k.startswith(fam))
            if not sevs:
                continue
            for j, (name, tag) in enumerate([(0, "XSTest (text-on-image)"),
                                             (1, "MMSA   (natural img)")]):
                vals = [models[m][k][name] for k in sevs]
                deltas = [v - clean[name] for v in vals]
                mean_d = sum(deltas) / len(deltas)
                summary[(m, fam, name)] = mean_d
                print("  %-15s %-22s delta %s | mean %+.1f"
                      % (fam if j == 0 else "", tag,
                         " ".join("%+5.1f" % d for d in deltas), mean_d))

    print("\n" + "=" * 84)
    print("THE DISSOCIATION (this is the control)")
    print("=" * 84)
    for m, (label, s_clean, s_corr) in SAFETY_REF.items():
        if (m, "gaussian_blur", 1) not in summary:
            continue
        sd = s_corr - s_clean
        xs = summary[(m, "gaussian_blur", 0)]
        mm = summary[(m, "gaussian_blur", 1)]
        print("\n%s  under blur" % m)
        print("  harmful, natural image  %+.1f   (%s)" % (sd, label))
        print("  benign,  natural image  %+.1f   <- MMSA: the compliance control" % mm)
        print("  benign,  text-on-image  %+.1f   <- XSTest: legibility, not compliance" % xs)
        if abs(mm) < abs(sd) / 2:
            print("  => on natural images the model does NOT become generally more compliant,")
            print("     yet harmful compliance rises. A global refusal shift cannot do this.")

    print("\n" + "=" * 84)
    print("AND THE ARGUMENT THAT NEEDS NO REFUSAL METRIC AT ALL")
    print("=" * 84)
    print("""A global compliance shift is MONOTONE: it must raise attack success on EVERY
benchmark. The paper observes the opposite under one corruption -- zoom blur LOWERS
typographic MM-SafetyBench ASR by 11.3 points while RAISING it on SIUO/VLSBench/
HoliSafe. Opposite signs cannot come from one scalar change in how often a model
answers. Note that XSTest's benign text-on-image drop above is the SAME legibility
mechanism as the MM-SafetyBench drop, now visible on benign content -- independent
support for the harm-location principle rather than a confound.""")


if __name__ == "__main__":
    main()
