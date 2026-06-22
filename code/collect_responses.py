#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
collect_responses.py — gather BeaverTails-V / FigStep / SIUO model responses for
LLaVA-CoT (reasoning) and base Llama-3.2-11B-Vision-Instruct (non-reasoning) into
ONE folder, RESPONSES ONLY.

The sources are in three formats (keyed JSON, list JSON, CSV) scattered across the
repo. This normalizes every one to the SAME clean schema and writes a nicely-named
file per (model, dataset, condition). NO scoring is carried over — `is_refusal`,
`attack_success`, `conclusion`, `predicted_letter`, etc. are all dropped. Refusal/
harm is judged LATER from these responses.

Conditions: clean + blur20 only.

Output:  model_responses/<model>_<dataset>_<cond>.json
  a JSON list, sorted by idx, each entry:
    {idx, model, dataset, condition, category, prompt, image_path, full_response}

  python code/collect_responses.py            # consolidate everything available
Re-run any time: missing sources are reported and skipped; present ones overwrite.
"""
import os
import sys
import csv
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
csv.field_size_limit(10 ** 7)            # long CoT responses

OUT_DIR = "model_responses"
EXPECTED = {"figstep": 500, "beavertails": 1180, "siuo": 167}

# (model, dataset, condition) -> ordered candidate source paths (first existing wins).
# base Llama beaver/siuo land in results_newton/... once pulled from Newton; the
# LLaVA-CoT FigStep CSVs likewise (figstep_{noise,blur}_pct, plain "base" = LLaVA-CoT).
MANIFEST = {
    # ---- base Llama (non-reasoning) ----
    ("base_llama", "figstep", "clean"):  ["results_6_18/base_vision/clean/responses_figstep.json",
                                          "results_newton/base_vision_eval/clean/responses_figstep.json"],
    ("base_llama", "figstep", "blur20"): ["results_6_18/base_vision/blur20/responses_figstep.json",
                                          "results_newton/base_vision_eval/blur20/responses_figstep.json"],
    ("base_llama", "beavertails", "clean"):  ["results_newton/base_vision_new_attacks/clean/responses_beavertails.json",
                                              "results/base_vision_new_attacks/clean/responses_beavertails.json"],
    ("base_llama", "beavertails", "blur20"): ["results_newton/base_vision_new_attacks/blur20/responses_beavertails.json",
                                              "results/base_vision_new_attacks/blur20/responses_beavertails.json"],
    ("base_llama", "siuo", "clean"):  ["results_newton/base_vision_new_attacks/clean/responses_siuo.json",
                                       "results/base_vision_new_attacks/clean/responses_siuo.json"],
    ("base_llama", "siuo", "blur20"): ["results_newton/base_vision_new_attacks/blur20/responses_siuo.json",
                                       "results/base_vision_new_attacks/blur20/responses_siuo.json"],
    # ---- LLaVA-CoT (reasoning); "base" in these filenames = LLaVA-CoT, no adapter ----
    ("llava_cot", "beavertails", "clean"):  ["results_6_18/task2_new_attacks/beavertails/responses_base_clean.csv"],
    ("llava_cot", "beavertails", "blur20"): ["results_6_18/task2_new_attacks/beavertails/responses_base_blur20.csv"],
    ("llava_cot", "siuo", "clean"):  ["results_6_18/task2_new_attacks/siuo/responses_base_clean.csv"],
    ("llava_cot", "siuo", "blur20"): ["results_6_18/task2_new_attacks/siuo/responses_base_blur20.csv"],
    ("llava_cot", "figstep", "clean"):  ["results_newton/figstep_blur_pct/responses_base_gaussian_blur_pct_p0.csv",
                                         "results/figstep_blur_pct/responses_base_gaussian_blur_pct_p0.csv"],
    ("llava_cot", "figstep", "blur20"): ["results_newton/figstep_blur_pct/responses_base_gaussian_blur_pct_p20.csv",
                                         "results/figstep_blur_pct/responses_base_gaussian_blur_pct_p20.csv"],
}

RESP_KEYS = ("full_response", "response", "model_response")  # try in order
KEEP_META = ("idx", "category", "prompt", "image_path")


def _first_existing(paths):
    for p in paths:
        if os.path.exists(p):
            return p
    return None


def _rows_from_source(path):
    """Yield raw row dicts from a CSV or JSON (keyed dict or list)."""
    if path.endswith(".csv"):
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                yield row
    else:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        rows = data.values() if isinstance(data, dict) else data
        for row in rows:
            yield row


def _norm(row, model, dataset, cond):
    resp = next((row[k] for k in RESP_KEYS if row.get(k)), None)
    if resp is None or not str(resp).strip():
        return None
    return {
        "idx":           str(row.get("idx", "")),
        "model":         model,
        "dataset":       dataset,
        "condition":     cond,
        "category":      row.get("category", "") or "",
        "prompt":        row.get("prompt", "") or "",
        "image_path":    row.get("image_path", "") or "",
        "full_response": str(resp),
    }


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Consolidating responses -> %s/  (responses only, no scoring)\n" % OUT_DIR)
    done, missing = [], []

    for (model, dataset, cond), candidates in sorted(MANIFEST.items()):
        src = _first_existing(candidates)
        label = "%s_%s_%s" % (model, dataset, cond)
        if src is None:
            missing.append(label)
            print("  [ -- ] %-34s  MISSING (looked: %s)" % (label, candidates[0]))
            continue

        entries = [e for e in (_norm(r, model, dataset, cond) for r in _rows_from_source(src)) if e]
        entries.sort(key=lambda e: e["idx"])
        exp = EXPECTED[dataset]
        flag = "" if len(entries) == exp else "  <-- WARN: expected %d" % exp
        out = os.path.join(OUT_DIR, "%s.json" % label)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(entries, f, indent=2, ensure_ascii=False)
        done.append(label)
        print("  [ ok ] %-34s  %4d responses  <- %s%s" % (label, len(entries), src, flag))

    print("\n  %d written, %d missing." % (len(done), len(missing)))
    if missing:
        print("  Missing (pull from Newton, then re-run):")
        for m in missing:
            print("    - %s" % m)


if __name__ == "__main__":
    main()
