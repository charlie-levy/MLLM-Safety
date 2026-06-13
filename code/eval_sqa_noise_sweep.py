#!/usr/bin/env python
"""
eval_sqa_noise_sweep.py
ScienceQA inference sweep over noise/blur severity 0-5, for BASE and BASE+TIS.

Severity 0 = clean (no corruption).

Saves BOTH:
  - raw responses    -> results/sqa_noise_sweep/raw_<model>_<noise>_sev<N>.jsonl
  - regex accuracy   -> results/sqa_noise_sweep/acc_<model>_<noise>_sev<N>.json

The JSONL is the source of truth: a separate LLaMA-3 judge pass
(code/judge_sqa_utility.py) reads it to compute the real utility score.
The regex accuracy is kept only as a rough sanity reference.
"""
import json, sys, os, argparse
from PIL import Image
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from evaluator import Evaluator
from metrics import compute_accuracy, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis",  action="store_true", help="Load TIS adapter")
parser.add_argument("--use_msr",  action="store_true", help="Load MSR-Align adapter")
parser.add_argument("--use_sage", action="store_true", help="Load SAGE adapter")
parser.add_argument("--severity", type=int, default=None, choices=[0,1,2,3,4,5],
                    help="0 = clean (no corruption)")
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
parser.add_argument("--noise_pct", type=int, default=None,
                    help="Percentage noise 0-100 (overrides severity/noise_type)")
parser.add_argument("--blur_pct", type=int, default=None,
                    help="Percentage blur 0-100 (overrides severity/noise_type)")
parser.add_argument("--corrupt", type=str, default=None,
                    choices=["jpeg", "brightness", "pixelate"],
                    help="Realistic corruption name (use with --corrupt_pct)")
parser.add_argument("--corrupt_pct", type=int, default=None,
                    help="Percentage 0-100 for --corrupt")
args = parser.parse_args()

if args.use_sage:
    model_tag = "base_sage"
elif args.use_msr:
    model_tag = "base_msr"
elif args.use_tis:
    model_tag = "base_tis"
else:
    model_tag = "base"

# Three modes: percentage noise, legacy severity 1-5, or clean (severity 0).
if args.noise_pct is not None:
    is_clean  = args.noise_pct == 0
    corr_type = "gaussian_noise_pct"
    corr_sev  = args.noise_pct
    tag       = "%s_clean" % model_tag if is_clean else "%s_gaussian_noise_pct_p%d" % (model_tag, args.noise_pct)
    out_dir   = "results/sqa_noise_pct"
elif args.blur_pct is not None:
    is_clean  = args.blur_pct == 0
    corr_type = "gaussian_blur_pct"
    corr_sev  = args.blur_pct
    tag       = "%s_clean" % model_tag if is_clean else "%s_gaussian_blur_pct_p%d" % (model_tag, args.blur_pct)
    out_dir   = "results/sqa_blur_pct"
elif args.corrupt is not None:
    if args.corrupt_pct is None:
        parser.error("--corrupt requires --corrupt_pct")
    is_clean  = args.corrupt_pct == 0
    corr_type = "%s_pct" % args.corrupt
    corr_sev  = args.corrupt_pct
    tag       = "%s_clean" % model_tag if is_clean else "%s_%s_pct_p%d" % (model_tag, args.corrupt, args.corrupt_pct)
    out_dir   = "results/sqa_%s_pct" % args.corrupt
else:
    if args.severity is None:
        parser.error("provide --severity 0-5 or --noise_pct 0-100")
    is_clean  = args.severity == 0
    corr_type = args.noise_type
    corr_sev  = args.severity
    tag       = "%s_clean" % model_tag if is_clean else "%s_%s_sev%d" % (model_tag, args.noise_type, args.severity)
    out_dir   = "results/sqa_noise_sweep"

print("=" * 80)
print("  ScienceQA | model=%s | corruption=%s%s" % (
    model_tag, tag, "  (CLEAN)" if is_clean else ""))
print("=" * 80)

print("\n[1/3] Loading ScienceQA...")
with open("datasets/scienceqa_250.json") as f:
    data = json.load(f)
samples = []
for key in sorted(data.keys(), key=lambda x: int(x)):
    item = data[key]
    samples.append({
        "prompt": item["prompt"],
        "image": Image.open(item["image_path"]).convert("RGB"),
        "label": item["label"],
        "metadata": item,
    })
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis, use_msr=args.use_msr, use_sage=args.use_sage)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor,
                      corruption_type=None if is_clean else corr_type,
                      corruption_severity=corr_sev)
results = evaluator.run(samples)

# ── Save raw responses (source of truth for LLM judge) ──────────────────────────
os.makedirs(out_dir, exist_ok=True)

raw_file = os.path.join(out_dir, "raw_%s.jsonl" % tag)
with open(raw_file, "w") as f:
    for r in results:
        f.write(json.dumps({
            "idx":      r["metadata"].get("idx"),
            "prompt":   r["prompt"],
            "label":    r["label"],
            "response": r["response"],
        }) + "\n")
print("Saved raw responses: %s" % raw_file)

# ── Regex accuracy (rough reference only) ───────────────────────────────────────
metrics = compute_accuracy(results)
print("\n" + "=" * 80)
print("Regex SQA Accuracy (%s): %.2f%% (%d/%d, unknown=%d)  [reference only]" % (
    tag, metrics["accuracy"], metrics["n_correct"], metrics["n_total"], metrics["n_unknown"]))
print("=" * 80)

acc_file = os.path.join(out_dir, "acc_%s.json" % tag)
with open(acc_file, "w") as f:
    json.dump({
        "model":      model_tag,
        "corruption": "clean" if is_clean else tag,
        "noise_pct":  args.noise_pct,
        "severity":   args.severity,
        "accuracy":   metrics["accuracy"],
        "correct":    metrics["n_correct"],
        "unknown":    metrics["n_unknown"],
        "total":      metrics["n_total"],
    }, f, indent=2)
print("Saved regex accuracy: %s" % acc_file)

csv_file = os.path.join(out_dir, "responses_%s.csv" % tag)
save_results_csv(results, csv_file)
print("Saved responses CSV: %s" % csv_file)
