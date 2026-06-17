#!/usr/bin/env python
"""
eval_figstep_noise_sweep.py
FigStep ASR sweep over gaussian_noise severity 1-5, for BASE and BASE+TIS.
Results written to results/figstep_noise_sweep/asr_<model>_severity<N>.json
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_figstep
from evaluator import Evaluator
from metrics import compute_asr, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis",  action="store_true", help="Load TIS adapter")
parser.add_argument("--use_msr",  action="store_true", help="Load MSR-Align adapter")
parser.add_argument("--use_sage", action="store_true", help="Load SAGE adapter")
parser.add_argument("--severity", type=int, default=None, choices=[1,2,3,4,5])
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
parser.add_argument("--noise_pct", type=int, default=None,
                    help="Percentage noise 0-100 (overrides severity/noise_type)")
parser.add_argument("--blur_pct", type=int, default=None,
                    help="Percentage blur 0-100 (overrides severity/noise_type)")
parser.add_argument("--corrupt", type=str, default=None,
                    choices=["jpeg", "brightness", "pixelate", "motion_blur"],
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

# Either a severity 1-5 (legacy noise/blur) or a noise percentage must be given.
if args.noise_pct is not None:
    corr_type = "gaussian_noise_pct"
    corr_sev  = args.noise_pct
    label     = "gaussian_noise_pct_p%d" % args.noise_pct
    out_dir   = "results/figstep_noise_pct"
elif args.blur_pct is not None:
    corr_type = "gaussian_blur_pct"
    corr_sev  = args.blur_pct
    label     = "gaussian_blur_pct_p%d" % args.blur_pct
    out_dir   = "results/figstep_blur_pct"
elif args.corrupt is not None:
    if args.corrupt_pct is None:
        parser.error("--corrupt requires --corrupt_pct")
    corr_type = "%s_pct" % args.corrupt
    corr_sev  = args.corrupt_pct
    label     = "%s_pct_p%d" % (args.corrupt, args.corrupt_pct)
    out_dir   = "results/figstep_%s_pct" % args.corrupt
else:
    if args.severity is None:
        parser.error("provide --severity 1-5 or --noise_pct 0-100")
    corr_type = args.noise_type
    corr_sev  = args.severity
    label     = "%s_sev%d" % (args.noise_type, args.severity)
    out_dir   = "results/figstep_noise_sweep"

print("=" * 80)
print("  FigStep | model=%s | corruption=%s" % (model_tag, label))
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis, use_msr=args.use_msr, use_sage=args.use_sage)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor,
                      corruption_type=corr_type,
                      corruption_severity=corr_sev)
results = evaluator.run(samples)

metrics = compute_asr(results)
print("\n" + "=" * 80)
print("FigStep ASR (%s, %s): %.2f%% (%d/%d)" % (
    model_tag, label,
    metrics["asr_pct"], metrics["n_successful"], metrics["n_total"]))
print("=" * 80)

os.makedirs(out_dir, exist_ok=True)
out_file = os.path.join(out_dir, "asr_%s_%s.json" % (model_tag, label))
with open(out_file, "w") as f:
    json.dump({
        "model":       model_tag,
        "corruption":  label,
        "noise_pct":   args.noise_pct,
        "blur_pct":    args.blur_pct,
        "severity":    args.severity,
        "asr_pct":     metrics["asr_pct"],
        "n_successful": metrics["n_successful"],
        "n_refused":   metrics["n_refused"],
        "n_total":     metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)

csv_file = os.path.join(out_dir, "responses_%s_%s.csv" % (model_tag, label))
save_results_csv(results, csv_file)
print("Saved: %s" % csv_file)
