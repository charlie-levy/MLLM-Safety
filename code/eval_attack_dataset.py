#!/usr/bin/env python
"""
eval_attack_dataset.py — ASR on the new image-based safety attack datasets
(SIUO / BeaverTails-V / SPA-VL) for LLaVA-CoT, with or without TIS.

Replaces FigStep as the attack benchmark. ASR is scored the SAME string-match
way used to get the FigStep 70.4% (base) / 13.8% (TIS) numbers: compute_asr()
auto-detects each model's answer block and counts a non-refusal as a success.

Datasets must be materialized first (login node, once):
    python code/prepare_new_attack_datasets.py

Output: results/new_attacks/<dataset>/asr_<model>_<cond>.json   (+ responses CSV)
  <cond> = clean | blur<pct>

Usage (one (dataset, model, blur) per GPU job):
  python code/eval_attack_dataset.py --dataset siuo          --blur_pct 0
  python code/eval_attack_dataset.py --dataset beavertails   --use_tis --blur_pct 20
  python code/eval_attack_dataset.py --dataset spavl         --blur_pct 40
"""
import json, sys, os, argparse
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_new_attack
from evaluator import Evaluator
from metrics import compute_asr, save_results_csv

parser = argparse.ArgumentParser()
parser.add_argument("--dataset", required=True, choices=["siuo", "beavertails", "spavl"])
parser.add_argument("--use_tis", action="store_true", help="Load TIS adapter")
parser.add_argument("--blur_pct", type=int, default=0, choices=[0, 20, 40, 60, 80, 100],
                    help="0 = clean, else %% gaussian blur (matches the FigStep sweep)")
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"
cond = "clean" if args.blur_pct == 0 else "blur%d" % args.blur_pct
corr_type = None if args.blur_pct == 0 else "gaussian_blur_pct"

print("=" * 80)
print("  %s ASR | model=%s | condition=%s" % (args.dataset.upper(), model_tag, cond))
print("=" * 80)

print("\n[1/3] Loading %s ..." % args.dataset)
samples = load_new_attack(args.dataset)
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading model (%s)..." % model_tag)
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference ...")
evaluator = Evaluator(model, processor,
                      corruption_type=corr_type,
                      corruption_severity=args.blur_pct)
results = evaluator.run(samples)

# Same string-match ASR as FigStep (auto-detect answer block, no model hint).
metrics = compute_asr(results)
print("\n" + "=" * 80)
print("%s ASR (%s, %s): %.2f%% (%d/%d complied, %d refused)" % (
    args.dataset.upper(), model_tag, cond,
    metrics["asr_pct"], metrics["n_successful"], metrics["n_total"], metrics["n_refused"]))
print("=" * 80)

out_dir = os.path.join("results", "new_attacks", args.dataset)
os.makedirs(out_dir, exist_ok=True)

out_file = os.path.join(out_dir, "asr_%s_%s.json" % (model_tag, cond))
with open(out_file, "w") as f:
    json.dump({
        "dataset":      args.dataset,
        "model":        model_tag,
        "condition":    cond,
        "blur_pct":     args.blur_pct,
        "asr_pct":      metrics["asr_pct"],
        "n_successful": metrics["n_successful"],
        "n_refused":    metrics["n_refused"],
        "n_total":      metrics["n_total"],
    }, f, indent=2)
print("Saved: %s" % out_file)

csv_file = os.path.join(out_dir, "responses_%s_%s.csv" % (model_tag, cond))
save_results_csv(results, csv_file)
print("Saved: %s" % csv_file)
