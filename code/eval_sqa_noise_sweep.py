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
from metrics import compute_accuracy

parser = argparse.ArgumentParser()
parser.add_argument("--use_tis", action="store_true", help="Load TIS adapter")
parser.add_argument("--severity", type=int, required=True, choices=[0,1,2,3,4,5],
                    help="0 = clean (no corruption)")
parser.add_argument("--noise_type", type=str, default="gaussian_noise",
                    choices=["gaussian_noise", "gaussian_blur"])
args = parser.parse_args()

model_tag = "base_tis" if args.use_tis else "base"
is_clean  = args.severity == 0

# clean files are named with a "clean" tag and no severity suffix
if is_clean:
    tag = "%s_clean" % model_tag
else:
    tag = "%s_%s_sev%d" % (model_tag, args.noise_type, args.severity)

print("=" * 80)
print("  ScienceQA | model=%s | noise=%s | severity=%d%s" % (
    model_tag, args.noise_type, args.severity, "  (CLEAN)" if is_clean else ""))
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
model, processor, _ = load_model_and_processor(use_tis=args.use_tis)
print("      OK: %s" % model_tag)

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor,
                      corruption_type=None if is_clean else args.noise_type,
                      corruption_severity=args.severity)
results = evaluator.run(samples)

# ── Save raw responses (source of truth for LLM judge) ──────────────────────────
out_dir = "results/sqa_noise_sweep"
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
        "noise_type": "clean" if is_clean else args.noise_type,
        "severity":   args.severity,
        "accuracy":   metrics["accuracy"],
        "correct":    metrics["n_correct"],
        "unknown":    metrics["n_unknown"],
        "total":      metrics["n_total"],
    }, f, indent=2)
print("Saved regex accuracy: %s" % acc_file)
