#!/usr/bin/env python
import json, sys, os
sys.path.insert(0, os.path.dirname(__file__))
os.chdir(os.path.dirname(os.path.dirname(__file__)))

from model_loader import load_model_and_processor
from dataset_loader import load_figstep
from evaluator import Evaluator
from metrics import compute_asr

print("=" * 80)
print("  FigStep (500 samples, BASE, no TIS, CLEAN)")
print("=" * 80)

print("\n[1/3] Loading FigStep...")
samples = load_figstep()
print("      OK: %d samples" % len(samples))

print("\n[2/3] Loading LLaVA-CoT (BASE only)...")
result = load_model_and_processor(use_tis=False)
model, processor = result[0], result[1] if isinstance(result, tuple) else result
print("      OK: base")

print("\n[3/3] Running inference...")
evaluator = Evaluator(model, processor, corruption_type=None)
results = evaluator.run(samples)

metrics = compute_asr(results)
print("\n" + "=" * 80)
print("FigStep ASR (BASE): %.2f%% (%d/%d)" % (metrics["asr_pct"], metrics["n_successful"], metrics["n_total"]))
print("=" * 80)

os.makedirs("results/figstep_base", exist_ok=True)
with open("results/figstep_base/asr.json", "w") as f:
    json.dump({"asr": metrics["asr_pct"]}, f, indent=2)
