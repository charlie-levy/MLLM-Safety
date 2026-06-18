#!/usr/bin/env python
"""
preflight.py — CPU-only pre-submission validator.

Run this BEFORE any GPU job. It catches the mistakes that waste GPU-hours
(wrong dataset counts, missing model weights, would-clobber existing results)
on the CPU, so a misconfigured job never reaches a GPU. Costs 0 GPU-hours.

Login-node safe: forces single-threaded BLAS before importing pandas/numpy so
it won't trip the node's process cap.

  python code/preflight.py --eval base_vision --blur_pct 20
  python code/preflight.py --eval msr_xstest  --blur_pct 0

Exit code 0  -> all checks passed, safe to submit.
Exit code 1  -> a check FAILED, do NOT submit.
"""
import os

# Force single-thread BEFORE numpy/pandas get imported (login-node proc cap).
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS",
           "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys
import json
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache", "huggingface", "hub")
SQA_JSON = "datasets/scienceqa_250.json"

DATASET_EXPECT = {"figstep": 500, "xstest": 250, "mmsa": 428, "sqa": 250}

# eval name -> what it needs and where it writes.
EVALS = {
    "base_vision": {
        "model_caches": ["models--meta-llama--Llama-3.2-11B-Vision-Instruct"],
        "lora_dirs":    [],
        "datasets":     ["figstep", "xstest", "mmsa", "sqa"],
        "out_tmpl":     "results/base_vision_eval/{cond}",
        "final_marker": "metrics.json",   # fresh run: refuse to clobber a finished one
        "blur_choices": [0, 20, 40, 60, 80, 100],
        "sec_per_sample": 5,              # base model, no CoT (512 tok)
    },
    "msr_xstest": {
        "model_caches": ["models--Xkev--Llama-3.2V-11B-cot"],
        "lora_dirs":    ["model_weights/llama_cot_msr"],
        "datasets":     ["xstest"],
        "out_tmpl":     "results/msr_guard_eval/{cond}",
        "final_marker": None,             # intentional re-infer into existing dir
        "blur_choices": [0, 20],
        "sec_per_sample": 20,             # CoT model, long generations (2048 tok)
    },
}


def _ok(msg):   print("  [ok]   %s" % msg)
def _warn(msg): print("  [warn] %s" % msg)
def _fail(msg): print("  [FAIL] %s" % msg)


def check_model_cache(cache_name):
    path = os.path.join(HF_CACHE, cache_name)
    if not os.path.isdir(path):
        _fail("model not cached: %s" % path)
        return False
    snap = os.path.join(path, "snapshots")
    if not os.path.isdir(snap) or not os.listdir(snap):
        _fail("model cache present but empty (no snapshots): %s" % path)
        return False
    _ok("model cached: %s" % cache_name)
    return True


def check_lora(lora_dir):
    if not os.path.isdir(lora_dir):
        _fail("LoRA weights missing: %s" % lora_dir)
        return False
    has_adapter = any(f.startswith("adapter") for f in os.listdir(lora_dir))
    if not has_adapter:
        _warn("LoRA dir has no adapter_* files: %s" % lora_dir)
    _ok("LoRA weights present: %s" % lora_dir)
    return True


def check_dataset(name):
    exp = DATASET_EXPECT[name]
    try:
        if name == "figstep":
            from dataset_loader import load_figstep
            n = len(load_figstep())
        elif name == "xstest":
            from dataset_loader import load_xstest
            n = len(load_xstest())
        elif name == "mmsa":
            from dataset_loader import load_mmsa
            n = len(load_mmsa())
        elif name == "sqa":
            if not os.path.exists(SQA_JSON):
                _fail("SQA json missing: %s" % SQA_JSON)
                return False
            with open(SQA_JSON, encoding="utf-8") as f:
                n = len(json.load(f))
        else:
            _fail("unknown dataset: %s" % name)
            return False
    except Exception as e:  # noqa: BLE001 - surface the real cause, don't swallow
        _fail("%s failed to load: %r" % (name, e))
        return False

    if n != exp:
        _fail("%s count = %d, expected %d" % (name, n, exp))
        return False
    _ok("%s: %d samples (expected %d)" % (name, n, exp))
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval", required=True, choices=sorted(EVALS.keys()))
    ap.add_argument("--blur_pct", type=int, required=True)
    ap.add_argument("--force", action="store_true",
                    help="allow submitting even if finished results already exist")
    args = ap.parse_args()

    cfg = EVALS[args.eval]
    if args.blur_pct not in cfg["blur_choices"]:
        print("ERROR: blur_pct %d not valid for %s (choices: %s)"
              % (args.blur_pct, args.eval, cfg["blur_choices"]))
        sys.exit(1)

    cond = "clean" if args.blur_pct == 0 else ("blur%d" % args.blur_pct)
    out_dir = cfg["out_tmpl"].format(cond=cond)

    print("=" * 70)
    print("  PREFLIGHT  eval=%s  blur_pct=%d  (%s)" % (args.eval, args.blur_pct, cond))
    print("  (single-threaded, CPU only — 0 GPU-hours)")
    print("=" * 70)

    ok = True

    print("\n[1] model weights")
    for c in cfg["model_caches"]:
        ok &= check_model_cache(c)
    for d in cfg["lora_dirs"]:
        ok &= check_lora(d)

    print("\n[2] datasets (counts must be exact)")
    n_samples = 0
    for ds in cfg["datasets"]:
        ok &= check_dataset(ds)
        n_samples += DATASET_EXPECT[ds]

    print("\n[3] output dir")
    marker = cfg["final_marker"]
    if marker:
        marker_path = os.path.join(out_dir, marker)
        if os.path.exists(marker_path) and not args.force:
            _fail("finished results already exist: %s "
                  "(pass --force to overwrite)" % marker_path)
            ok = False
        elif os.path.exists(marker_path):
            _warn("overwriting existing results (--force): %s" % marker_path)
        else:
            _ok("clean output dir: %s" % out_dir)
    else:
        _ok("re-run target (overwrite expected): %s" % out_dir)

    print("\n[4] cost")
    est_h = n_samples * cfg["sec_per_sample"] / 3600.0
    print("  ~%d samples  ·  ~%.1f GPU-h wall-clock est." % (n_samples, est_h))
    print("  On 'preemptable' this is BILLED 0.0 — does NOT draw down the pool.")

    print("\n" + "=" * 70)
    if ok:
        print("  PREFLIGHT PASS — safe to submit.")
        print("=" * 70)
        sys.exit(0)
    else:
        print("  PREFLIGHT FAIL — DO NOT SUBMIT. Fix the [FAIL] lines above.")
        print("=" * 70)
        sys.exit(1)


if __name__ == "__main__":
    main()
