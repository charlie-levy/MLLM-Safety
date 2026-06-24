#!/usr/bin/env python
"""
verify_setup.py — end-to-end preflight for the MM-SafetyBench SD_TYPO study.
Actually performs every check (no assumptions), prints [PASS]/[FAIL] per item,
exits non-zero if anything fails. Run on a GPU node with models staged to /tmp.

NOTE: Newton has no /scratch, so paths are /home/ch169788/mmsafety_sdtypo/.
      Models load in bf16 (the frozen loaders), not fp16.

  python3 mmsafety_sdtypo/verify_setup.py
"""
import os
import io
import sys
import json
import glob
import gc
import traceback

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "code"))
sys.path.insert(0, HERE)

RUN = "/home/ch169788/mmsafety_sdtypo"
FAILS = []


def ok(name, cond, extra=""):
    print(("[PASS] " if cond else "[FAIL] ") + name + (("   " + extra) if extra else ""), flush=True)
    if not cond:
        FAILS.append(name)


def section(t):
    print("\n==== %s ====" % t, flush=True)


# ── 1. ENVIRONMENT ──
section("ENVIRONMENT")
print("python:", sys.version.split()[0])
import torch
import transformers
print("torch:", torch.__version__, "| transformers:", transformers.__version__)
try:
    import datasets
    print("datasets:", datasets.__version__)
except Exception as e:
    print("datasets import FAILED:", e)
cuda = torch.cuda.is_available()
ok("cuda available", cuda, torch.cuda.get_device_name(0) if cuda else "NO GPU VISIBLE")

import pyarrow as pa
pa.set_cpu_count(1)
pa.set_io_thread_count(1)
import pyarrow.parquet as pq
from PIL import Image
import run_eval as RE
from corruption_utils import CORRUPTIONS


def find_parquet(sub):
    return glob.glob(os.path.expanduser("~/.cache/huggingface/**/%s/SD_TYPO.parquet" % sub),
                     recursive=True)[0]


# ── 2. DATASET ──
section("DATASET")
tables = {}
for sub, exp in [("Sex", 109), ("Physical_Harm", 144)]:
    t = pq.read_table(find_parquet(sub))
    tables[sub] = t
    ok("%s rows==%d" % (sub, exp), t.num_rows == exp,
       "got %d, cols=%s" % (t.num_rows, t.column_names))
qs = tables["Sex"].column("question").to_pylist()
print("  sample questions:")
for q in qs[:3]:
    print("    -", q[:100])
img0 = Image.open(io.BytesIO(tables["Sex"].column("image")[0].as_py())).convert("RGB")
ok("image loads", img0.size[0] > 0, "size=%s" % (img0.size,))

# ── 3. CORRUPTIONS ──
section("CORRUPTIONS")
qc = os.path.join(RUN, "qc_check")
os.makedirs(qc, exist_ok=True)
for name in ["clean", "blur", "noise", "jpeg", "occlusion"]:
    try:
        out = CORRUPTIONS[name](img0)
        p = os.path.join(qc, "%s.png" % name)
        out.save(p)
        sz = os.path.getsize(p)
        ok("corrupt %s" % name, sz > 0, "out_size=%s file=%d bytes" % (out.size, sz))
    except Exception as e:
        ok("corrupt %s" % name, False, str(e))


def free(m):
    del m
    gc.collect()
    torch.cuda.empty_cache()


# ── 4. MODEL LOADING + one forward ──
section("MODEL LOADING (bf16)")
resp = {}
for tag, mn in [("base", "llava_cot"), ("tis", "llava_cot_tis")]:
    try:
        model, proc = RE.load(mn)
        r = RE.generate_one(model, proc, img0, qs[0])
        resp[tag] = r
        ok("%s load+forward" % tag, bool(r and r.strip()),
           "resp[:90]=%s" % r.replace("\n", " ")[:90])
        free(model)
    except Exception:
        ok("%s load+forward" % tag, False, "see traceback below")
        traceback.print_exc()

# ── 5. LLAMA GUARD ──
section("LLAMA GUARD (text-only, locked)")
if resp:
    try:
        from judge_grid_guards import LlamaGuardTextJudge, LLAMAGUARD_MODEL
        jg = LlamaGuardTextJudge(LLAMAGUARD_MODEL, batch_size=8)
        for tag in resp:
            _, raw, fields = jg.judge_batch([(qs[0], resp[tag])])[0]
            print("  [%s] raw=%r" % (tag, raw[:120]))
            ok("guard parses %s" % tag, fields["verdict"] in ("safe", "unsafe"),
               "label=%s" % fields["verdict"])
    except Exception:
        ok("guard", False, "see traceback below")
        traceback.print_exc()
else:
    ok("guard", False, "no model responses to judge")

# ── 6. OUTPUT PATHS ──
section("OUTPUT PATHS")
res = os.path.join(RUN, "results")
logs = os.path.join(RUN, "logs")
os.makedirs(res, exist_ok=True)
os.makedirs(logs, exist_ok=True)
dummy = os.path.join(res, "_verify_dummy.jsonl")
try:
    with open(dummy, "w") as f:
        f.write(json.dumps({"idx": -1, "test": True}) + "\n")
    back = json.loads(open(dummy).readline())
    ok("results writable + readback", back.get("idx") == -1)
    os.remove(dummy)
except Exception as e:
    ok("results writable + readback", False, str(e))
ok("logs dir exists", os.path.isdir(logs))

# ── SUMMARY ──
section("SUMMARY")
if FAILS:
    print("FAILED %d check(s): %s" % (len(FAILS), FAILS))
    sys.exit(1)
print("ALL PYTHON CHECKS PASSED")
