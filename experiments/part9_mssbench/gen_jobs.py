#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 9 (MSSBench) Slurm scripts.
One static sbatch per model (each runs the whole 96-item 'if' subset). Submits NOTHING.

Responses only, no judge (GPT-4o safe/unsafe done externally). All models read the
SAME manifest (mss_subset.jsonl) so results are directly comparable across models.
Loads from ~/.cache (HF_HUB_OFFLINE) with --mem=80G. Set PARTITION below.
  python gen_jobs.py
  bash $JOBS/submit_part9.sh
"""
import os

# ── partition: flip this one line, then re-run gen_jobs.py ───────────────────────
PARTITION = "normal"             # "normal" (billed, reliable) | "preemptable" (free, may preempt)

MODELS = ["llava_cot", "base_llama", "llamav_o1", "qwen2_5_vl", "r1_onevision", "r1_onevision_nothink"]

# 96-item subset is small; reasoning models generate longer, so keep headroom.
TIME = {"llava_cot": "02:00:00", "base_llama": "01:30:00", "llamav_o1": "04:00:00",  # 4-turn staged = slower
        "r1_onevision": "02:00:00", "r1_onevision_nothink": "01:30:00", "qwen2_5_vl": "01:30:00"}

REPO = "/home/ch169788/llava_cot_eval"
P9 = os.path.join(REPO, "experiments", "part9_mssbench")   # run_inference.py / mss_prompts.py
RUN = "/home/ch169788/experiments/part9"                   # data / manifest / outputs / jobs / logs
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")
DATA = os.path.join(RUN, "data")
SUBSET = os.path.join(RUN, "mss_subset.jsonl")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P9

_QOS = "#SBATCH --qos=preemptable\n" if PARTITION == "preemptable" else ""
HEADER = """#!/bin/bash
#SBATCH --job-name=%(jn)s
#SBATCH --partition=""" + PARTITION + """
""" + _QOS + """#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=%(time)s
#SBATCH --output=%(logs)s/%(name)s_%%j.out
#SBATCH --error=%(logs)s/%(name)s_%%j.err
"""


def write(path, txt):
    with open(path, "w") as f:
        f.write(txt)
    os.chmod(path, 0o755)


def main():
    for d in (JOBS, LOGS, RESULTS):
        os.makedirs(d, exist_ok=True)

    names = []
    for m in MODELS:
        name = "infer_mss_%s" % m
        body = HEADER % {"jn": "p9_%s" % m, "logs": LOGS, "name": name, "time": TIME[m]}
        body += "\n" + ENV + "\n\n"
        body += ("python run_inference.py --model %s --subset %s --data_root %s --output_dir %s\n"
                 % (m, SUBSET, DATA, RESULTS))
        fn = "%s.sh" % name
        write(os.path.join(JOBS, fn), body)
        names.append(fn)

    submit = ("#!/bin/bash\n# Part 9 (MSSBench): submit all %d model jobs over the shared subset.\n"
              "# Responses only, no judge. Resume-safe per-uid.\nset -e\ncd %s\n" % (len(names), JOBS))
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part9.sh"), submit)

    print("wrote %d sbatch + submit_part9.sh to %s" % (len(names), JOBS))
    print("  partition=%s%s" % (PARTITION, "  (BILLED)" if PARTITION == "normal" else "  (FREE)"))
    print("  models=%s" % MODELS)
    print("  subset=%s   data_root=%s" % (SUBSET, DATA))
    print("PREREQ: data downloaded to %s and manifest at %s (see README)." % (DATA, SUBSET))
    print("Then: bash %s/submit_part9.sh" % JOBS)


if __name__ == "__main__":
    main()
