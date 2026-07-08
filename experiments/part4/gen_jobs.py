#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 4 (Experiment 4) Slurm scripts.

Writes 16 inference sbatch (4 models x 4 conditions on SIUO) plus submit_part4.sh.
Responses only, no judge. Apples-to-apples reasoning-vs-base pairs:
    llava_cot / base_llama       -> Llama family (run_eval path, 2048 tok)
    r1_onevision / qwen2_5_vl    -> Qwen family (qwen_models path, 4096 tok)
Conditions: clean, zoom_blur, snow, glass_blur. Produces STATIC files; submits NOTHING.

Loads from ~/.cache (HF_HUB_OFFLINE) with --mem=80G — no /tmp staging / HF_HOME
redirect (robust: a missing pre-stage can't crash an offline job). All four models
fit on one H100 (11B Llama, 7B Qwen).

Set PARTITION below before generating:
  "normal"      -> billed; higher-priority tier; faster start (quotas not a concern).
  "preemptable" -> free; adds --qos=preemptable; jobs may be preempted (resume-safe).

  python gen_jobs.py
"""
import os

# ── partition: flip this one line, then re-run gen_jobs.py ───────────────────────
PARTITION = "normal"             # "normal" (billed) | "preemptable" (free)

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
MODELS = ["llava_cot", "base_llama", "llamav_o1", "r1_onevision", "qwen2_5_vl"]

# per-model time limit on SIUO-167 (reasoning models generate longer)
TIME = {"llava_cot": "04:00:00", "base_llama": "03:00:00", "llamav_o1": "04:00:00",
        "r1_onevision": "04:00:00", "qwen2_5_vl": "03:00:00"}

REPO = "/home/ch169788/llava_cot_eval"
P4 = os.path.join(REPO, "experiments", "part4")         # where run_inference.py / qwen_models.py are
RUN = "/home/ch169788/experiments/part4"                # outputs / jobs / logs (not git-tracked)
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P4

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
        for c in CONDITIONS:
            name = "infer_siuo_%s_%s" % (c, m)
            body = HEADER % {"jn": "p4_%s_%s" % (m, c), "logs": LOGS, "name": name, "time": TIME[m]}
            body += "\n" + ENV + "\n\n"
            body += ("python run_inference.py --model %s --condition %s --output_dir %s\n"
                     % (m, c, RESULTS))
            fn = "%s.sh" % name
            write(os.path.join(JOBS, fn), body)
            names.append(fn)

    submit = ("#!/bin/bash\n# Part 4: submit all %d inference jobs (%d models x 4 conditions on SIUO).\n"
              "# Responses only, no judge. Independent; no chaining. Resume-safe: already-complete\n"
              "# cells are skipped per-idx, so re-running does not redo finished models.\nset -e\ncd %s\n"
              % (len(names), len(MODELS), JOBS))
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part4.sh"), submit)

    # convenience: submit ONLY the newly-added LlamaV-o1 cells (4 conditions) without
    # touching the 16 already-run llava_cot/base_llama/r1_onevision/qwen2_5_vl jobs.
    llamav = [fn for fn in names if fn.endswith("_llamav_o1.sh")]
    if llamav:
        sub_l = ("#!/bin/bash\n# Part 4: submit ONLY the LlamaV-o1 inference jobs (4 conditions on SIUO).\n"
                 "set -e\ncd %s\n" % JOBS)
        for fn in llamav:
            sub_l += "sbatch %s\n" % fn
        write(os.path.join(JOBS, "submit_part4_llamav.sh"), sub_l)

    print("wrote %d sbatch + submit_part4.sh (+ submit_part4_llamav.sh) to %s" % (len(names), JOBS))
    print("  partition=%s%s" % (PARTITION, "  (FREE)" if PARTITION == "preemptable" else "  (BILLED)"))
    print("  models=%s" % MODELS)
    print("  conditions=%s" % CONDITIONS)
    print("NOTHING submitted. Review, then: bash %s/submit_part4.sh" % JOBS)


if __name__ == "__main__":
    main()
