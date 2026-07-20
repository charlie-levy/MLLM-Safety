#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 12 (SIUO zoom-blur dose-response)
Slurm scripts. Static sbatch, one per (model x severity); submits NOTHING.

Grid: {llava_cot, qwen2_5_vl} x zoom_blur severities {1,2,4,5} = 8 jobs.
Severity 3 is NOT run: it IS the Part 4 zoom_blur cell (severity_for == 3) and is
reused at judge time; clean (sev 0) is likewise the Part 4 clean cell.
Same frozen decode as Part 4; loads from ~/.cache (HF_HUB_OFFLINE), --mem=80G.

  python gen_jobs.py
  bash /home/ch169788/experiments/part12/jobs/submit_part12.sh
"""
import os

# ── partition: flip this one line, then re-run gen_jobs.py ───────────────────────
PARTITION = "normal"        # "preemptable" (free, resume-safe) | "normal" (billed)

CONDITION = "zoom_blur"
SEVERITIES = [1, 2, 4, 5]        # 3 == Part 4 cell (reused); 0/clean == Part 4 clean
MODELS = ["llava_cot", "qwen2_5_vl"]

TIME = {"llava_cot": "04:00:00", "qwen2_5_vl": "03:00:00"}

REPO = "/home/ch169788/llava_cot_eval"
P12 = os.path.join(REPO, "experiments", "part12_dose_response")
RUN = "/home/ch169788/experiments/part12"
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P12

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
        for sev in SEVERITIES:
            name = "infer_siuo_%s_sev%d_%s" % (CONDITION, sev, m)
            body = HEADER % {"jn": "p12_%s_s%d" % (m, sev), "logs": LOGS,
                             "name": name, "time": TIME[m]}
            body += "\n" + ENV + "\n\n"
            body += ("python run_inference.py --model %s --condition %s --severity %d --output_dir %s\n"
                     % (m, CONDITION, sev, RESULTS))
            fn = "%s.sh" % name
            write(os.path.join(JOBS, fn), body)
            names.append(fn)

    submit = ("#!/bin/bash\n# Part 12: submit all %d jobs (%d models x severities %s, %s on SIUO).\n"
              "# Responses only, no judge. Independent; resume-safe per-idx.\nset -e\ncd %s\n"
              % (len(names), len(MODELS), SEVERITIES, CONDITION, JOBS))
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part12.sh"), submit)

    print("wrote %d sbatch + submit_part12.sh to %s" % (len(names), JOBS))
    print("  partition=%s%s" % (PARTITION, "  (FREE)" if PARTITION == "preemptable" else "  (BILLED)"))
    print("  models=%s  severities=%s  condition=%s" % (MODELS, SEVERITIES, CONDITION))
    print("NOTHING submitted. Canary one job first, then: bash %s/submit_part12.sh" % JOBS)


if __name__ == "__main__":
    main()
