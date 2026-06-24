#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the MM-SafetyBench SD_TYPO Slurm scripts.
Writes 10 inference + 10 judge sbatch files + submit_all.sh into RUN/jobs/, with
Newton-adapted headers (preemptable / H100 / --mem=80G / /tmp staging / resume).
Produces STATIC files; submits NOTHING.

  python mmsafety_sdtypo/gen_jobs.py
"""
import os

MODELS = ["base", "tis"]
CORRS = ["clean", "blur", "noise", "jpeg", "occlusion"]
ROOT = "/home/ch169788/llava_cot_eval"          # repo (run_inference / run_judge live here)
RUN = "/home/ch169788/mmsafety_sdtypo"          # outputs / jobs / logs (not git-tracked)
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % ROOT

HEADER = """#!/bin/bash
#SBATCH --job-name=%(jn)s
#SBATCH --partition=preemptable
#SBATCH --qos=preemptable
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=%(time)s
#SBATCH --output=%(logs)s/%(name)s_%%j.out
#SBATCH --error=%(logs)s/%(name)s_%%j.err
"""

STAGE_MODEL = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf"""

STAGE_GUARD = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-11B-Vision /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf"""


def write(path, txt):
    with open(path, "w") as f:
        f.write(txt)
    os.chmod(path, 0o755)


def main():
    os.makedirs(JOBS, exist_ok=True)
    os.makedirs(LOGS, exist_ok=True)
    os.makedirs(os.path.join(RUN, "results"), exist_ok=True)

    for m in MODELS:
        for c in CORRS:
            inf = HEADER % {"jn": "mmI_%s_%s" % (m, c), "time": "6:00:00",
                            "logs": LOGS, "name": "%s_%s_inf" % (m, c)}
            inf += "\n" + ENV + "\n\n" + STAGE_MODEL + "\n\n"
            inf += ("python mmsafety_sdtypo/run_inference.py --model %s --corruption %s "
                    "--output_dir %s/results\n" % (m, c, RUN))
            write(os.path.join(JOBS, "inference_%s_%s.sh" % (m, c)), inf)

            jdg = HEADER % {"jn": "mmJ_%s_%s" % (m, c), "time": "2:00:00",
                            "logs": LOGS, "name": "%s_%s_judge" % (m, c)}
            jdg += "\n" + ENV + "\n\n" + STAGE_GUARD + "\n\n"
            jdg += ("python mmsafety_sdtypo/run_judge.py "
                    "--input %s/results/%s_%s_responses.jsonl "
                    "--output %s/results/%s_%s_judged.jsonl\n" % (RUN, m, c, RUN, m, c))
            write(os.path.join(JOBS, "judge_%s_%s.sh" % (m, c)), jdg)

    submit = """#!/bin/bash
# Submits 10 inference jobs, each with its judge job chained via afterok.
set -e
cd %s
MODELS=(base tis)
CORRS=(clean blur noise jpeg occlusion)
for M in "${MODELS[@]}"; do
  for C in "${CORRS[@]}"; do
    INF=$(sbatch --parsable jobs/inference_${M}_${C}.sh)
    echo "inference ${M}_${C} -> $INF"
    JDG=$(sbatch --parsable --dependency=afterok:${INF} jobs/judge_${M}_${C}.sh)
    echo "judge     ${M}_${C} -> $JDG (after $INF)"
  done
done
""" % RUN
    write(os.path.join(JOBS, "submit_all.sh"), submit)

    print("wrote 10 inference + 10 judge sbatch + submit_all.sh to %s" % JOBS)
    print("NOTHING submitted. Review the scripts, then submit manually.")


if __name__ == "__main__":
    main()
