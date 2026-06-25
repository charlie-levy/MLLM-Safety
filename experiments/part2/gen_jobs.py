#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 2 Slurm scripts.

Writes 8 sbatch (base|tis x {mmsafety_tiny,spa_vl,vls_bench,holisafe}) plus
submit_part2.sh into RUN/jobs/. Same Newton-adapted header as Part 1 (normal
partition / H100 / --mem=80G / /tmp staging / resume). Submits NOTHING.

  python gen_jobs.py
"""
import os

MODELS = ["base", "tis"]
DATASETS = ["mmsafety_tiny", "spa_vl", "vls_bench", "holisafe"]

REPO = "/home/ch169788/llava_cot_eval"
P2 = os.path.join(REPO, "experiments", "part2")
RUN = "/home/ch169788/experiments/part2"
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P2

# both models loaded per job are the base Xkev (TIS adds a local LoRA) + Llama-Guard
STAGE = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
cp -an ~/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-11B-Vision /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf"""

HEADER = """#!/bin/bash
#SBATCH --job-name=%(jn)s
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=80G
#SBATCH --time=8:00:00
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
        for ds in DATASETS:
            name = "%s_%s" % (ds, m)
            body = HEADER % {"jn": "p2_%s_%s" % (m, ds), "logs": LOGS, "name": name}
            body += "\n" + ENV + "\n\n" + STAGE + "\n\n"
            body += ("python run_inference.py --model %s --dataset %s --output_dir %s\n"
                     % (m, ds, RESULTS))
            fn = "%s.sh" % name
            write(os.path.join(JOBS, fn), body)
            names.append(fn)

    submit = "#!/bin/bash\n# Part 2: submit all 8 jobs (base|tis x 4 datasets). Independent.\nset -e\ncd %s\n" % JOBS
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part2.sh"), submit)

    print("wrote %d sbatch + submit_part2.sh to %s" % (len(names), JOBS))
    print("NOTHING submitted. Review, then: bash %s/submit_part2.sh" % JOBS)


if __name__ == "__main__":
    main()