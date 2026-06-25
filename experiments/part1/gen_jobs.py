#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 1 Slurm scripts.

Writes 20 ASR (figstep|siuo x 10 corruptions) + 10 SQA = 30 sbatch files plus
submit_part1.sh into RUN/jobs/. Newton-adapted headers (normal partition / H100 /
--mem=80G / /tmp model staging / per-idx resume). Produces STATIC files; submits
NOTHING.

  python gen_jobs.py
"""
import os

CORRS = ["elastic_transform", "contrast", "frost", "defocus_blur", "glass_blur",
         "motion_blur", "zoom_blur", "snow", "fog", "jpeg_compression"]
DATASETS = ["figstep", "siuo"]

REPO = "/home/ch169788/llava_cot_eval"                  # git repo (scripts live here)
P1 = os.path.join(REPO, "experiments", "part1")         # where run_asr.py / run_sqa.py are
RUN = "/home/ch169788/experiments/part1"                # outputs / jobs / logs (not git-tracked)
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P1

# /tmp staging: copy the by-name HF models to node-local SSD (the LoRA adapter
# loads from /home, unaffected). ASR needs Xkev + Llama-Guard; SQA needs Xkev +
# Meta-Llama-3-8B. cp -an is a no-op if already present.
STAGE_ASR = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
cp -an ~/.cache/huggingface/hub/models--meta-llama--Llama-Guard-3-11B-Vision /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf"""

STAGE_SQA = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
cp -an ~/.cache/huggingface/hub/models--NousResearch--Meta-Llama-3-8B-Instruct /tmp/hf/hub/ 2>/dev/null
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
    # 20 ASR jobs
    for ds in DATASETS:
        for c in CORRS:
            name = "asr_%s_%s" % (ds, c)
            body = HEADER % {"jn": "p1_%s_%s" % (ds, c), "logs": LOGS, "name": name}
            body += "\n" + ENV + "\n\n" + STAGE_ASR + "\n\n"
            body += ("python run_asr.py --dataset %s --corruption %s --output_dir %s\n"
                     % (ds, c, RESULTS))
            fn = "%s.sh" % name
            write(os.path.join(JOBS, fn), body)
            names.append(fn)

    # 10 SQA jobs
    for c in CORRS:
        name = "sqa_%s" % c
        body = HEADER % {"jn": "p1_sqa_%s" % c, "logs": LOGS, "name": name}
        body += "\n" + ENV + "\n\n" + STAGE_SQA + "\n\n"
        body += ("python run_sqa.py --corruption %s --output_dir %s\n" % (c, RESULTS))
        fn = "%s.sh" % name
        write(os.path.join(JOBS, fn), body)
        names.append(fn)

    submit = "#!/bin/bash\n# Part 1: submit all 30 jobs (20 ASR + 10 SQA). Independent; no chaining.\nset -e\ncd %s\n" % JOBS
    for fn in names:
        submit += 'sbatch %s\n' % fn
    write(os.path.join(JOBS, "submit_part1.sh"), submit)

    print("wrote %d sbatch + submit_part1.sh to %s" % (len(names), JOBS))
    print("  ASR : %d   SQA : %d" % (len(DATASETS) * len(CORRS), len(CORRS)))
    print("NOTHING submitted. Review, then: bash %s/submit_part1.sh" % JOBS)


if __name__ == "__main__":
    main()