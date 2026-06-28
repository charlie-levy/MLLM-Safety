#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 3 Slurm scripts.

Writes 30 inference sbatch ({mmsafety_tiny,vls_bench,holisafe} x 10 corruptions,
TIS only) plus submit_part3.sh into RUN/jobs/. Inference ONLY — no judge model is
loaded or staged (you score the responses separately). Produces STATIC files;
submits NOTHING.

Set PARTITION below before generating:
  "preemptable" -> free (billed 0.0); adds --qos=preemptable; jobs may be
                   preempted but every job resumes from its JSONL.  [budget-safe]
  "normal"      -> billed (GRES/gpu=2.0); not preempted; faster to finish.

  python gen_jobs.py
"""
import os

# ── partition: flip this one line, then re-run gen_jobs.py ───────────────────────
PARTITION = "normal"             # "preemptable" (free) | "normal" (billed)

CORRS = ["elastic_transform", "contrast", "frost", "defocus_blur", "glass_blur",
         "motion_blur", "zoom_blur", "snow", "fog", "jpeg_compression"]
DATASETS = ["mmsafety_tiny", "vls_bench", "holisafe"]

REPO = "/home/ch169788/llava_cot_eval"                  # git repo (scripts live here)
P3 = os.path.join(REPO, "experiments", "part3")         # where run_inference.py is
RUN = "/home/ch169788/experiments/part3"                # outputs / jobs / logs (not git-tracked)
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P3

# /tmp staging: copy the base model to node-local SSD (the LoRA adapter loads from
# /home, unaffected). Inference-only -> just Xkev, NO judge. cp -an is a no-op if
# already present.
STAGE = """mkdir -p /tmp/hf/hub
cp -an ~/.cache/huggingface/hub/models--Xkev--Llama-3.2V-11B-cot /tmp/hf/hub/ 2>/dev/null
export HF_HOME=/tmp/hf"""

_QOS = "#SBATCH --qos=preemptable\n" if PARTITION == "preemptable" else ""
HEADER = """#!/bin/bash
#SBATCH --job-name=%(jn)s
#SBATCH --partition=""" + PARTITION + """
""" + _QOS + """#SBATCH --gres=gpu:nvidia_h100_pcie:1
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
    for ds in DATASETS:
        for c in CORRS:
            name = "infer_%s_%s" % (ds, c)
            body = HEADER % {"jn": "p3_%s_%s" % (ds, c), "logs": LOGS, "name": name}
            body += "\n" + ENV + "\n\n" + STAGE + "\n\n"
            body += ("python run_inference.py --dataset %s --corruption %s --output_dir %s\n"
                     % (ds, c, RESULTS))
            fn = "%s.sh" % name
            write(os.path.join(JOBS, fn), body)
            names.append(fn)

    submit = ("#!/bin/bash\n# Part 3: submit all %d inference jobs (3 datasets x 10 corruptions, TIS).\n"
              "# Responses only, no judge. Independent; no chaining.\nset -e\ncd %s\n" % (len(names), JOBS))
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part3.sh"), submit)

    print("wrote %d sbatch + submit_part3.sh to %s" % (len(names), JOBS))
    print("  partition=%s%s" % (PARTITION, "  (FREE)" if PARTITION == "preemptable" else "  (BILLED)"))
    print("  datasets=%s   corruptions=%d" % (DATASETS, len(CORRS)))
    print("NOTHING submitted. Review, then: bash %s/submit_part3.sh" % JOBS)


if __name__ == "__main__":
    main()
