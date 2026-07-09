#!/usr/bin/env python
"""
gen_jobs.py — one-time materializer for the Part 8 (prompt-based defense) Slurm
scripts. Static sbatch, one per (model x condition x prompt); submits NOTHING.

Grid (deliberately SMALL): llava_cot x 4 conditions x {safety, blur_safe} on SIUO
= 8 cells (responses only, no judge). The "none" baseline == Part 4, reused.
Same frozen decode as Part 4 (run_eval 2048 tok); the only difference is the
prepended safety system prompt.

To widen: add models to MODELS (driver supports all 6 Part-4 models) and/or add
"perceive" to the grid. Loads from ~/.cache (HF_HUB_OFFLINE) with --mem=80G.
  python gen_jobs.py
  bash $JOBS/submit_part8.sh     # all 8 (== submit_part8_llava.sh here)
"""
import os

from prompts import INTERVENTIONS   # ["safety", "blur_safe"]

# ── partition: flip this one line, then re-run gen_jobs.py ───────────────────────
PARTITION = "preemptable"        # "preemptable" (free, resume-safe) | "normal" (billed)

CONDITIONS = ["clean", "zoom_blur", "snow", "glass_blur"]
# small grid: just the TIS base (llava_cot), the flagship reasoning model. Driver
# supports all 6 Part-4 models; add here to widen.
MODELS = ["llava_cot"]
PROMPTS = INTERVENTIONS

TIME = {"llava_cot": "04:00:00", "base_llama": "03:00:00", "llamav_o1": "04:00:00",
        "r1_onevision": "04:00:00", "r1_onevision_nothink": "03:00:00", "qwen2_5_vl": "03:00:00"}

REPO = "/home/ch169788/llava_cot_eval"
P8 = os.path.join(REPO, "experiments", "part8_prompt_defense")   # run_inference.py / prompts.py
RUN = "/home/ch169788/experiments/part8"                         # outputs / jobs / logs (not git-tracked)
JOBS = os.path.join(RUN, "jobs")
LOGS = os.path.join(RUN, "logs")
RESULTS = os.path.join(RUN, "results")

ENV = """source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
cd %s""" % P8

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
            for p in PROMPTS:
                name = "infer_siuo_%s_%s_%s" % (c, m, p)
                body = HEADER % {"jn": "p8_%s_%s_%s" % (m, c, p), "logs": LOGS,
                                 "name": name, "time": TIME[m]}
                body += "\n" + ENV + "\n\n"
                body += ("python run_inference.py --model %s --condition %s --prompt %s --output_dir %s\n"
                         % (m, c, p, RESULTS))
                fn = "%s.sh" % name
                write(os.path.join(JOBS, fn), body)
                names.append(fn)

    # submit-all
    submit = ("#!/bin/bash\n# Part 8: submit all %d jobs (%d models x %d conditions x %d prompts on SIUO).\n"
              "# Responses only, no judge. Independent; resume-safe per-idx.\nset -e\ncd %s\n"
              % (len(names), len(MODELS), len(CONDITIONS), len(PROMPTS), JOBS))
    for fn in names:
        submit += "sbatch %s\n" % fn
    write(os.path.join(JOBS, "submit_part8.sh"), submit)

    # per-model submit scripts (staged rollout). Reconstruct the exact filenames
    # for this model so we never mis-match on the underscored model keys.
    for m in MODELS:
        subset = ["infer_siuo_%s_%s_%s.sh" % (c, m, p) for c in CONDITIONS for p in PROMPTS]
        body = ("#!/bin/bash\n# Part 8: submit ONLY %s (%d conditions x %d prompts).\nset -e\ncd %s\n"
                % (m, len(CONDITIONS), len(PROMPTS), JOBS))
        for fn in subset:
            body += "sbatch %s\n" % fn
        short = "llava" if m == "llava_cot" else m
        write(os.path.join(JOBS, "submit_part8_%s.sh" % short), body)

    print("wrote %d sbatch + submit_part8.sh (+ per-model submits) to %s" % (len(names), JOBS))
    print("  partition=%s%s" % (PARTITION, "  (FREE)" if PARTITION == "preemptable" else "  (BILLED)"))
    print("  models=%s" % MODELS)
    print("  conditions=%s" % CONDITIONS)
    print("  prompts=%s   (baseline 'none' == Part 4, reused)" % PROMPTS)
    print("NOTHING submitted. Debug first, then: bash %s/submit_part8_llava.sh" % JOBS)


if __name__ == "__main__":
    main()