#!/bin/bash
# Submit all R1-OneVision evaluation jobs.
# BEFORE RUNNING: verify R1ONEVISION_MODEL_ID in code/config.py.
# Usage: bash slurm_scripts/submit_r1onevision.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

WRAP_HEADER="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
"

# ── ORR (XSTest + MMSA, clean) ─────────────────────────────────────────────
sbatch --job-name="r1_orr" \
       --partition=normal \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=6:00:00 \
       --output="logs/r1_orr.log" \
       --wrap="${WRAP_HEADER}python code/eval_orr_r1.py"
echo "Submitted: r1_orr"

# ── FigStep clean baseline ──────────────────────────────────────────────────
sbatch --job-name="r1_figstep_clean" \
       --partition=normal \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=4:00:00 \
       --output="logs/r1_figstep_clean.log" \
       --wrap="${WRAP_HEADER}python code/eval_figstep_r1.py"
echo "Submitted: r1_figstep_clean"

# ── SQA clean baseline ──────────────────────────────────────────────────────
sbatch --job-name="r1_sqa_clean" \
       --partition=normal \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=64G \
       --time=2:00:00 \
       --output="logs/r1_sqa_clean.log" \
       --wrap="${WRAP_HEADER}python code/eval_sqa_r1.py"
echo "Submitted: r1_sqa_clean"

# ── FigStep noise sweep sev 1-5 ─────────────────────────────────────────────
for SEV in 1 2 3 4 5; do
  sbatch --job-name="r1_fig_n_sev${SEV}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --output="logs/r1_figstep_noise_sev${SEV}.log" \
         --wrap="${WRAP_HEADER}python code/eval_figstep_r1.py --severity ${SEV} --noise_type gaussian_noise"
  echo "Submitted: r1_figstep_noise_sev${SEV}"

  sbatch --job-name="r1_fig_b_sev${SEV}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=4:00:00 \
         --output="logs/r1_figstep_blur_sev${SEV}.log" \
         --wrap="${WRAP_HEADER}python code/eval_figstep_r1.py --severity ${SEV} --noise_type gaussian_blur"
  echo "Submitted: r1_figstep_blur_sev${SEV}"

  sbatch --job-name="r1_sqa_n_sev${SEV}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=2:00:00 \
         --output="logs/r1_sqa_noise_sev${SEV}.log" \
         --wrap="${WRAP_HEADER}python code/eval_sqa_r1.py --severity ${SEV} --noise_type gaussian_noise"
  echo "Submitted: r1_sqa_noise_sev${SEV}"

  sbatch --job-name="r1_sqa_b_sev${SEV}" \
         --partition=normal \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time=2:00:00 \
         --output="logs/r1_sqa_blur_sev${SEV}.log" \
         --wrap="${WRAP_HEADER}python code/eval_sqa_r1.py --severity ${SEV} --noise_type gaussian_blur"
  echo "Submitted: r1_sqa_blur_sev${SEV}"
done
