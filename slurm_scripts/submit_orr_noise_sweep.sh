#!/bin/bash
# Submit ORR noise sweep jobs (gaussian_noise + gaussian_blur, sev 1-5, base + TIS).
# 20 jobs total. Each runs XSTest (250) + MMSA (428) = 678 samples.
# Estimated wall time: ~4h per job on a V100-32GB node.
#
# Usage: bash slurm_scripts/submit_orr_noise_sweep.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

WRAP_HEADER="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/ch169788/llava_cot_eval
"

for NOISE in gaussian_noise gaussian_blur; do
  for SEV in 1 2 3 4 5; do

    # BASE model
    sbatch --job-name="orr_base_${NOISE:0:5}_s${SEV}" \
           --partition=normal \
           --gres=gpu:tesla_v100-pcie-32gb:1 \
           --mem=64G \
           --time=5:00:00 \
           --output="logs/orr_base_${NOISE}_sev${SEV}.log" \
           --wrap="${WRAP_HEADER}python code/eval_orr_noise_sweep.py --severity ${SEV} --noise_type ${NOISE}"
    echo "Submitted: orr_base ${NOISE} sev${SEV}"

    # BASE + TIS
    sbatch --job-name="orr_tis_${NOISE:0:5}_s${SEV}" \
           --partition=normal \
           --gres=gpu:tesla_v100-pcie-32gb:1 \
           --mem=64G \
           --time=5:00:00 \
           --output="logs/orr_tis_${NOISE}_sev${SEV}.log" \
           --wrap="${WRAP_HEADER}python code/eval_orr_noise_sweep.py --severity ${SEV} --noise_type ${NOISE} --use_tis"
    echo "Submitted: orr_tis ${NOISE} sev${SEV}"

  done
done

echo ""
echo "Submitted 20 ORR noise sweep jobs."
echo "Results will appear in: results/orr_noise_sweep/"
