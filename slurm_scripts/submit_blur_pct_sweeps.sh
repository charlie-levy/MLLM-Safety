#!/bin/bash
# MSR only, blur 20% and 40%

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {
  sbatch --job-name="$1" \
         --partition=preemptable --qos=preemptable \
         --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G \
         --time="$3:00:00" \
         --exclude=evc42 \
         --output="logs/$2.log" \
         --wrap="
source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:\$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

cd /home/ch169788/llava_cot_eval
python code/$4
"
  echo "Submitted: $1"
}

# MSR only
SHORT="msr"
FLAG="--use_msr"

# Blur only up to 40%
for P in 20 40; do
    submit "${SHORT}_bpf_${P}" "${SHORT}_figstep_blurpct_${P}" 8 \
           "eval_figstep_noise_sweep.py ${FLAG} --blur_pct ${P}"

    submit "${SHORT}_bpo_${P}" "${SHORT}_orr_blurpct_${P}" 8 \
           "eval_orr_noise_sweep.py ${FLAG} --blur_pct ${P}"

    submit "${SHORT}_bps_${P}" "${SHORT}_sqa_blurpct_${P}" 4 \
           "eval_sqa_noise_sweep.py ${FLAG} --blur_pct ${P}"
done

echo ""
echo "Submitted 6 MSR blur sweep jobs (20%, 40%)."