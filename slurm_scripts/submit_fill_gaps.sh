#!/bin/bash
# Re-run the corruption cells that failed in the first batch (figstep+orr+sqa for
# each, to guarantee no partial gaps):
#   SAGE brightness p40,p60,p80 | MSR brightness p20,p40,p60,p80 | TIS pixelate p20
# 8 cells x 3 metrics = 24 jobs. Offline-safe. After they finish:
#   bash slurm_scripts/submit_judge_realistic.sh   (judge the new SQA)
#   python3 code/rescore_from_responses.py
# Usage: bash slurm_scripts/submit_fill_gaps.sh

cd /home/ch169788/llava_cot_eval
mkdir -p logs

submit () {  # $1=jobname $2=logname $3=hours $4=args
  sbatch --job-name="$1" --partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 \
         --mem=64G --time="$3:00:00" --exclude=evc42 --output="logs/$2.log" --wrap="
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

run_cell () {  # $1=flag $2=short $3=corrupt $4=pct
  submit "${2}_${3}f_${4}" "${2}_figstep_${3}_${4}" 8 "eval_figstep_noise_sweep.py $1 --corrupt $3 --corrupt_pct $4"
  submit "${2}_${3}o_${4}" "${2}_orr_${3}_${4}"     8 "eval_orr_noise_sweep.py $1 --corrupt $3 --corrupt_pct $4"
  submit "${2}_${3}s_${4}" "${2}_sqa_${3}_${4}"     4 "eval_sqa_noise_sweep.py $1 --corrupt $3 --corrupt_pct $4"
}

for P in 40 60 80;     do run_cell "--use_sage" sage brightness $P; done
for P in 20 40 60 80;  do run_cell "--use_msr"  msr  brightness $P; done
run_cell "--use_tis" tis pixelate 20

echo ""
echo "Submitted 24 gap-fill jobs. After they finish: submit_judge_realistic.sh, then rescore."
