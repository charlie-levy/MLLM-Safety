#!/bin/bash
# MSR-Align CLEAN images: fresh re-inference, then LLaMA-judged ASR + ORR
# (alternative to string matching). One H100 job, ~1.5-2 h.
#
#   FigStep  -> ASR   (LLaMA judge: COMPLY vs REFUSE)
#   XSTest+MMSA -> ORR (LLaMA judge: ANSWER vs REFUSE / SAFE vs OVERREFUSE)
#
# Outputs (full responses + per-sample verdicts):
#   results/figstep_noise_sweep/judged_llama_base_msr_clean.{csv,json}
#   results/orr/judged_llama_base_msr_clean.{csv,json}

cd /home/ch169788/llava_cot_eval
mkdir -p logs

sbatch --job-name="msr_clean_llama" \
       --partition=preemptable --qos=preemptable \
       --gres=gpu:nvidia_h100_pcie:1 \
       --mem=80G \
       --time=4:00:00 \
       --exclude=evc42 \
       --output="logs/msr_clean_llama_judge.log" \
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

echo '########## STEP 1: clean re-inference (LLaVA-CoT + MSR-Align) ##########'
python code/eval_clean_msr.py || exit 1

echo '########## STEP 2: LLaMA-judged ASR (FigStep, clean) ##########'
python code/judge_safety_hf.py --mode asr results/figstep_noise_sweep/responses_base_msr_clean.csv || exit 1

echo '########## STEP 3: LLaMA-judged ORR (XSTest + MMSA, clean) ##########'
python code/judge_safety_hf.py --mode orr results/orr/responses_base_msr_clean.csv || exit 1

echo '########## DONE ##########'
"
echo "Submitted: msr_clean_llama  (re-inference + LLaMA judge, clean images)"
