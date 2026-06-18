#!/bin/bash
# ============================================================================
# Re-run ONLY the Llama Guard 3 Vision ASR judge for the MSR-Align eval.
# Inference is already done (responses_figstep.json exists for both conditions),
# so this just scores those saved responses — no re-inference. One H100 job
# judges clean then blur20 (~10-15 min total).
#
#   bash slurm_scripts/submit_msr_guard_asr.sh
# When it finishes:  python code/report_msr_guard.py
# ============================================================================
set -euo pipefail

REPO=/home/ch169788/llava_cot_eval
cd "$REPO"
mkdir -p logs

ENVBLOCK='source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export PYTHONPATH=/home/ch169788/llava_cot_eval/code:$PYTHONPATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_XET=1
export HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1
cd /home/ch169788/llava_cot_eval'

COMMON="--partition=preemptable --qos=preemptable --gres=gpu:nvidia_h100_pcie:1 --mem=80G --exclude=evc42"

JID=$(sbatch --parsable $COMMON --job-name=msrg_asr_rerun --time=3:00:00 \
  --output=logs/msrg_asr_rerun.log --wrap="${ENVBLOCK}
python code/judge_figstep_guard.py results/msr_guard_eval/clean/responses_figstep.json  --out_dir results/msr_guard_eval/clean  || exit 1
python code/judge_figstep_guard.py results/msr_guard_eval/blur20/responses_figstep.json --out_dir results/msr_guard_eval/blur20 || exit 1")

echo "Submitted msrg_asr_rerun = $JID"
echo "Watch:  tail -f logs/msrg_asr_rerun.log    (look for 'ASR (Llama Guard) = ..%')"
echo "Then:   python code/report_msr_guard.py"
