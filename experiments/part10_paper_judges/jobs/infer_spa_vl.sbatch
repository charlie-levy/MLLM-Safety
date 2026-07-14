#!/bin/bash
#SBATCH --job-name=p10_spa_vl
#SBATCH --partition=normal
#SBATCH --gres=gpu:nvidia_h100_pcie:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=80G
#SBATCH --time=07:00:00
#SBATCH --requeue
#SBATCH --exclude=evc16,evc43,evc26
#SBATCH --output=/home/ch169788/experiments/part10/logs/%x_%j.out
#SBATCH --error=/home/ch169788/experiments/part10/logs/%x_%j.err

# --- self-heal: tolerate transient GPU-init delay (retry ~30s), and if the GPU is truly
#     absent, PERSIST the exclusion across requeues via a badnodes file. (scontrol resets
#     ExcNodeList to the batch --exclude on every requeue, so a file is required to
#     accumulate dead nodes; we re-apply the full list before each requeue.) ---
BADF=/home/ch169788/experiments/part10/.badnodes
gpu_ok=0
for _i in 1 2 3 4 5 6; do
  if nvidia-smi -L 2>/dev/null | grep -q 'GPU'; then gpu_ok=1; break; fi
  sleep 5
done
if [ "$gpu_ok" -eq 0 ]; then
  echo "[requeue] no GPU on $(hostname) after ~30s — excluding + requeueing"
  echo "$(hostname)" >> "$BADF"
  BAD=$(sort -u "$BADF" | paste -sd, -)
  scontrol update JobId="$SLURM_JOB_ID" ExcNodeList="evc16,evc43,$BAD"
  scontrol requeue "$SLURM_JOB_ID"; sleep 10; exit 1
fi
echo "[ok] GPU present on $(hostname): $(nvidia-smi -L | head -1)"

source /apps/anaconda/anaconda-2024.10/etc/profile.d/conda.sh
conda activate REU
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_XET=1 HF_HUB_ENABLE_HF_TRANSFER=0
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1

cd ~/llava_cot_eval/experiments/part10_paper_judges
python run_inference.py --dataset spa_vl --conditions clean,zoom_blur
