#!/bin/bash -l
#SBATCH --job-name=critic_stance_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/critic_array_%A_%a.out
#SBATCH --error=logs/critic_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7   # 8 GPUs en parallèle → ~6 250 lignes chacun pour 50 000 lignes

set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
dcsrsoft use 20241118
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/critic_stance"

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ARRAY ==="
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>} TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers
NUM_TASKS=8   # doit correspondre au nombre de tâches dans --array

python scripts/04_classify_stance.py \
  --input "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/critic_base.csv" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col sentence \
  --decision_col STANCE \
  --batch_size 4 \
  --max_new_tokens 300 \
  --temperature 0.0 \
  --num_tasks "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminée."
