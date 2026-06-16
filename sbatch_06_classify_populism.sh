#!/bin/bash -l
#SBATCH --job-name=populism_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/populism_array_%A_%a.out
#SBATCH --error=logs/populism_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7   # 8 GPUs en parallèle

# Usage: sbatch sbatch_06_classify_populism.sh <CRITIC_JOB_ID>
# Exemple: sbatch sbatch_06_classify_populism.sh 60015520
# Le CRITIC_JOB_ID est l'ID du job de sbatch_05_merge_stance.sh

set -eo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/populism"

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ARRAY ==="
echo "ARRAY_JOB_ID=${SLURM_ARRAY_JOB_ID:-<unset>} TASK_ID=${SLURM_ARRAY_TASK_ID:-<unset>}"
echo "HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

CRITIC_JOB_ID=${1:-""}
if [ -z "$CRITIC_JOB_ID" ]; then
    echo "[ERROR] Passer le CRITIC_JOB_ID en argument: sbatch sbatch_06_classify_populism.sh <CRITIC_JOB_ID>"
    exit 1
fi

# Input: fichier merged produit par sbatch_05_merge_stance.sh
INPUT_PARQUET="$WORKDIR/data/processed/critic_stance_merged_job${CRITIC_JOB_ID}.parquet"
INPUT_CSV="$WORKDIR/data/processed/critic_stance_merged_job${CRITIC_JOB_ID}.csv"

if [ -f "$INPUT_PARQUET" ]; then
    INPUT="$INPUT_PARQUET"
elif [ -f "$INPUT_CSV" ]; then
    INPUT="$INPUT_CSV"
else
    echo "[ERROR] Fichier d'entrée introuvable: $INPUT_PARQUET"
    exit 1
fi

MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers
NUM_TASKS=8   # doit correspondre au nombre de tâches dans --array

python scripts/06_classify_populism.py \
  --input "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col sentence \
  --stance_col STANCE \
  --batch_size 8 \
  --max_new_tokens 150 \
  --temperature 0.0 \
  --num_tasks "$NUM_TASKS"
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminée."
