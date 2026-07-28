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

# Usage: sbatch sbatch_04_classify_stance.sh <PREV_JOB_ID>
# Exemple: sbatch sbatch_04_classify_stance.sh 60015520
# Le PREV_JOB_ID est l'ID du job de sbatch_03_extract_sentences.sh

set -eo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO

cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (partagé par les 8 tâches de l'array) ───
OUTDIR="$WORKDIR/data/output/04_classify_stance_job${SLURM_ARRAY_JOB_ID}"
OUTBASE="$OUTDIR/critic_stance"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_03_extract_sentences.sh) ───────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_04_classify_stance.sh <PREV_JOB_ID>"
    exit 1
fi
INPUT="$WORKDIR/data/output/03_extract_sentences_job${PREV_JOB_ID}/critic_base.csv"

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
  --input "$INPUT" \
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
