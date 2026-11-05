#!/bin/bash -l
#SBATCH --job-name=step3a_stance_array
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=3-00:00:00
#SBATCH --output=logs/step3a_stance_array_%A_%a.out
#SBATCH --error=logs/step3a_stance_array_%A_%a.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL
#SBATCH --array=0-7   # 8 GPUs en parallèle → ~6 250 lignes chacun pour 50 000 lignes

# Usage: sbatch sbatch_step3a_stance_array.sh <PREV_JOB_ID> [MAX_ROWS]
# Exemple: sbatch sbatch_step3a_stance_array.sh 60015520
# Exemple (test sur 1000 lignes): sbatch sbatch_step3a_stance_array.sh 60015520 1000
# Le PREV_JOB_ID est l'ID du job de sbatch_step2_sentence_cutting.sh
# MAX_ROWS (optionnel) limite le dataset d'entrée aux N premières lignes
# AVANT le découpage en tâches de l'array : le fichier fusionné en sortie
# (après sbatch_step3b_stance_merge.sh) contiendra alors exactement N lignes, pas plus.

set -eo pipefail

# Logs écrits immédiatement dans le .out (sinon Python les garde en mémoire)
export PYTHONUNBUFFERED=1

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO

cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (partagé par les 8 tâches de l'array) ───
OUTDIR="$WORKDIR/data/output/step3a_stance_array_job${SLURM_ARRAY_JOB_ID}"
OUTBASE="$OUTDIR/criticism_stance"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_step2_sentence_cutting.sh) ───────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_step3a_stance_array.sh <PREV_JOB_ID>"
    exit 1
fi
INPUT="$WORKDIR/data/output/step2_sentence_cutting_job${PREV_JOB_ID}/criticism_base.csv"

# ── MAX_ROWS: limite optionnelle du nombre de lignes à traiter (test rapide) ─
MAX_ROWS="${2:-}"

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

MAX_ROWS_ARGS=()
if [ -n "$MAX_ROWS" ]; then
    echo "[INFO] MAX_ROWS=$MAX_ROWS : seules les $MAX_ROWS premières lignes du dataset seront traitées."
    MAX_ROWS_ARGS=(--max_rows "$MAX_ROWS")
fi

python scripts/step3a_stance_array.py \
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
  --num_tasks "$NUM_TASKS" \
  ${MAX_ROWS_ARGS[@]+"${MAX_ROWS_ARGS[@]}"}
  # --task_id est lu automatiquement depuis SLURM_ARRAY_TASK_ID

echo "Task ${SLURM_ARRAY_TASK_ID} terminée."
