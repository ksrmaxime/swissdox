#!/bin/bash
#SBATCH --job-name=swissdox_s_t_s_p
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/pipeline_s_t_s_p_%j.out
#SBATCH --error=logs/pipeline_s_t_s_p_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO

# ---- INPUT (à ajuster si besoin) ----
INPUT0="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox/swissdox_sentences.parquet"

# ---- OUTPUT RUN FOLDER (auto jobid) ----
RUN_DIR_REL="data/output/pipeline_job${SLURM_JOB_ID}"
RUN_DIR_ABS="${WORKDIR}/${RUN_DIR_REL}"
PROMPTS_DIR="${RUN_DIR_ABS}/prompts"

# ---- MODEL ----
MODEL_PATH="/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509"
DTYPE="bf16"

# ---- Common params ----
TEXT_COL="sentence"
BATCH_SIZE=50
TEMP=0.0
MAX_NEW_TOKENS=160

mkdir -p "${WORKDIR}/logs"
mkdir -p "${RUN_DIR_ABS}"
mkdir -p "${PROMPTS_DIR}"

cd "$WORKDIR"
source .venv/bin/activate

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

# ------------------------------------------------------------
# RUN 1: SWISS_RELATED (YES/NO)
# ------------------------------------------------------------
OUTBASE1="${RUN_DIR_ABS}/swissdox_sentences_with_s"

python scripts/run1_pipeline.py \
  --input "$INPUT0" \
  --output_base "$OUTBASE1" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --trust_remote_code \
  --text_col "$TEXT_COL" \
  --decision_col SWISS_RELATED \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP"

IN2="${OUTBASE1}_job${SLURM_JOB_ID}.parquet"

# ------------------------------------------------------------
# RUN 2: TOPIC
# ------------------------------------------------------------
OUTBASE2="${RUN_DIR_ABS}/swissdox_sentences_with_s_t"

python scripts/run2_pipeline.py \
  --input "$IN2" \
  --output_base "$OUTBASE2" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --trust_remote_code \
  --text_col "$TEXT_COL" \
  --swiss_related_col SWISS_RELATED \
  --topic_col TOPIC \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP"

IN3="${OUTBASE2}_job${SLURM_JOB_ID}.parquet"

# ------------------------------------------------------------
# RUN 3: SENTIMENT toward public administration (sp)
# ------------------------------------------------------------
OUTBASE3="${RUN_DIR_ABS}/swissdox_sentences_with_s_t_s"

python scripts/run3_pipeline.py \
  --input "$IN3" \
  --output_base "$OUTBASE3" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --trust_remote_code \
  --text_col "$TEXT_COL" \
  --swiss_related_col SWISS_RELATED \
  --sentiment_col SENTIMENT \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP"

IN4="${OUTBASE3}_job${SLURM_JOB_ID}.parquet"

# ------------------------------------------------------------
# RUN 4: POPULISM (p)
# ------------------------------------------------------------
OUTBASE4="${RUN_DIR_ABS}/swissdox_sentences_with_s_t_s_p"

python scripts/run4_pipeline.py \
  --input "$IN4" \
  --output_base "$OUTBASE4" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --trust_remote_code \
  --text_col "$TEXT_COL" \
  --swiss_related_col SWISS_RELATED \
  --populism_col POPULISM \
  --batch_size "$BATCH_SIZE" \
  --max_new_tokens "$MAX_NEW_TOKENS" \
  --temperature "$TEMP"

FINAL_CSV_WITH_JOB="${OUTBASE4}_job${SLURM_JOB_ID}.csv"
FINAL_PARQUET_WITH_JOB="${OUTBASE4}_job${SLURM_JOB_ID}.parquet"

# ------------------------------------------------------------
# FINALIZATION: copy prompts + final “clean name” csv
# ------------------------------------------------------------
# Copie des prompts utilisés (depuis src/)
cp -f "src/run1_prompts.py" "${PROMPTS_DIR}/run1_prompts.py"
cp -f "src/run2_prompts.py" "${PROMPTS_DIR}/run2_prompts.py"
cp -f "src/run3_prompts.py" "${PROMPTS_DIR}/run3_prompts.py"
cp -f "src/run4_prompts.py" "${PROMPTS_DIR}/run4_prompts.py"

# Fichier final demandé (sans suffixe jobid)
cp -f "$FINAL_CSV_WITH_JOB" "${RUN_DIR_ABS}/swissdox_sentences_with_s_t_s_p.csv"
cp -f "$FINAL_PARQUET_WITH_JOB" "${RUN_DIR_ABS}/swissdox_sentences_with_s_t_s_p.parquet"

# Garder une trace du sbatch
cp -f "$0" "${RUN_DIR_ABS}/sbatch_used.sbatch"

echo "DONE. Output folder:"
echo "${RUN_DIR_ABS}"
echo "Final CSV:"
echo "${RUN_DIR_ABS}/swissdox_sentences_with_s_t_s_p.csv"