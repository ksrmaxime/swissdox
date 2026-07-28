#!/bin/bash -l
#SBATCH --job-name=run_article_swissrelated
#SBATCH --partition=gpu-l40
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=10:00:00
#SBATCH --output=logs/run_article_swissrelated_%j.out
#SBATCH --error=logs/run_article_swissrelated_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch sbatch_02_classify_articles.sh <PREV_JOB_ID>
# Exemple: sbatch sbatch_02_classify_articles.sh 60015520
# Le PREV_JOB_ID est l'ID du job de sbatch_01_download.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
SCRATCHDIR=/scratch/mkaiser3

cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/02_classify_articles_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_01_download.sh) ────────────────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_02_classify_articles.sh <PREV_JOB_ID>"
    exit 1
fi
INPUT="$WORKDIR/data/output/01_download_job${PREV_JOB_ID}/swissdox_articles_lead.parquet"
OUTBASE="$OUTDIR/articles_classified"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

PYTORCH_ALLOC_CONF=expandable_segments:True python scripts/02_classify_articles.py \
  --input "$INPUT" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col lead \
  --decision_col non_swiss \
  --batch_size 16 \
  --max_new_tokens 160 \
  --temperature 0.0


# --- Score + archive prompt/config/sbatch utilisés, tout dans $OUTDIR ---
PRED_CSV="${OUTBASE}.csv"
GOLD_CSV="data/PARP_RUN1_Scoring_Gold.csv"

SCORE_LOG=$(python scripts/score.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_CSV" \
  --id_col article_id \
  --cols swiss \
  --col_kinds swiss=label \
  --rename_gold_cols non_swiss=swiss \
  --invert_gold_cols swiss \
  --max_rows 302 \
  --report_dir "$OUTDIR/eval")

echo "$SCORE_LOG"

SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
SCORE=${SCORE:-NA}
echo "${SCORE}" > "$OUTDIR/score.txt"

cp "src/article_prompts.py" "$OUTDIR/prompts_used.py" || true
cp "src/article_config.py" "$OUTDIR/config_used.py" || true
cp "$0" "$OUTDIR/sbatch_used.sbatch" || true

echo "Archived in: $OUTDIR"
echo "Score: ${SCORE}%"