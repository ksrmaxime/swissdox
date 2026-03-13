#!/bin/bash
#SBATCH --job-name=run_article_swissrelated
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_article_swissrelated_%j.out
#SBATCH --error=logs/run_article_swissrelated_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
SCRATCHDIR=/scratch/mkaiser3
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox_article_with_s_t"

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) PARTITION=${SLURM_JOB_PARTITION:-<unset>}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"
echo "DATE=$(date -Is)"
nvidia-smi -L || true

python scripts/run_article_pipeline.py \
  --input "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox/swissdox_articles_lead.parquet" \
  --output_base "$OUTBASE" \
  --model_path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --trust_remote_code \
  --text_col lead \
  --decision_col non_swiss \
  --batch_size 50 \
  --max_new_tokens 160 \
  --temperature 0.0


# --- Archive: outputs + prompt/config/sbatch ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"

# ton fichier "gold" (humain) ici:
GOLD_CSV="data/swissdox_article_with_s_GOLD.csv"

# capture du score (ligne: "Similarity: 51.08%")
SCORE=$(python scripts/score.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_CSV" \
  --use_row_order \
  --cols non_swiss \
  --max_rows 300 \
  | awk '/Similarity:/ {gsub(/%/,"",$2); print $2}')

# normaliser pour nom de dossier (51.08 -> 51p08)
SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')

RUN_DIR="data/output/run_article_with_s${SCORE_TAG}"
mkdir -p "$RUN_DIR"

cp "$PRED_CSV" "$RUN_DIR/" || true
cp "src/run_all_prompts.py" "$RUN_DIR/prompts_used.py"
cp "$0" "$RUN_DIR/sbatch_used.sbatch"

echo "Archived in: $RUN_DIR"
echo "Score: ${SCORE}%"