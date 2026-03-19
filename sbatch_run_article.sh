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

MODEL_PATH=/reference/LLM/swiss-ai/Apertus-8B-Instruct-2509
DTYPE=bf16
BACKEND=transformers

python scripts/run_article_pipeline.py \
  --input "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox/swissdox_articles_lead.parquet" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col lead \
  --decision_col non_swiss \
  --batch_size 50 \
  --max_new_tokens 160 \
  --temperature 0.0


# --- Archive: outputs + prompt/config/sbatch ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
GOLD_CSV="data/swissdox_article_with_s_GOLD.csv"

# temporary run dir first
RUN_DIR="data/output/run_article_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# run evaluation and capture stdout
SCORE_LOG=$(python scripts/score.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_CSV" \
  --id_col article_id \
  --cols swiss \
  --col_kinds swiss=label \
  --rename_gold_cols non_swiss=swiss \
  --invert_gold_cols swiss \
  --max_rows 300 \
  --report_dir "$RUN_DIR/eval")

echo "$SCORE_LOG"

# extract numeric similarity from stdout
SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
SCORE=${SCORE:-NA}

# optional final renamed folder
if [ "$SCORE" = "NA" ]; then
  FINAL_RUN_DIR="data/output/run_article_with_s_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  FINAL_RUN_DIR="data/output/run_article_with_s${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$FINAL_RUN_DIR"

cp "$PRED_CSV" "$FINAL_RUN_DIR/" || true
cp "src/run_article_prompts.py" "$FINAL_RUN_DIR/prompts_used.py" || true
cp "src/run_article_config.py" "$FINAL_RUN_DIR/config_used.py" || true
cp "$0" "$FINAL_RUN_DIR/sbatch_used.sbatch" || true

# move eval reports too
if [ -d "$RUN_DIR/eval" ]; then
  mv "$RUN_DIR/eval" "$FINAL_RUN_DIR/eval"
fi

# cleanup temp dir if empty
rmdir "$RUN_DIR" 2>/dev/null || true

echo "Archived in: $FINAL_RUN_DIR"
echo "Score: ${SCORE}%"