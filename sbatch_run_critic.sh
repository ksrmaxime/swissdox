#!/bin/bash
#SBATCH --job-name=run_critic_stance
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_critic_stance_%j.out
#SBATCH --error=logs/run_critic_stance_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/critic_stance"

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
MAX_ROWS=  # Nombre max de lignes traitées par le LLM (laisser vide pour tout traiter)

python scripts/run_critic_pipeline.py \
  --input "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/critic_base.csv" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col sentence \
  --decision_col STANCE \
  --batch_size 16 \
  --max_new_tokens 300 \
  --temperature 0.0 \
  ${MAX_ROWS:+--max_rows "$MAX_ROWS"}


# --- Archive: outputs + prompt/config/sbatch ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"
GOLD_CSV="data/critic_gold.csv"

# temporary run dir first
RUN_DIR="data/output/run_critic_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

# run evaluation and capture stdout
SCORE_LOG=$(python scripts/score.py \
  --pred "$PRED_CSV" \
  --gold "$GOLD_CSV" \
  --id_col sentence_id \
  --cols STANCE \
  --col_kinds STANCE=label \
  --max_rows 300 \
  --report_dir "$RUN_DIR/eval")

echo "$SCORE_LOG"

# extract numeric similarity from stdout
SCORE=$(echo "$SCORE_LOG" | awk '/^Similarity:/ {gsub(/%/,"",$2); print $2; exit}')
SCORE=${SCORE:-NA}

# optional final renamed folder
if [ "$SCORE" = "NA" ]; then
  FINAL_RUN_DIR="data/output/run_critic_no_score_job${SLURM_JOB_ID}"
else
  SCORE_TAG=$(printf "%.2f" "$SCORE" | tr '.' 'p')
  FINAL_RUN_DIR="data/output/run_critic_stance${SCORE_TAG}_job${SLURM_JOB_ID}"
fi

mkdir -p "$FINAL_RUN_DIR"

cp "$PRED_CSV" "$FINAL_RUN_DIR/" || true
cp "src/run_critic_prompts.py" "$FINAL_RUN_DIR/prompts_used.py" || true
cp "src/run_critic_config.py" "$FINAL_RUN_DIR/config_used.py" || true
cp "$0" "$FINAL_RUN_DIR/sbatch_used.sbatch" || true

# move eval reports too
if [ -d "$RUN_DIR/eval" ]; then
  mv "$RUN_DIR/eval" "$FINAL_RUN_DIR/eval"
fi

# cleanup temp dir if empty
rmdir "$RUN_DIR" 2>/dev/null || true

echo "Archived in: $FINAL_RUN_DIR"
echo "Score: ${SCORE}%"
