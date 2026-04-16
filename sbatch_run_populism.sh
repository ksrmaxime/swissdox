#!/bin/bash
#SBATCH --job-name=run_populism
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --output=logs/run_populism_%j.out
#SBATCH --error=logs/run_populism_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export PYTORCH_ALLOC_CONF=expandable_segments:True

module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
OUTDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed
OUTBASE="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/populism"

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
MAX_ROWS=  # Max number of CRITIC rows sent to the LLM (leave empty to process all)

# Input: output CSV produced by run_critic_pipeline (must contain STANCE column)
INPUT_CSV="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/PARP_RUN3_CRITIC_INPUT.csv"
# If you want to point to a specific critic output, replace the line above with e.g.:
# INPUT_CSV="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/critic_stance_job12345.csv"

python scripts/run_populism_pipeline.py \
  --input "$INPUT_CSV" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col sentence \
  --stance_col STANCE \
  --batch_size 16 \
  --max_new_tokens 150 \
  --temperature 0.0 \
  ${MAX_ROWS:+--max_rows "$MAX_ROWS"}


# --- Archive: outputs + prompt/config/sbatch ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"

RUN_DIR="data/output/run_populism_job${SLURM_JOB_ID}"
mkdir -p "$RUN_DIR"

FINAL_RUN_DIR="data/output/run_populism_job${SLURM_JOB_ID}"
mkdir -p "$FINAL_RUN_DIR"

cp "$PRED_CSV" "$FINAL_RUN_DIR/" || true
cp "src/run_populism_prompts.py" "$FINAL_RUN_DIR/prompts_used.py" || true
cp "src/run_populism_config.py" "$FINAL_RUN_DIR/config_used.py" || true
cp "$0" "$FINAL_RUN_DIR/sbatch_used.sbatch" || true

rmdir "$RUN_DIR" 2>/dev/null || true

echo "Archived in: $FINAL_RUN_DIR"
