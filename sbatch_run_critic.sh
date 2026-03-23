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

python scripts/run_critic_pipeline.py \
  --input "/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/critic_base.csv" \
  --output_base "$OUTBASE" \
  --model_path "$MODEL_PATH" \
  --dtype "$DTYPE" \
  --backend "$BACKEND" \
  --trust_remote_code \
  --text_col sentence \
  --decision_col STANCE \
  --batch_size 50 \
  --max_new_tokens 80 \
  --temperature 0.0


# --- Archive: outputs + prompt/config/sbatch ---
PRED_CSV="${OUTBASE}_job${SLURM_JOB_ID}.csv"

FINAL_RUN_DIR="data/output/run_critic_job${SLURM_JOB_ID}"
mkdir -p "$FINAL_RUN_DIR"

cp "$PRED_CSV" "$FINAL_RUN_DIR/" || true
cp "src/run_critic_prompts.py" "$FINAL_RUN_DIR/prompts_used.py" || true
cp "src/run_critic_config.py" "$FINAL_RUN_DIR/config_used.py" || true
cp "$0" "$FINAL_RUN_DIR/sbatch_used.sbatch" || true

echo "Archived in: $FINAL_RUN_DIR"
