#!/bin/bash
#SBATCH --job-name=swissdox_pr
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=12
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --output=logs/pr_%j.out
#SBATCH --error=logs/pr_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail
module purge
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
SCRATCHDIR=/scratch/mkaiser3
OUTDIR=$WORKDIR/data/processed/swissdox/pr

cd "$WORKDIR"
source .venv/bin/activate
mkdir -p logs "$OUTDIR" "$SCRATCHDIR/swissdox/pr"

echo "JOBID=${SLURM_JOB_ID:-} HOST=$(hostname) DATE=$(date -Is)"
nvidia-smi -L || true

python scripts/run_pr_apertus.py \
  --workdir "$WORKDIR" \
  --scratchdir "$SCRATCHDIR" \
  --input "data/processed/swissdox/swissdox_sentences.parquet" \
  --outdir "$OUTDIR" \
  --model-path /reference/LLM/swiss-ai/Apertus-8B-Instruct-2509 \
  --dtype bf16 \
  --items-per-prompt 60 \
  --prompts-per-batch 8 \
  --max-tokens 250 \
  --temperature 0.0 \
  --resume

nvidia-smi || true
echo "Done."

