#!/bin/bash
#SBATCH --job-name=swissdox_pipe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=logs/swissdox_%j.out
#SBATCH --error=logs/swissdox_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

set -euo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO"
cd "${REPO_DIR}"

mkdir -p logs data/processed/swissdox

module purge
module load python/3.12.1

source .venv/bin/activate

python --version
which python

# Optionnel: utiliser /scratch pour temp si tu fais du gros I/O
export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

python scripts/run_swissdox_pipeline.py \
  --start 2025-01-01 \
  --end 2025-12-31 \
  --max-results 20000 \
  --outdir data/processed/swissdox

echo "Job finished."
