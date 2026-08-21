#!/bin/bash -l
#SBATCH --job-name=swissdox_pipe
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/swissdox_%j.out
#SBATCH --error=logs/swissdox_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -eo pipefail

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO"
cd "${REPO_DIR}"

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="${REPO_DIR}/data/output/01_download_job${SLURM_JOB_ID}"
mkdir -p logs "${OUTDIR}"

# module/dcsrsoft scripts reference unset vars internally, so -u must stay off until after they load
module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

python --version
which python

# Optionnel: utiliser /scratch pour temp si tu fais du gros I/O
export TMPDIR="/scratch/mkaiser3/tmp_${SLURM_JOB_ID}"
mkdir -p "${TMPDIR}"

python scripts/01_download.py \
  --start 2015-01-01 \
  --end 2025-12-31 \
  --max-results 100000 \
  --max-wait-hours 22 \
  --outdir "${OUTDIR}"

echo "Job finished. Output: ${OUTDIR}"