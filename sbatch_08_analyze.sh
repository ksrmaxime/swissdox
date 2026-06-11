#!/bin/bash -l
#SBATCH --job-name=stance_analysis
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/analysis_%j.out
#SBATCH --error=logs/analysis_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch sbatch_08_analyze.sh <POPULISM_JOB_ID>
# Exemple: sbatch sbatch_08_analyze.sh 60015520

set -euo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

ARRAY_JOB_ID=${1:-""}
if [ -z "$ARRAY_JOB_ID" ]; then
    echo "[ERROR] Passer le POPULISM_JOB_ID: sbatch sbatch_08_analyze.sh <POPULISM_JOB_ID>"
    exit 1
fi

INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/populism_merged_job${ARRAY_JOB_ID}.parquet"
OUTDIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/output/analysis_job${ARRAY_JOB_ID}"

# Fallback sur CSV si parquet absent
if [ ! -f "$INPUT" ]; then
    INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/populism_merged_job${ARRAY_JOB_ID}.csv"
fi

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "INPUT  = $INPUT"
echo "OUTDIR = $OUTDIR"

python scripts/08_analyze.py \
    --input   "$INPUT" \
    --outdir  "$OUTDIR" \
    --top_journals 25

echo "Done. Figures and stats in: $OUTDIR"
