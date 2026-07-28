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

# Usage: sbatch sbatch_08_analyze.sh <PREV_JOB_ID>
# Exemple: sbatch sbatch_08_analyze.sh 60015520
# Le PREV_JOB_ID est l'ID du job de sbatch_07_merge_populism.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/08_analyze_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_07_merge_populism.sh) ──────────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_08_analyze.sh <PREV_JOB_ID>"
    exit 1
fi

INPUT="$WORKDIR/data/output/07_merge_populism_job${PREV_JOB_ID}/populism_merged.parquet"

# Fallback sur CSV si parquet absent
if [ ! -f "$INPUT" ]; then
    INPUT="$WORKDIR/data/output/07_merge_populism_job${PREV_JOB_ID}/populism_merged.csv"
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
