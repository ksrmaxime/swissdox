#!/bin/bash -l
#SBATCH --job-name=step1b_merge_years
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=06:00:00
#SBATCH --output=logs/step1b_merge_years_%j.out
#SBATCH --error=logs/step1b_merge_years_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# ═══════════════════════════════════════════════════════════════════════════
# Step 1b — regroupe les années téléchargées par sbatch_step1_download_yearly.sh
# en un seul fichier, au même format qu'un step 1 classique :
#     data/output/step1_download_job<ID de CE job>/swissdox_articles_lead.parquet
#
# Soumis automatiquement à la fin de la chaîne. Lancement manuel possible :
#     sbatch sbatch_step1b_merge_years.sh <dossier de la chaîne>
#
# Ensuite : sbatch sbatch_step2_sentence_cutting.sh <ID de CE job>
# ═══════════════════════════════════════════════════════════════════════════

set -eo pipefail
export PYTHONUNBUFFERED=1

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO"
cd "${REPO_DIR}"
mkdir -p logs

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
source .venv/bin/activate

CHAIN_DIR="${1:-}"
if [ -z "$CHAIN_DIR" ] || [ ! -d "$CHAIN_DIR" ]; then
    echo "[ERROR] Usage: sbatch sbatch_step1b_merge_years.sh <dossier de la chaîne>"
    exit 1
fi
OUTDIR="${REPO_DIR}/data/output/step1_download_job${SLURM_JOB_ID}"
mkdir -p "$OUTDIR"

echo "=== Step 1b : regroupement des années ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"
echo "CHAIN_DIR=${CHAIN_DIR}"
echo "OUTDIR=${OUTDIR}"

python scripts/step1b_merge_years.py --chain_dir "$CHAIN_DIR" --outdir "$OUTDIR"

echo
echo "Regroupement terminé. Pour le sentence cutting :"
echo "    sbatch sbatch_step2_sentence_cutting.sh ${SLURM_JOB_ID}"
