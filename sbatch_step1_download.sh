#!/bin/bash -l
#SBATCH --job-name=step1_download
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/step1_download_%j.out
#SBATCH --error=logs/step1_download_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

set -eo pipefail

# Logs écrits immédiatement dans le .out (sinon Python les garde en mémoire)
export PYTHONUNBUFFERED=1

REPO_DIR="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO"
cd "${REPO_DIR}"

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="${REPO_DIR}/data/output/step1_download_job${SLURM_JOB_ID}"
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

# ── Reprise optionnelle : sbatch sbatch_step1_download.sh <QUERY_ID> ─────
# Si un job précédent a échoué APRÈS avoir soumis la requête, son .out contient
# "[Swissdox] queryId=XXXX". Passer cet ID reprend la même requête sans la renvoyer.
QUERY_ID="${1:-}"
QUERY_ID_ARGS=()
if [ -n "$QUERY_ID" ]; then
    echo "[INFO] Reprise de la requête Swissdox existante: $QUERY_ID"
    QUERY_ID_ARGS=(--query-id "$QUERY_ID")
fi

python scripts/step1_download.py \
  --start 2000-01-01 \
  --end 2015-12-31 \
  --max-results 1000000 \
  --max-wait-hours 22 \
  --outdir "${OUTDIR}" \
  ${QUERY_ID_ARGS[@]+"${QUERY_ID_ARGS[@]}"}

echo "Job finished. Output: ${OUTDIR}"