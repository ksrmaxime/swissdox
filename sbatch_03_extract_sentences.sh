#!/bin/bash -l
#SBATCH --job-name=sentence_cutting
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=01:00:00
#SBATCH --output=logs/sentence_cutting_%j.out
#SBATCH --error=logs/sentence_cutting_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch sbatch_03_extract_sentences.sh <PREV_JOB_ID>
# Exemple: sbatch sbatch_03_extract_sentences.sh 60015520
# Le PREV_JOB_ID est l'ID du job de sbatch_02_classify_articles.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO

cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/03_extract_sentences_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"

# ── INPUT: job id du run précédent (sbatch_02_classify_articles.sh) ───────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_03_extract_sentences.sh <PREV_JOB_ID>"
    exit 1
fi
INPUT="$WORKDIR/data/output/02_classify_articles_job${PREV_JOB_ID}/articles_classified.csv"
OUTPUT_BASE="$OUTDIR/critic_base"

echo "INPUT       = $INPUT"
echo "OUTPUT_BASE = $OUTPUT_BASE"

python scripts/03_extract_sentences.py \
    --input  "$INPUT" \
    --output_base "$OUTPUT_BASE"

OUTPUT="${OUTPUT_BASE}.csv"
echo "Done. critic_base ready at: $OUTPUT"
echo "Rows: $(wc -l < "$OUTPUT")"
