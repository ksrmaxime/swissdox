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

set -euo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO

cd "$WORKDIR"
source .venv/bin/activate

mkdir -p logs

echo "=== SLURM ==="
echo "JOBID=${SLURM_JOB_ID:-<unset>} HOST=$(hostname) DATE=$(date -Is)"

# ── Input: output of sbatch_run_article.sh ────────────────────────────────────
# Remplacer JOBID_ARTICLE par le job ID du run article (ex: 12345678)
# ou pointer directement vers le fichier checkpoint si le run n'est pas terminé
ARTICLE_JOBID=${1:-""}   # passer en argument: sbatch sbatch_sentence_cutting.sh 12345678

if [ -n "$ARTICLE_JOBID" ]; then
    INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox_article_with_s_t_job${ARTICLE_JOBID}.csv"
else
    # fallback: cherche le fichier checkpoint si le job article tourne encore
    INPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/processed/swissdox_article_with_s_t_checkpoint.parquet"
fi

OUTPUT="/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO/data/critic_base.csv"

echo "INPUT  = $INPUT"
echo "OUTPUT = $OUTPUT"

python scripts/03_extract_sentences.py \
    --input  "$INPUT" \
    --output "$OUTPUT"

echo "Done. critic_base.csv ready at: $OUTPUT"
echo "Rows: $(wc -l < "$OUTPUT")"
