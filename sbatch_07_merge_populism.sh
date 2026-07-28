#!/bin/bash -l
#SBATCH --job-name=merge_populism
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_populism_%j.out
#SBATCH --error=logs/merge_populism_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch --dependency=afterok:<PREV_JOB_ID> sbatch_07_merge_populism.sh <PREV_JOB_ID>
# Exemple: sbatch --dependency=afterok:12345678 sbatch_07_merge_populism.sh 12345678
# Le PREV_JOB_ID est l'ID du job array de sbatch_06_classify_populism.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/07_merge_populism_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_06_classify_populism.sh) ───────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_07_merge_populism.sh <PREV_JOB_ID>"
    exit 1
fi

INBASE="$WORKDIR/data/output/06_classify_populism_job${PREV_JOB_ID}/populism"
MERGED_CSV="$OUTDIR/populism_merged.csv"
MERGED_PARQUET="$OUTDIR/populism_merged.parquet"

echo "=== MERGE populism array job ${PREV_JOB_ID} ==="
echo "DATE=$(date -Is)"

export INBASE MERGED_CSV MERGED_PARQUET

python3 - <<'PYEOF'
import sys, os, glob
import pandas as pd

inbase         = os.environ["INBASE"]
merged_csv     = os.environ["MERGED_CSV"]
merged_parquet = os.environ["MERGED_PARQUET"]

pattern = f"{inbase}_task*.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{inbase}_task*.csv"
    files = sorted(glob.glob(pattern))
    if not files:
        print(f"[ERROR] Aucun fichier trouvé avec le pattern: {pattern}", file=sys.stderr)
        sys.exit(1)
    dfs = [pd.read_csv(f) for f in files]
else:
    dfs = [pd.read_parquet(f) for f in files]

print(f"[merge] {len(files)} fichiers trouvés:")
for f in files:
    df_f = dfs[files.index(f)]
    print(f"  {f}  →  {len(df_f):,} lignes")

merged = pd.concat(dfs, ignore_index=True)
merged = merged.sort_values("sentence_id").reset_index(drop=True)

merged.to_parquet(merged_parquet, index=False)
merged.to_csv(merged_csv, index=False, encoding="utf-8-sig")

print(f"\n[merge] Total: {len(merged):,} lignes")
print(f"[merge] Saved → {merged_parquet}")
print(f"[merge] Saved → {merged_csv}")

filled = merged["POPULISM"].notna().sum()
print(f"[merge] Lignes avec POPULISM remplie: {filled:,} / {len(merged):,}")
PYEOF

echo "Merge terminé."
