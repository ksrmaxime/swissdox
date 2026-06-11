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

# Usage: sbatch --dependency=afterok:<ARRAY_JOB_ID> sbatch_07_merge_populism.sh <ARRAY_JOB_ID>
# Exemple: sbatch --dependency=afterok:12345678 sbatch_07_merge_populism.sh 12345678

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
    echo "[ERROR] Passer le ARRAY_JOB_ID en argument: sbatch sbatch_07_merge_populism.sh <ARRAY_JOB_ID>"
    exit 1
fi

OUTDIR="$WORKDIR/data/processed"
OUTBASE="$OUTDIR/populism"
MERGED_CSV="$OUTDIR/populism_merged_job${ARRAY_JOB_ID}.csv"
MERGED_PARQUET="$OUTDIR/populism_merged_job${ARRAY_JOB_ID}.parquet"

echo "=== MERGE populism array job ${ARRAY_JOB_ID} ==="
echo "DATE=$(date -Is)"

export OUTDIR OUTBASE ARRAY_JOB_ID MERGED_CSV MERGED_PARQUET

python3 - <<'PYEOF'
import sys, os, glob
import pandas as pd

outdir         = os.environ["OUTDIR"]
outbase        = os.environ["OUTBASE"]
job_id         = os.environ["ARRAY_JOB_ID"]
merged_csv     = os.environ["MERGED_CSV"]
merged_parquet = os.environ["MERGED_PARQUET"]

pattern = f"{outbase}_task*_job{job_id}.parquet"
files = sorted(glob.glob(pattern))

if not files:
    pattern = f"{outbase}_task*_job{job_id}.csv"
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
