#!/bin/bash -l
#SBATCH --job-name=step3b_stance_merge
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/step3b_stance_merge_%j.out
#SBATCH --error=logs/step3b_stance_merge_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch --dependency=afterok:<PREV_JOB_ID> sbatch_step3b_stance_merge.sh <PREV_JOB_ID>
# Exemple: sbatch --dependency=afterok:12345678 sbatch_step3b_stance_merge.sh 12345678
# Le PREV_JOB_ID est l'ID du job array de sbatch_step3a_stance_array.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/step3b_stance_merge_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_step3a_stance_array.sh) ─────────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_step3b_stance_merge.sh <PREV_JOB_ID>"
    exit 1
fi

INBASE="$WORKDIR/data/output/step3a_stance_array_job${PREV_JOB_ID}/criticism_stance"
MERGED_CSV="$OUTDIR/criticism_stance_merged.csv"
MERGED_PARQUET="$OUTDIR/criticism_stance_merged.parquet"

echo "=== MERGE criticism array job ${PREV_JOB_ID} ==="
echo "DATE=$(date -Is)"

export INBASE MERGED_CSV MERGED_PARQUET

python3 - <<'PYEOF'
import sys, os, glob
import pandas as pd

inbase         = os.environ["INBASE"]
merged_csv     = os.environ["MERGED_CSV"]
merged_parquet = os.environ["MERGED_PARQUET"]

pattern = f"{inbase}_task*.parquet"
# "_task*" matche aussi les checkpoints intermédiaires ("_task0_checkpoint.parquet"),
# qui contiennent les mêmes lignes que le fichier final "_task0.parquet" une fois le
# job terminé : on les exclut explicitement pour ne pas compter chaque ligne deux fois.
files = sorted(f for f in glob.glob(pattern) if "checkpoint" not in os.path.basename(f))

if not files:
    # fallback sur CSV si parquet absent
    pattern = f"{inbase}_task*.csv"
    files = sorted(f for f in glob.glob(pattern) if "checkpoint" not in os.path.basename(f))
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

filled = merged["STANCE"].notna().sum()
print(f"[merge] Lignes avec STANCE remplie: {filled:,} / {len(merged):,}")
PYEOF

echo "Merge terminé. Résultats dans: $OUTDIR"
