#!/bin/bash -l
#SBATCH --job-name=merge_criticism
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/merge_criticism_%j.out
#SBATCH --error=logs/merge_criticism_%j.err
#SBATCH --mail-user=maxime.kaiser@unil.ch
#SBATCH --mail-type=END,FAIL

# Usage: sbatch --dependency=afterok:<PREV_JOB_ID> sbatch_05_merge_stance.sh <PREV_JOB_ID>
# Exemple: sbatch --dependency=afterok:12345678 sbatch_05_merge_stance.sh 12345678
# Le PREV_JOB_ID est l'ID du job array de sbatch_04_classify_stance.sh

set -eo pipefail

module purge
dcsrsoft use 20241118
module load python/3.12.1

set -u
WORKDIR=/work/FAC/FDCA/IDHEAP/mhinterl/parp/SWISSDOX_REPO
cd "$WORKDIR"
source .venv/bin/activate

# ── OUTPUT: dossier dédié à ce run (nom de l'étape + son propre job id) ────
OUTDIR="$WORKDIR/data/output/05_merge_stance_job${SLURM_JOB_ID}"
mkdir -p logs "$OUTDIR"

# ── INPUT: job id du run précédent (sbatch_04_classify_stance.sh) ─────────
PREV_JOB_ID="${1:-}"
if [ -z "$PREV_JOB_ID" ]; then
    echo "[ERROR] PREV_JOB_ID manquant. Édite la ligne PREV_JOB_ID dans ce script, ou: sbatch sbatch_05_merge_stance.sh <PREV_JOB_ID>"
    exit 1
fi

INBASE="$WORKDIR/data/output/04_classify_stance_job${PREV_JOB_ID}/criticism_stance"
MERGED_CSV="$OUTDIR/criticism_stance_merged.csv"
MERGED_PARQUET="$OUTDIR/criticism_stance_merged.parquet"

# ── ÉVALUATION (optionnelle) vs un fichier GOLD annoté à la main ──────────
# Renseigne ici le chemin vers ton fichier gold (CSV, XLSX ou Parquet) pour activer
# l'évaluation. Il doit contenir au minimum GOLD_ID_COL et GOLD_STANCE_COL.
# Laisse vide pour ne faire que le merge, comme avant.
GOLD_DATA="$WORKDIR/data/criticism_stance_GOLD.csv"
# GOLD_DATA="$WORKDIR/data/external/mon_fichier_gold.csv"
GOLD_ID_COL="${GOLD_ID_COL:-sentence_id}"
GOLD_STANCE_COL="${GOLD_STANCE_COL:-STANCE}"

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

# ── ÉVALUATION vs GOLD_DATA (si renseigné) ─────────────────────────────────
if [ -n "$GOLD_DATA" ]; then
    if [ ! -f "$GOLD_DATA" ]; then
        echo "[WARN] GOLD_DATA=$GOLD_DATA introuvable, évaluation ignorée."
    else
        echo "=== EVALUATION vs GOLD_DATA=$GOLD_DATA ==="

        ACC_SUMMARY_CSV="$OUTDIR/criticism_stance_accuracy_by_type.csv"
        ACC_ERRORS_CSV="$OUTDIR/criticism_stance_errors.csv"
        ACC_TAG_FILE="$OUTDIR/.accuracy_tag"

        # "|| true" : un échec de l'évaluation (ex: colonne gold manquante) ne doit
        # pas faire échouer tout le job, le merge lui a déjà réussi et été sauvegardé.
        python3 scripts/05_evaluate_stance.py \
            --merged "$MERGED_PARQUET" \
            --gold "$GOLD_DATA" \
            --id_col "$GOLD_ID_COL" \
            --gold_stance_col "$GOLD_STANCE_COL" \
            --out_summary "$ACC_SUMMARY_CSV" \
            --out_errors "$ACC_ERRORS_CSV" \
            --out_accuracy_tag "$ACC_TAG_FILE" || true

        if [ -f "$ACC_TAG_FILE" ]; then
            ACC_TAG="$(cat "$ACC_TAG_FILE")"
            rm -f "$ACC_TAG_FILE"

            # Renomme le dossier de résultat pour y intégrer l'accuracy,
            # ex: 05_merge_stance_job64297021 → 05_merge_stance_75p34_job64297021
            NEWDIR="$WORKDIR/data/output/05_merge_stance_${ACC_TAG}_job${SLURM_JOB_ID}"
            mv "$OUTDIR" "$NEWDIR"
            OUTDIR="$NEWDIR"
            echo "[eval] Dossier de résultat renommé → $OUTDIR"
        else
            echo "[WARN] Évaluation échouée, dossier de résultat non renommé."
        fi
    fi
fi

echo "Merge terminé. Résultats dans: $OUTDIR"
