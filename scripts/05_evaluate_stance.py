# 05_evaluate_stance.py
"""
Évalue la colonne STANCE d'un fichier merged (sortie de sbatch_05_merge_stance.sh)
contre un fichier GOLD annoté manuellement.

Input  : --merged  fichier merged (csv ou parquet), doit contenir id_col, pred_stance_col,
                    text_col, justification_col
         --gold    fichier gold annoté à la main (csv, xlsx ou parquet), doit contenir
                    id_col et gold_stance_col

Output : --out_summary        CSV : accuracy (%) par type de réponse (STANCE) + ligne OVERALL
         --out_errors         CSV : lignes où STANCE prédite != STANCE gold, avec le texte
                               de la phrase, la justification du LLM, et les deux réponses
         --out_accuracy_tag   fichier texte contenant juste le tag d'accuracy (ex: "75p34"),
                               utilisé par le sbatch pour renommer le dossier de résultat

La comparaison ne porte que sur les lignes où STANCE est remplie des deux côtés
(jointure interne sur id_col, puis filtrage des valeurs manquantes/vides).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd


def read_table(path: str) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(p)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix == ".csv":
        return pd.read_csv(p)
    raise ValueError(f"Extension non supportée pour {path}: {suffix} (attendu: .csv, .xlsx, .parquet)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)

    ap.add_argument("--merged", required=True)
    ap.add_argument("--gold", required=True)

    ap.add_argument("--id_col", default="sentence_id")
    ap.add_argument("--pred_stance_col", default="STANCE")
    ap.add_argument("--gold_stance_col", default="STANCE")
    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--justification_col", default="justification")

    ap.add_argument("--out_summary", required=True)
    ap.add_argument("--out_errors", required=True)
    ap.add_argument("--out_accuracy_tag", required=True)

    args = ap.parse_args()

    merged = read_table(args.merged)
    gold = read_table(args.gold)

    required = [
        (merged, "merged", [args.id_col, args.pred_stance_col, args.text_col, args.justification_col]),
        (gold, "gold", [args.id_col, args.gold_stance_col]),
    ]
    for df, name, cols in required:
        missing = [c for c in cols if c not in df.columns]
        if missing:
            print(
                f"[ERROR] Colonne(s) {missing} absente(s) du fichier {name}. "
                f"Colonnes disponibles: {list(df.columns)}",
                file=sys.stderr,
            )
            return 1

    pred = merged[[args.id_col, args.pred_stance_col, args.text_col, args.justification_col]].rename(
        columns={
            args.pred_stance_col: "STANCE_pred",
            args.text_col: "sentence_text",
            args.justification_col: "llm_justification",
        }
    )
    gold = gold[[args.id_col, args.gold_stance_col]].rename(
        columns={args.gold_stance_col: "STANCE_gold"}
    )

    df = pred.merge(gold, on=args.id_col, how="inner")
    n_joined = len(df)

    def _filled(s: pd.Series) -> pd.Series:
        return s.notna() & (s.astype(str).str.strip() != "")

    df = df[_filled(df["STANCE_pred"]) & _filled(df["STANCE_gold"])].copy()
    n_compared = len(df)

    if n_compared == 0:
        print(
            f"[ERROR] Aucune ligne comparable: {n_joined:,} lignes jointes sur {args.id_col}, "
            "mais aucune n'a STANCE remplie des deux côtés.",
            file=sys.stderr,
        )
        return 1

    df["STANCE_pred_norm"] = df["STANCE_pred"].astype(str).str.strip().str.upper()
    df["STANCE_gold_norm"] = df["STANCE_gold"].astype(str).str.strip().str.upper()
    df["correct"] = df["STANCE_pred_norm"] == df["STANCE_gold_norm"]

    overall_acc = 100.0 * df["correct"].mean()

    # ── Accuracy par type de réponse (basé sur le label GOLD) ──────────────
    per_type = (
        df.groupby("STANCE_gold_norm")["correct"]
        .agg(n_gold="size", n_correct="sum")
        .reset_index()
        .rename(columns={"STANCE_gold_norm": "stance_type"})
        .sort_values("stance_type")
    )
    per_type["accuracy_pct"] = (100.0 * per_type["n_correct"] / per_type["n_gold"]).round(2)

    overall_row = pd.DataFrame([{
        "stance_type": "OVERALL",
        "n_gold": n_compared,
        "n_correct": int(df["correct"].sum()),
        "accuracy_pct": round(overall_acc, 2),
    }])
    per_type = pd.concat([per_type, overall_row], ignore_index=True)
    per_type.to_csv(args.out_summary, index=False, encoding="utf-8-sig")

    # ── Fichier d'erreurs ────────────────────────────────────────────────────
    errors = df.loc[
        ~df["correct"],
        [args.id_col, "sentence_text", "llm_justification", "STANCE_pred", "STANCE_gold"],
    ].copy()
    errors.to_csv(args.out_errors, index=False, encoding="utf-8-sig")

    # ── Tag pour le nom du dossier de résultat (ex: 75.34 -> "75p34") ───────
    tag = f"{overall_acc:.2f}".replace(".", "p")
    Path(args.out_accuracy_tag).write_text(tag)

    print(f"[eval] Lignes jointes sur {args.id_col}: {n_joined:,}")
    print(f"[eval] Lignes comparées (STANCE remplie des 2 côtés): {n_compared:,}")
    print(f"[eval] Accuracy globale: {overall_acc:.2f}%")
    print(f"[eval] Erreurs: {len(errors):,}")
    print(f"[eval] Saved → {args.out_summary}")
    print(f"[eval] Saved → {args.out_errors}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
