from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from swissdox.evaluation.compare import (
    ComparisonConfig,
    agreement_report,
    load_and_align_three,
    topic_confusion_pair,
)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="CSV path for A (Apertus / LLM)")
    ap.add_argument("--b", required=True, help="CSV path for B (GPT)")
    ap.add_argument("--c", required=True, help="CSV path for C (Embed+RoBERTa)")
    ap.add_argument("--out-dir", default="data/processed/comparison")
    ap.add_argument("--filters", action="store_true", help="Enable pairwise filters (exclude only if BOTH excluded)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Rename hooks (your current ones)
    RENAME_A = {"id":"article_id","pubtime":"date_col","topic_llm":"topic_col","sentiment_public_admin_llm":"sentiment_col","populism_llm":"populism_col"}
    RENAME_B = {"article_id_x":"article_id","pubtime":"date_col","t":"topic_col","sp":"sentiment_col","p":"populism_col"}
    RENAME_C = {"id":"article_id","pubtime":"date_col","main_theme":"topic_col","sentiment_label":"sentiment_col"}

    cfg = ComparisonConfig(use_filters=bool(args.filters), is_sentence_level_b=True, is_sentence_level_c=True)

    df_all = load_and_align_three(
        Path(args.a), Path(args.b), Path(args.c),
        cfg=cfg,
        rename_a=RENAME_A,
        rename_b=RENAME_B,
        rename_c=RENAME_C,
    )

    # Save aligned dataset (useful for later)
    aligned_path = out_dir / "aligned_three_models.parquet"
    df_all.to_parquet(aligned_path, index=False)

    # Agreement tables
    rows = []
    for col_base, title in [("topic_col","TOPIC"), ("sentiment_col","SENTIMENT"), ("populism_col","POPULISM")]:
        rows.append({"dimension": title, **agreement_report(df_all, cfg=cfg, col_base=col_base, s1="A", s2="B")})
        rows.append({"dimension": title, **agreement_report(df_all, cfg=cfg, col_base=col_base, s1="A", s2="C")})
        rows.append({"dimension": title, **agreement_report(df_all, cfg=cfg, col_base=col_base, s1="B", s2="C")})

    rep = pd.DataFrame(rows)
    rep_path = out_dir / "agreement_report.csv"
    rep.to_csv(rep_path, index=False)

    # Topic confusion matrices
    for (s1, s2, name) in [("A","B","A_vs_B"), ("A","C","A_vs_C"), ("B","C","B_vs_C")]:
        ct = topic_confusion_pair(df_all, cfg=cfg, s1=s1, s2=s2, normalize=None)
        ct_row = topic_confusion_pair(df_all, cfg=cfg, s1=s1, s2=s2, normalize="row")

        if ct is not None:
            ct.to_csv(out_dir / f"topic_confusion_counts_{name}.csv")
        if ct_row is not None:
            (ct_row * 100).round(3).to_csv(out_dir / f"topic_confusion_rowpct_{name}.csv")

    print("✅ Aligned rows:", len(df_all))
    print("✅ Saved:", aligned_path)
    print("✅ Saved:", rep_path)
    print("✅ Saved confusion matrices in:", out_dir)


if __name__ == "__main__":
    main()
