from __future__ import annotations

import argparse

from swissdox.evaluation.plots import run_all_plots


def main():
    parser = argparse.ArgumentParser(description="Generate comparison plots for 3 techniques (A/B/C).")
    parser.add_argument("--a", required=True, help="Path to technique A file (csv/parquet) [Apertus/LLM article-level]")
    parser.add_argument("--b", required=True, help="Path to technique B file (csv/parquet) [GPT sentence-level]")
    parser.add_argument("--c", required=True, help="Path to technique C file (csv/parquet) [Embed+RoBERTa sentence-level]")
    parser.add_argument("--outdir", default="outputs/plots", help="Directory to save PNG plots (created if missing)")
    parser.add_argument("--no-show", action="store_true", help="Do not show plots (save only)")
    args = parser.parse_args()

    # These mappings must match the ones used in your compare step.
    # Keep them centralized (same label harmonization everywhere).
    NAME_A = "APERTUS"
    NAME_B = "GPT"
    NAME_C = "Embed+RoBERTa"

    KEY_COL = "article_id"  # standardized key

    RENAME_A = {
        "id": "article_id",
        "pubtime": "date_col",
        "topic_llm": "topic_col",
        "sentiment_public_admin_llm": "sentiment_col",
        "populism_llm": "populism_col",
    }
    RENAME_B = {
        "article_id_x": "article_id",
        "pubtime": "date_col",
        "t": "topic_col",
        "sp": "sentiment_col",
        "p": "populism_col",
    }
    RENAME_C = {
        "id": "article_id",
        "pubtime": "date_col",
        "main_theme": "topic_col",
        "sentiment_label": "sentiment_col",
    }

    TOPIC_MAP = {"Others": "Other"}
    SENTIMENT_MAP = {
        "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
        "LABEL_0": "NEGATIVE", "LABEL_1": "NEUTRAL", "LABEL_2": "POSITIVE",
        -1.0: "NEGATIVE", 0.0: "NEUTRAL", 1.0: "POSITIVE",
        "-1.0": "NEGATIVE", "0.0": "NEUTRAL", "1.0": "POSITIVE",
    }
    POPULISM_MAP = {
        "0.0": "NO", "1.0": "YES",
        "0": "NO", "1": "YES",
        0.0: "NO", 1.0: "YES",
        0: "NO", 1: "YES",
        "NO": "NO", "YES": "YES",
    }

    run_all_plots(
        path_a=args.a,
        path_b=args.b,
        path_c=args.c,
        name_a=NAME_A,
        name_b=NAME_B,
        name_c=NAME_C,
        rename_a=RENAME_A,
        rename_b=RENAME_B,
        rename_c=RENAME_C,
        topic_map=TOPIC_MAP,
        sentiment_map=SENTIMENT_MAP,
        populism_map=POPULISM_MAP,
        outdir=args.outdir,
        show=not args.no_show,
    )


if __name__ == "__main__":
    main()
