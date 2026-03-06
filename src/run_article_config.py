# run_article_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    title_col: str = "title",
    lead_col: str = "lead",
) -> pd.Series:
    """
    Send all rows that have a non-empty title or a non-empty lead.
    Resume logic is handled by skip_if_already_filled in the pipeline.
    """
    if title_col not in df.columns:
        raise KeyError(f"Missing title column: {title_col}")
    if lead_col not in df.columns:
        raise KeyError(f"Missing lead column: {lead_col}")

    title_ok = df[title_col].notna() & (df[title_col].astype(str).str.strip() != "")
    lead_ok = df[lead_col].notna() & (df[lead_col].astype(str).str.strip() != "")

    return title_ok | lead_ok