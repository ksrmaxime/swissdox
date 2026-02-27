# run4_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    text_col: str = "sentence",
    swiss_related_col: str = "SWISS_RELATED",
) -> pd.Series:
    """
    Send only rows where:
      - sentence is non-empty
      - SWISS_RELATED == "YES" (case-insensitive)
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")
    if swiss_related_col not in df.columns:
        # nothing to send if run1 not done / missing column
        return pd.Series([False] * len(df), index=df.index)

    s = df[text_col]
    nonempty = s.notna() & (s.astype(str).str.strip() != "")

    swiss = df[swiss_related_col].astype("string").str.strip().str.upper() == "YES"
    swiss = swiss.fillna(False)

    return nonempty & swiss