# run1_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    text_col: str = "sentence",
) -> pd.Series:
    """
    Send all rows that have a non-empty sentence.
    (Resume logic is handled by skip_if_already_filled in the pipeline.)
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    s = df[text_col]
    # Keep only non-null and non-empty after strip
    mask = s.notna() & (s.astype(str).str.strip() != "")
    return mask