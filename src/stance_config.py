# stance_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    sentence_col: str = "sentence",
) -> pd.Series:
    """
    Send all rows that have a non-empty sentence.
    Resume logic is handled by skip_if_already_filled in the pipeline.
    """
    if sentence_col not in df.columns:
        raise KeyError(f"Missing sentence column: {sentence_col}")

    return df[sentence_col].notna() & (df[sentence_col].astype(str).str.strip() != "")
