# run_all_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    text_col: str = "sentence",
    max_rows: int | None = None,
    sample_n: int | None = None,
    seed: int = 0,
    row_uid_min: int | None = None,
    row_uid_max: int | None = None,
) -> pd.Series:
    """
    Returns a boolean mask indicating which rows to send to the LLM.

    Filtering order:
      1) eligibility (non-empty sentence)
      2) optional row_uid range filter
      3) optional sampling / truncation:
         - if sample_n is set: random sample of eligible rows (reproducible with seed)
         - elif max_rows is set: first max_rows eligible rows (stable order)
         - else: all eligible rows
    """
    if text_col not in df.columns:
        raise KeyError(f"Missing text column: {text_col}")

    eligible = df[text_col].notna() & (df[text_col].astype(str).str.strip() != "")

    # Optional: restrict by row_uid range (only if column exists)
    if (row_uid_min is not None or row_uid_max is not None) and "row_uid" in df.columns:
        ru = pd.to_numeric(df["row_uid"], errors="coerce")
        if row_uid_min is not None:
            eligible &= ru >= row_uid_min
        if row_uid_max is not None:
            eligible &= ru <= row_uid_max

    idx = df.index[eligible]

    # Optional: random sample
    if sample_n is not None:
        if sample_n <= 0:
            return pd.Series(False, index=df.index)
        if len(idx) <= sample_n:
            return eligible
        sampled_idx = df.loc[idx].sample(n=sample_n, random_state=seed, replace=False).index
        mask = pd.Series(False, index=df.index)
        mask.loc[sampled_idx] = True
        return mask

    # Optional: take first N eligible rows
    if max_rows is not None:
        if max_rows <= 0:
            return pd.Series(False, index=df.index)
        take = idx[:max_rows]
        mask = pd.Series(False, index=df.index)
        mask.loc[take] = True
        return mask

    return eligible