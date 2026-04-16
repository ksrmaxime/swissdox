# run_populism_config.py
from __future__ import annotations

import pandas as pd


def build_sentences_to_send_mask(
    df: pd.DataFrame,
    *,
    sentence_col: str = "sentence",
    stance_col: str = "STANCE",
) -> pd.Series:
    """
    Send only rows that:
      - have a non-empty sentence
      - have been classified as STANCE == "CRITIC" in the upstream critic pipeline
    Resume logic is handled by skip_if_already_filled in the pipeline.
    """
    if sentence_col not in df.columns:
        raise KeyError(f"Missing sentence column: {sentence_col!r}")
    if stance_col not in df.columns:
        raise KeyError(
            f"Missing stance column: {stance_col!r}. "
            "Make sure the input file is the output of run_critic_pipeline."
        )

    has_sentence = df[sentence_col].notna() & (df[sentence_col].astype(str).str.strip() != "")
    is_critic = df[stance_col].astype(str).str.strip().str.upper() == "CRITIC"

    return has_sentence & is_critic
