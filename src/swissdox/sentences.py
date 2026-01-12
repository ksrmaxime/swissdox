from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Sequence

import pandas as pd
import re


@dataclass(frozen=True)
class SentenceSplitConfig:
    content_col: str = "content"
    sentence_col: str = "sentence"
    matched_col: str = "matched_keywords"
    sentence_id_col: str = "sentence_id"
    split_regex: str = r"(?<=[.!?])\s+"
    ignore_case: bool = True


def build_kw_pattern(keywords: Sequence[str], *, ignore_case: bool = True) -> re.Pattern:
    """Build a regex that matches any keyword.
    - Acronyms (<=4 uppercase) are matched as whole words (\bABC\b)
    - Others are matched as escaped substrings
    """
    if not keywords:
        raise ValueError("keywords must be non-empty")

    patterns: List[str] = []
    for k in keywords:
        if isinstance(k, str) and k.isupper() and len(k) <= 4:
            patterns.append(rf"\b{re.escape(k)}\b")
        else:
            patterns.append(re.escape(str(k)))

    flags = re.IGNORECASE if ignore_case else 0
    return re.compile("|".join(patterns), flags=flags)


def split_and_filter_sentences(
    df_articles: pd.DataFrame,
    keywords: Sequence[str],
    *,
    keep_cols: Sequence[str],
    cfg: SentenceSplitConfig = SentenceSplitConfig(),
) -> pd.DataFrame:
    """Split articles into sentences, keep only sentences containing any keyword.

    Returns a dataframe with:
    - sentence_id
    - matched_keywords (unique, comma-separated)
    - sentence
    - selected propagated article columns (keep_cols)
    """
    if cfg.content_col not in df_articles.columns:
        raise ValueError(f"Missing '{cfg.content_col}' column in df_articles")

    kw_pattern = build_kw_pattern(keywords, ignore_case=cfg.ignore_case)

    tmp = df_articles.copy()
    tmp[cfg.sentence_col] = tmp[cfg.content_col].fillna("").astype(str)

    # Split -> explode
    tmp[cfg.sentence_col] = tmp[cfg.sentence_col].str.split(cfg.split_regex, regex=True)
    tmp = tmp.explode(cfg.sentence_col, ignore_index=True)

    # Clean + drop empty
    tmp[cfg.sentence_col] = tmp[cfg.sentence_col].astype(str).str.strip()
    tmp = tmp[tmp[cfg.sentence_col].ne("")].copy()

    # Filter by keyword presence
    mask = tmp[cfg.sentence_col].str.contains(kw_pattern, na=False)
    tmp = tmp[mask].copy()

    # Matched keywords (unique)
    tmp[cfg.matched_col] = tmp[cfg.sentence_col].str.findall(kw_pattern).apply(
        lambda lst: ", ".join(sorted({x.strip() for x in lst if isinstance(x, str) and x.strip()}))
    )

    # sentence_id
    tmp.insert(0, cfg.sentence_id_col, range(1, len(tmp) + 1))

    # Keep only selected columns
    missing = [c for c in keep_cols if c not in tmp.columns]
    if missing:
        raise ValueError(f"keep_cols contains missing columns: {missing}")

    out_cols = [cfg.sentence_id_col, *keep_cols]
    # ensure sentence + matched are included even if not in keep_cols
    if cfg.matched_col not in out_cols:
        out_cols.append(cfg.matched_col)
    if cfg.sentence_col not in out_cols:
        out_cols.append(cfg.sentence_col)

    return tmp[out_cols].copy()
