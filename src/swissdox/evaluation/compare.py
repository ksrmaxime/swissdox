from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Set, Tuple

import numpy as np
import pandas as pd


# -----------------------------
# Canon label space
# -----------------------------
CANON_TOPICS = [
    "Foreign Affairs","Culture","Health","Social","Justice","Migration","Defence","Sport","Finance",
    "Economy","Education","Research","Environment","Transports","Energy","Communication","Other"
]
CANON_SENTIMENT = ["POSITIVE","NEGATIVE","NEUTRAL"]
CANON_POPULISM = ["YES","NO"]


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ComparisonConfig:
    key_col: str = "article_id"

    # sentence-level inputs that need aggregation
    is_sentence_level_b: bool = True
    is_sentence_level_c: bool = True

    # filter logic
    use_filters: bool = False
    filters: Dict[str, Dict[str, Set[str]]] = None  # e.g. {"topic_col":{"exclude":{"Other"}}}

    # mapping hooks
    topic_map: Dict[Any, Any] = None
    sentiment_map: Dict[Any, Any] = None
    populism_map: Dict[Any, Any] = None

    def __post_init__(self):
        # dataclass(frozen=True) trick: defaults via object.__setattr__
        if self.filters is None:
            object.__setattr__(self, "filters", {
                "topic_col": {"exclude": {"Other"}},
                "sentiment_col": {"exclude": {"NEUTRAL"}},
                "populism_col": {"exclude": {"NO"}},
            })
        if self.topic_map is None:
            object.__setattr__(self, "topic_map", {"Others": "Other"})
        if self.sentiment_map is None:
            object.__setattr__(self, "sentiment_map", {
                "negative": "NEGATIVE", "neutral": "NEUTRAL", "positive": "POSITIVE",
                "-1.0": "NEGATIVE", "0.0": "NEUTRAL", "1.0": "POSITIVE",
                -1.0: "NEGATIVE", 0.0: "NEUTRAL", 1.0: "POSITIVE",
            })
        if self.populism_map is None:
            object.__setattr__(self, "populism_map", {
                "0.0": "NO", "1.0": "YES",
                "0": "NO", "1": "YES",
            })


# -----------------------------
# Core utilities
# -----------------------------
def _normalize_string(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return np.nan if x == "" else x


def _apply_mapping(series: pd.Series, mapping: dict):
    return series.map(lambda v: mapping.get(v, mapping.get(_normalize_string(v), v)))


def cohen_kappa(a: pd.Series, b: pd.Series) -> float:
    mask = a.notna() & b.notna()
    a = a[mask].astype(str)
    b = b[mask].astype(str)
    if len(a) == 0:
        return np.nan
    labels = sorted(set(a.unique()).union(set(b.unique())))
    cm = pd.crosstab(a, b, dropna=False).reindex(index=labels, columns=labels, fill_value=0)
    n = cm.values.sum()
    if n == 0:
        return np.nan
    po = np.trace(cm.values) / n
    pe = (cm.sum(axis=1).values @ cm.sum(axis=0).values) / (n * n)
    if pe == 1:
        return 1.0
    return float((po - pe) / (1 - pe))


def standardize_df(
    df: pd.DataFrame,
    *,
    rename_dict: dict,
    cfg: ComparisonConfig,
    technique_name: str,
    dedupe: bool,
) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=rename_dict)

    if cfg.key_col not in df.columns:
        raise ValueError(f"[{technique_name}] key_col='{cfg.key_col}' not found after renaming. Columns: {list(df.columns)}")

    for col in ["date_col","topic_col","sentiment_col","populism_col"]:
        if col in df.columns:
            df[col] = df[col].map(_normalize_string)

    if "date_col" in df.columns:
        df["date_col"] = pd.to_datetime(df["date_col"], errors="coerce")

    if "topic_col" in df.columns:
        df["topic_col"] = _apply_mapping(df["topic_col"], cfg.topic_map).map(_normalize_string)
    if "sentiment_col" in df.columns:
        df["sentiment_col"] = _apply_mapping(df["sentiment_col"], cfg.sentiment_map).map(_normalize_string)
    if "populism_col" in df.columns:
        s_num = pd.to_numeric(df["populism_col"], errors="coerce")
        df["populism_col"] = np.where(s_num.notna(), s_num, df["populism_col"])
        df["populism_col"] = _apply_mapping(pd.Series(df["populism_col"]), cfg.populism_map).map(_normalize_string)

    keep_cols = [cfg.key_col] + [c for c in ["date_col","topic_col","sentiment_col","populism_col"] if c in df.columns]
    df = df[keep_cols]

    if dedupe:
        df = df.drop_duplicates(subset=[cfg.key_col], keep="first")

    return df


def aggregate_sentence_to_article(df_sent: pd.DataFrame, *, cfg: ComparisonConfig) -> pd.DataFrame:
    df = df_sent.copy()
    if cfg.key_col not in df.columns:
        raise ValueError(f"Missing key_col='{cfg.key_col}' for aggregation.")

    def majority(s: pd.Series):
        s = s.dropna()
        if s.empty:
            return np.nan
        return s.value_counts().index[0]

    agg = {}
    if "date_col" in df.columns:
        agg["date_col"] = ("date_col", "min")
    if "topic_col" in df.columns:
        agg["topic_col"] = ("topic_col", majority)
    if "sentiment_col" in df.columns:
        agg["sentiment_col"] = ("sentiment_col", majority)

    out = df.groupby(cfg.key_col, as_index=False).agg(**agg) if agg else df[[cfg.key_col]].drop_duplicates()

    if "populism_col" in df.columns:
        def agg_pop(s: pd.Series):
            s = s.dropna().astype(str).str.strip().str.upper()
            if s.empty:
                return np.nan
            return "YES" if (s == "YES").any() else ("NO" if (s == "NO").any() else np.nan)

        pop = df.groupby(cfg.key_col)["populism_col"].apply(agg_pop)
        out = out.merge(pop.rename("populism_col"), left_on=cfg.key_col, right_index=True, how="left")

    return out


def suffix_cols(df: pd.DataFrame, suffix: str) -> pd.DataFrame:
    df = df.copy()
    ren = {c: f"{c}_{suffix}" for c in ["date_col","topic_col","sentiment_col","populism_col"] if c in df.columns}
    return df.rename(columns=ren)


# -----------------------------
# Pairwise filtering helpers
# -----------------------------
def _filtered_pair_df(df: pd.DataFrame, *, cfg: ComparisonConfig, col_base: str, s1: str, s2: str):
    c1, c2 = f"{col_base}_{s1}", f"{col_base}_{s2}"
    if c1 not in df.columns or c2 not in df.columns:
        return None, None, None

    tmp = df[[c1, c2]].copy()
    mask = tmp[c1].notna() & tmp[c2].notna()

    if cfg.use_filters:
        excl = cfg.filters.get(col_base, {}).get("exclude", set())
        if excl:
            mask &= ~(tmp[c1].isin(excl) & tmp[c2].isin(excl))

    return tmp, mask, (c1, c2)


# -----------------------------
# Reports
# -----------------------------
def agreement_report(df: pd.DataFrame, *, cfg: ComparisonConfig, col_base: str, s1: str, s2: str) -> Dict[str, Any]:
    tmp, mask, cols = _filtered_pair_df(df, cfg=cfg, col_base=col_base, s1=s1, s2=s2)
    if tmp is None:
        return {"pair": f"{s1} vs {s2}", "n": 0, "accuracy": np.nan, "kappa": np.nan}

    c1, c2 = cols
    n = int(mask.sum())
    if n == 0:
        return {"pair": f"{s1} vs {s2}", "n": 0, "accuracy": np.nan, "kappa": np.nan}

    acc = float((tmp.loc[mask, c1].astype(str) == tmp.loc[mask, c2].astype(str)).mean())
    kap = cohen_kappa(tmp.loc[mask, c1], tmp.loc[mask, c2])
    return {"pair": f"{s1} vs {s2}", "n": n, "accuracy": acc, "kappa": kap}


def topic_confusion_pair(df: pd.DataFrame, *, cfg: ComparisonConfig, s1: str, s2: str, normalize: Optional[str] = None) -> Optional[pd.DataFrame]:
    tmp, mask, cols = _filtered_pair_df(df, cfg=cfg, col_base="topic_col", s1=s1, s2=s2)
    if tmp is None:
        return None
    c1, c2 = cols
    tmp = tmp.loc[mask].copy()
    if tmp.empty:
        return None

    ct = pd.crosstab(tmp[c1], tmp[c2], dropna=False).reindex(index=CANON_TOPICS, columns=CANON_TOPICS, fill_value=0)

    if normalize is None:
        return ct
    if normalize == "row":
        denom = ct.sum(axis=1).replace(0, np.nan)
        return (ct.div(denom, axis=0)).fillna(0)
    if normalize == "col":
        denom = ct.sum(axis=0).replace(0, np.nan)
        return (ct.div(denom, axis=1)).fillna(0)
    if normalize == "all":
        total = ct.values.sum()
        return ct / total if total > 0 else ct.astype(float)
    raise ValueError("normalize must be one of: None, 'row', 'col', 'all'")


def read_table(path: Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext} ({path})")


def load_and_align_three(
    path_a: Path,
    path_b: Path,
    path_c: Path,
    *,
    cfg: ComparisonConfig,
    rename_a: dict,
    rename_b: dict,
    rename_c: dict,
) -> pd.DataFrame:
    df_a_raw = read_table(path_a)
    df_b_raw = read_table(path_b)
    df_c_raw = read_table(path_c)


    df_a = standardize_df(df_a_raw, rename_dict=rename_a, cfg=cfg, technique_name="A", dedupe=True)

    df_b_std = standardize_df(df_b_raw, rename_dict=rename_b, cfg=cfg, technique_name="B", dedupe=not cfg.is_sentence_level_b)
    df_b = aggregate_sentence_to_article(df_b_std, cfg=cfg) if cfg.is_sentence_level_b else df_b_std

    df_c_std = standardize_df(df_c_raw, rename_dict=rename_c, cfg=cfg, technique_name="C", dedupe=not cfg.is_sentence_level_c)
    df_c = aggregate_sentence_to_article(df_c_std, cfg=cfg) if cfg.is_sentence_level_c else df_c_std

    A = suffix_cols(df_a, "A")
    B = suffix_cols(df_b, "B")
    C = suffix_cols(df_c, "C")

    df_all = A.merge(B, on=cfg.key_col, how="inner").merge(C, on=cfg.key_col, how="inner")
    return df_all
