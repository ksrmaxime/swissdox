from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -------------------------------
# IO
# -------------------------------
def read_table(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    ext = path.suffix.lower()
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".parquet", ".pq"}:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported file extension: {ext} ({path})")


# -------------------------------
# Canonical label spaces
# -------------------------------
CANON_TOPICS = [
    "Foreign Affairs", "Culture", "Health", "Social", "Justice", "Migration", "Defence", "State Politics", "Sport", "Finance",
    "Economy", "Education", "Research", "Environment", "Transports", "Energy", "Communication", "Other"
]

def topic_color_map(cmap_name: str = "tab20"):
    cmap = plt.get_cmap(cmap_name)
    # stable: la couleur dépend uniquement de la position dans CANON_TOPICS
    return {t: cmap(i % cmap.N) for i, t in enumerate(CANON_TOPICS)}

TOPIC_COLORS = topic_color_map("tab20")

CANON_SENTIMENT = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
CANON_POPULISM = ["YES", "NO"]


TOPIC_TO_DEPT = {
    "Foreign Affairs": "EDA",
    "Culture": "EDI", "Health": "EDI", "Social": "EDI",
    "Justice": "EJPD", "Migration": "EJPD",
    "Defence": "VBS", "Sport": "VBS",
    "Finance": "EFD",
    "Economy": "WBF", "Education": "WBF", "Research": "WBF",
    "Environment": "UVEK", "Transports": "UVEK", "Energy": "UVEK", "Communication": "UVEK",
    "Other": "Other",
}
# Canonical department order (fixed!)
# Canonical department order (fixed)
DEPT_ORDER = ["EDA", "EDI", "EJPD", "VBS", "EFD", "WBF", "UVEK", "Other"]

# Hard-coded matplotlib tab10 colors (stable across everything)
DEPT_COLORS = {
    "EDA":  "#1f77b4",  # blue
    "EDI":  "#ff7f0e",  # orange
    "EJPD": "#2ca02c",  # green
    "VBS":  "#d62728",  # red
    "EFD":  "#9467bd",  # purple
    "WBF":  "#8c564b",  # brown
    "UVEK": "#e377c2",  # pink
    "Other":"#7f7f7f",  # grey
}

FALLBACK_COLOR = "#7f7f7f"



# -------------------------------
# Config
# -------------------------------
@dataclass(frozen=True)
class StandardizeConfig:
    key_col: str = "article_id"

    # after rename => must exist:
    date_col: str = "date_col"
    topic_col: str = "topic_col"
    sentiment_col: str = "sentiment_col"
    populism_col: str = "populism_col"

    canon_topics: List[str] = None  # set in __post_init__
    canon_sentiment: List[str] = None
    canon_populism: List[str] = None


def _default_cfg() -> StandardizeConfig:
    return StandardizeConfig(
        canon_topics=CANON_TOPICS,
        canon_sentiment=CANON_SENTIMENT,
        canon_populism=CANON_POPULISM,
    )


# -------------------------------
# Standardization helpers
# -------------------------------
def _normalize_string(x):
    if pd.isna(x):
        return np.nan
    x = str(x).strip()
    return np.nan if x == "" else x

def _apply_mapping(series: pd.Series, mapping: Dict) -> pd.Series:
    def f(v):
        nv = _normalize_string(v)
        return mapping.get(v, mapping.get(nv, v))
    return series.map(f)

def standardize_df(
    df: pd.DataFrame,
    *,
    rename: Dict[str, str],
    technique_name: str,
    cfg: Optional[StandardizeConfig] = None,
    topic_map: Optional[Dict] = None,
    sentiment_map: Optional[Dict] = None,
    populism_map: Optional[Dict] = None,
    dedupe_on_key: bool = False,
) -> pd.DataFrame:
    """
    Returns a standardized df with columns:
      key_col, date_col, topic_col, sentiment_col, populism_col (if present)
    """
    cfg = cfg or _default_cfg()
    df = df.copy()
    df.columns = [c.strip() for c in df.columns]
    df = df.rename(columns=rename)

    if cfg.key_col not in df.columns:
        raise ValueError(f"[{technique_name}] key_col='{cfg.key_col}' missing after rename. Columns={list(df.columns)}")

    for col in [cfg.date_col, cfg.topic_col, cfg.sentiment_col, cfg.populism_col]:
        if col in df.columns:
            df[col] = df[col].map(_normalize_string)

    if cfg.date_col in df.columns:
        df[cfg.date_col] = pd.to_datetime(df[cfg.date_col], errors="coerce")

    if cfg.topic_col in df.columns and topic_map:
        df[cfg.topic_col] = _apply_mapping(df[cfg.topic_col], topic_map).map(_normalize_string)

    if cfg.sentiment_col in df.columns and sentiment_map:
        df[cfg.sentiment_col] = _apply_mapping(df[cfg.sentiment_col], sentiment_map).map(_normalize_string)
        # normalize sentiment to UPPER if already textual
        df[cfg.sentiment_col] = df[cfg.sentiment_col].astype(str).str.strip().str.upper().replace({"NAN": np.nan})

    if cfg.populism_col in df.columns and populism_map:
        s_num = pd.to_numeric(df[cfg.populism_col], errors="coerce")
        df[cfg.populism_col] = np.where(s_num.notna(), s_num, df[cfg.populism_col])
        df[cfg.populism_col] = _apply_mapping(pd.Series(df[cfg.populism_col]), populism_map).map(_normalize_string)
        df[cfg.populism_col] = df[cfg.populism_col].astype(str).str.strip().str.upper().replace({"NAN": np.nan})

    keep = [cfg.key_col] + [c for c in [cfg.date_col, cfg.topic_col, cfg.sentiment_col, cfg.populism_col] if c in df.columns]
    df = df[keep]

    if dedupe_on_key:
        df = df.drop_duplicates(subset=[cfg.key_col], keep="first")

    return df


def to_month_frame(df: pd.DataFrame, technique_name: str, cfg: Optional[StandardizeConfig] = None) -> pd.DataFrame:
    cfg = cfg or _default_cfg()
    for c in [cfg.date_col, cfg.topic_col, cfg.sentiment_col]:
        if c not in df.columns:
            raise ValueError(f"[{technique_name}] Missing '{c}' for plots.")
    out = df.dropna(subset=[cfg.date_col]).copy()
    out["month"] = out[cfg.date_col].dt.to_period("M").dt.to_timestamp()
    out["dept"] = out[cfg.topic_col].map(TOPIC_TO_DEPT).fillna("Other")
    return out


# -------------------------------
# Tables
# -------------------------------
def share_table(df: pd.DataFrame, index_col: str, value_col: str, categories: List[str]) -> pd.DataFrame:
    ct = pd.crosstab(df[index_col], df[value_col])
    ct = ct.reindex(columns=categories, fill_value=0)
    shares = ct.div(ct.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
    return shares.sort_index()


# -------------------------------
# Plotting utilities
# -------------------------------
def _ensure_outdir(outdir: Optional[str | Path]) -> Optional[Path]:
    if outdir is None:
        return None
    p = Path(outdir)
    p.mkdir(parents=True, exist_ok=True)
    return p

def _save_or_show(fig, outdir: Optional[Path], filename: str, show: bool):
    if outdir is not None:
        fig.savefig(outdir / filename, dpi=200, bbox_inches="tight")
    if show:
        plt.show()
    plt.close(fig)


# -------------------------------
# Plots
# -------------------------------
def plot_sentiment_shares_over_time(df_m: pd.DataFrame, model_name: str, *, outdir: Optional[Path] = None, show: bool = True):
    shares = share_table(df_m, "month", "sentiment_col", CANON_SENTIMENT)
    if shares.empty:
        return

    fig, ax = plt.subplots()
    for s in shares.columns:
        ax.plot(shares.index, shares[s].values, label=s)
    ax.set_title(f"{model_name} — Sentiment shares over time (monthly)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right")
    ax.set_xticks(shares.index)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in shares.index], rotation=45, ha="right")
    fig.tight_layout()
    _save_or_show(fig, outdir, f"{model_name}_sentiment_shares.png", show)


def plot_topic_shares_stacked(df_m: pd.DataFrame, model_name: str, *, outdir: Optional[Path] = None, show: bool = True):
    shares = share_table(df_m, "month", "topic_col", CANON_TOPICS)
    if shares.empty:
        return

    x = np.arange(len(shares.index))
    bottom = np.zeros(len(shares.index))

    fig, ax = plt.subplots()
    for t in shares.columns:
        vals = shares[t].values
        ax.bar(x, vals, bottom=bottom, label=t, color=TOPIC_COLORS.get(t))
        bottom += vals


    ax.set_title(f"{model_name} — Topic shares over time (stacked, normalized)")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Share")
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in shares.index], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.25), ncol=4, frameon=False)
    fig.tight_layout()
    _save_or_show(fig, outdir, f"{model_name}_topic_shares_stacked.png", show)


def plot_negativity_line_plus_dept_scaled(df_m: pd.DataFrame, model_name: str, *, outdir: Optional[Path] = None, show: bool = True):
    """
    One plot:
      - line: monthly negativity rate (NEGATIVE / total)
      - stacked bars: dept shares among NEGATIVE, scaled by negativity rate
        => total bar height equals negativity rate
    """
    total = df_m.groupby("month").size().sort_index()
    if total.empty:
        return

    neg_df = df_m[df_m["sentiment_col"] == "NEGATIVE"].copy()
    neg_count = neg_df.groupby("month").size().reindex(total.index).fillna(0)
    neg_rate = (neg_count / total).fillna(0)

    if neg_df.empty:
        dept_sh = pd.DataFrame(index=total.index, columns=DEPT_ORDER).fillna(0)
    else:
        dept_sh = share_table(neg_df, "month", "dept", DEPT_ORDER).reindex(total.index).fillna(0)

    dept_scaled = dept_sh.mul(neg_rate, axis=0)

    months = total.index
    x = np.arange(len(months))
    bottom = np.zeros(len(months))

    fig, ax = plt.subplots()

    for d in DEPT_ORDER:
        vals = dept_scaled[d].values if d in dept_scaled.columns else np.zeros(len(months))
        ax.bar(x, vals, bottom=bottom, label=d, color=DEPT_COLORS.get(d))
        bottom += vals

    ax.plot(x, neg_rate.values, linewidth=2)

    ax.set_title(f"{model_name} — NEGATIVE share + dept composition (scaled)")
    ax.set_ylabel("Share")
    ax.set_ylim(0, max(0.01, float(neg_rate.max()) * 1.15))
    ax.set_xticks(x)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in months], rotation=45, ha="right")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend(loc="upper right", frameon=True)
    fig.tight_layout()
    _save_or_show(fig, outdir, f"{model_name}_negativity_dept_scaled.png", show)


def plot_populism_yes_share_over_time(df: pd.DataFrame, model_name: str, *, outdir: Optional[Path] = None, show: bool = True):
    if "populism_col" not in df.columns:
        return

    tmp = df.dropna(subset=["date_col"]).copy()
    tmp["month"] = tmp["date_col"].dt.to_period("M").dt.to_timestamp()
    tmp = tmp.dropna(subset=["populism_col"])
    if tmp.empty:
        return

    tmp["pop"] = tmp["populism_col"].astype(str).str.strip().str.upper()
    yes_share = tmp.groupby("month")["pop"].apply(lambda s: (s == "YES").mean()).sort_index()

    fig, ax = plt.subplots()
    ax.plot(yes_share.index, yes_share.values, linewidth=2)
    ax.set_title(f"{model_name} — Populism YES share over time (monthly)")
    ax.set_ylim(0, max(0.01, float(yes_share.max()) * 1.15))
    ax.set_ylabel("Share YES")
    ax.grid(True, axis="y", alpha=0.3)
    ax.set_xticks(yes_share.index)
    ax.set_xticklabels([d.strftime("%Y-%m") for d in yes_share.index], rotation=45, ha="right")
    fig.tight_layout()
    _save_or_show(fig, outdir, f"{model_name}_populism_yes_share.png", show)


# -------------------------------
# Orchestrator
# -------------------------------
def run_all_plots(
    *,
    path_a: str | Path,
    path_b: str | Path,
    path_c: str | Path,
    name_a: str,
    name_b: str,
    name_c: str,
    rename_a: Dict[str, str],
    rename_b: Dict[str, str],
    rename_c: Dict[str, str],
    topic_map: Dict,
    sentiment_map: Dict,
    populism_map: Dict,
    outdir: Optional[str | Path] = None,
    show: bool = True,
) -> None:
    outdir_p = _ensure_outdir(outdir)
    cfg = _default_cfg()

    # A is article-level => dedupe on key
    dfA = standardize_df(
        read_table(path_a),
        rename=rename_a,
        technique_name=name_a,
        cfg=cfg,
        topic_map=topic_map,
        sentiment_map=sentiment_map,
        populism_map=populism_map,
        dedupe_on_key=True,
    )

    # B and C: sentence-level for plots => do NOT dedupe
    dfB = standardize_df(
        read_table(path_b),
        rename=rename_b,
        technique_name=name_b,
        cfg=cfg,
        topic_map=topic_map,
        sentiment_map=sentiment_map,
        populism_map=populism_map,
        dedupe_on_key=False,
    )

    dfC = standardize_df(
        read_table(path_c),
        rename=rename_c,
        technique_name=name_c,
        cfg=cfg,
        topic_map=topic_map,
        sentiment_map=sentiment_map,
        populism_map=populism_map,
        dedupe_on_key=False,
    )

    mA = to_month_frame(dfA, name_a, cfg)
    mB = to_month_frame(dfB, name_b, cfg)
    mC = to_month_frame(dfC, name_c, cfg)

    # 1) sentiment shares over time
    plot_sentiment_shares_over_time(mA, name_a, outdir=outdir_p, show=show)
    plot_sentiment_shares_over_time(mB, name_b, outdir=outdir_p, show=show)
    plot_sentiment_shares_over_time(mC, name_c, outdir=outdir_p, show=show)

    # 2) topic stacked shares
    plot_topic_shares_stacked(mA, name_a, outdir=outdir_p, show=show)
    plot_topic_shares_stacked(mB, name_b, outdir=outdir_p, show=show)
    plot_topic_shares_stacked(mC, name_c, outdir=outdir_p, show=show)

    # 3) negativity + dept composition (scaled)
    plot_negativity_line_plus_dept_scaled(mA, name_a, outdir=outdir_p, show=show)
    plot_negativity_line_plus_dept_scaled(mB, name_b, outdir=outdir_p, show=show)
    plot_negativity_line_plus_dept_scaled(mC, name_c, outdir=outdir_p, show=show)

    # 4) populism YES share over time — A and B only
    plot_populism_yes_share_over_time(dfA, name_a, outdir=outdir_p, show=show)
    plot_populism_yes_share_over_time(dfB, name_b, outdir=outdir_p, show=show)
