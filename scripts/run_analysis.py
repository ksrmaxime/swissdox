# scripts/run_analysis.py
"""
Comprehensive stance analysis on the merged critic output.

Produces:
  01_temporal_evolution.png           — 3 stances over time
  02_by_language.png                  — stances by language (fr/de) over time
  03_by_journal.png                   — stance distribution per journal (top N)
  04a_category_temporal.png           — stances per keyword category over time
  04b_category_by_journal.png         — category targeting per journal
  04c_category_by_language.png        — category targeting by language
  05_critic_stacked_categories.png    — critic curve + category breakdown underneath
  06_critic_stacked_departments.png   — same but for federal departments only
  stats/                              — CSV summaries for every analysis
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd

# ── Plotting style ─────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.dpi": 150,
    "figure.facecolor": "white",
    "axes.facecolor": "#f8f9fa",
    "axes.grid": True,
    "grid.color": "white",
    "grid.linewidth": 1.2,
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "legend.fontsize": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

STANCES       = ["CRITIC", "PRAISE", "NEUTRAL_STATEMENT"]
STANCE_COLORS = {"CRITIC": "#e74c3c", "PRAISE": "#2ecc71", "NEUTRAL_STATEMENT": "#3498db"}
STANCE_LABELS = {"CRITIC": "Critic", "PRAISE": "Praise", "NEUTRAL_STATEMENT": "Neutral"}

# ── Keyword → category mapping ─────────────────────────────────────────────────
CATEGORIES: dict[str, list[str]] = {
    "Bureaucracy": [
        "Bürokratie", "Papierkrieg", "Amtsschimmel", "Regulierungsdichte",
        "Bureaucratie", "Technocratie", "Pouvoir administratif",
        "Appareil administratif", "Appareil étatique", "Appareil de l'État",
        "Bürokraten", "Bureaucrates",
    ],
    "Public Administration": [
        "Verwaltung", "Behörden", "Beamtenapparat",
        "Administration publique", "Administration fédérale", "Administration centrale",
        "Autorités administratives", "Services de l'État", "Services publics",
        "Fonction publique", "Autorités cantonales", "Organes de l'État",
        "Beamte", "Staatsangestellte", "Fonctionnaires", "Employés de l'État",
    ],
    "Bern Executive": [
        "Berner Verwaltung", "Bundesverwaltung",
        "Départements fédéraux", "Offices fédéraux",
    ],
    "Dept: Defence": [
        "VBS", "DDPS",
        "Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport",
        "Département fédéral de la défense, de la protection de la population et des sports",
    ],
    "Dept: Foreign Affairs": [
        "EDA", "DFAE",
        "Eidgenössische Departement für auswärtige Angelegenheiten",
        "Département fédéral des affaires étrangères",
    ],
    "Dept: Environment": [
        "UVEK", "DETEC",
        "Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation",
        "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    ],
    "Dept: Justice": [
        "EJPD", "DFJP",
        "Eidgenössische Justiz- und Polizeidepartement",
        "Département fédéral de justice et police",
    ],
    "Dept: Interior": [
        "EDI", "DFI",
        "Eidgenössische Departement des Innern",
        "Département fédéral de l'intérieur",
    ],
    "Dept: Finance": [
        "EFD", "DFF",
        "Eidgenössische Finanzdepartement",
        "Département fédéral des finances",
    ],
    "Dept: Economy": [
        "WBF", "DEFR",
        "Eidgenössische Departement für Wirtschaft, Bildung und Forschung",
        "Département fédéral de l'économie, de la formation et de la recherche",
    ],
}

DEPT_CATEGORIES  = [c for c in CATEGORIES if c.startswith("Dept:")]
MAIN_CATEGORIES  = ["Bureaucracy", "Public Administration", "Bern Executive"] + DEPT_CATEGORIES

# Assign colours per category
_CMAP_MAIN = plt.cm.tab10.colors
CATEGORY_COLORS: dict[str, str] = {
    cat: _CMAP_MAIN[i % len(_CMAP_MAIN)] for i, cat in enumerate(MAIN_CATEGORIES)
}
CATEGORY_COLORS["Other"] = "#aaaaaa"

# Priority order for primary category (departments first)
_PRIORITY = DEPT_CATEGORIES + ["Bern Executive", "Public Administration", "Bureaucracy"]

# Lookup sets (lower-cased for matching)
_CAT_KW_LOWER: dict[str, set[str]] = {
    cat: {kw.lower() for kw in kws} for cat, kws in CATEGORIES.items()
}


# ── Helper functions ───────────────────────────────────────────────────────────

def get_all_categories(matched_str: str) -> list[str]:
    """Return all categories matched by a matched_keywords string."""
    if pd.isna(matched_str) or not str(matched_str).strip():
        return []
    found = {kw.strip().lower() for kw in str(matched_str).split(",")}
    return [cat for cat, kws in _CAT_KW_LOWER.items() if found & kws]


def assign_primary_category(matched_str: str) -> str:
    cats = set(get_all_categories(matched_str))
    for cat in _PRIORITY:
        if cat in cats:
            return cat
    if cats:
        return next(iter(cats))
    return "Other"


def stance_proportions(df: pd.DataFrame) -> pd.DataFrame:
    """Return count + proportion of each stance in df."""
    counts = df["STANCE"].value_counts().reindex(STANCES, fill_value=0)
    total  = counts.sum()
    props  = counts / total if total > 0 else counts * 0
    return pd.DataFrame({"count": counts, "proportion": props})


def save(fig: plt.Figure, path: Path, tight: bool = True) -> None:
    if tight:
        fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  → {path.name}", flush=True)


def pct_fmt(x, _):
    return f"{x:.0f}%"


# ── Load & prepare ─────────────────────────────────────────────────────────────

def load(path: str) -> pd.DataFrame:
    p = Path(path)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p, low_memory=False)
    print(f"[analysis] Loaded {len(df):,} rows", flush=True)

    df = df[df["STANCE"].isin(STANCES)].copy()
    print(f"[analysis] {len(df):,} rows with valid STANCE", flush=True)

    df["year"] = pd.to_datetime(df["pubtime"], errors="coerce").dt.year
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    df["primary_category"] = df["matched_keywords"].apply(assign_primary_category)
    df["all_categories"]   = df["matched_keywords"].apply(get_all_categories)

    if "language" in df.columns:
        df["language"] = df["language"].astype(str).str.strip().str.lower()

    return df


# ── Figure 01 — Temporal evolution of 3 stances ───────────────────────────────

def fig_01_temporal(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 01] Temporal evolution", flush=True)
    grp = (
        df.groupby(["year", "STANCE"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STANCES, fill_value=0)
    )
    totals = grp.sum(axis=1)
    pct    = grp.div(totals, axis=0) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    for stance in STANCES:
        ax1.plot(grp.index, grp[stance], marker="o", label=STANCE_LABELS[stance],
                 color=STANCE_COLORS[stance], linewidth=2)
    ax1.set_title("Temporal evolution of stances — absolute counts")
    ax1.set_ylabel("Number of sentences")
    ax1.legend()

    for stance in STANCES:
        ax2.plot(pct.index, pct[stance], marker="o", label=STANCE_LABELS[stance],
                 color=STANCE_COLORS[stance], linewidth=2)
    ax2.set_title("Temporal evolution of stances — proportions (%)")
    ax2.set_ylabel("Share (%)")
    ax2.yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax2.set_xlabel("Year")
    ax2.legend()

    save(fig, outdir / "01_temporal_evolution.png")

    pct.round(2).to_csv(outdir / "stats" / "01_temporal_proportions.csv")
    grp.to_csv(outdir / "stats" / "01_temporal_counts.csv")


# ── Figure 02 — By language ────────────────────────────────────────────────────

def fig_02_language(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 02] By language", flush=True)
    langs = df["language"].unique() if "language" in df.columns else []
    langs = sorted([l for l in langs if l in ("fr", "de")])
    if not langs:
        print("  [skip] no language column or no fr/de rows", flush=True)
        return

    fig, axes = plt.subplots(len(langs), 2, figsize=(14, 5 * len(langs)), sharex=True)
    if len(langs) == 1:
        axes = [axes]

    rows = []
    for i, lang in enumerate(langs):
        sub  = df[df["language"] == lang]
        grp  = (sub.groupby(["year", "STANCE"]).size().unstack(fill_value=0)
                .reindex(columns=STANCES, fill_value=0))
        tots = grp.sum(axis=1)
        pct  = grp.div(tots, axis=0) * 100

        for stance in STANCES:
            axes[i][0].plot(grp.index, grp[stance], marker="o",
                            label=STANCE_LABELS[stance], color=STANCE_COLORS[stance], linewidth=2)
            axes[i][1].plot(pct.index, pct[stance], marker="o",
                            label=STANCE_LABELS[stance], color=STANCE_COLORS[stance], linewidth=2)

        axes[i][0].set_title(f"[{lang.upper()}] Absolute counts")
        axes[i][1].set_title(f"[{lang.upper()}] Proportions (%)")
        axes[i][0].set_ylabel("Sentences")
        axes[i][1].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        axes[i][0].legend(fontsize=8)

        for _, row in pct.iterrows():
            for stance in STANCES:
                rows.append({"language": lang, "year": row.name, "stance": stance,
                             "proportion_pct": round(row[stance], 2)})

    axes[-1][0].set_xlabel("Year")
    axes[-1][1].set_xlabel("Year")
    save(fig, outdir / "02_by_language.png")
    pd.DataFrame(rows).to_csv(outdir / "stats" / "02_language_proportions.csv", index=False)


# ── Figure 03 — By journal ─────────────────────────────────────────────────────

def fig_03_journal(df: pd.DataFrame, outdir: Path, top_n: int = 25) -> None:
    print(f"[fig 03] By journal (top {top_n})", flush=True)
    if "medium_name" not in df.columns:
        print("  [skip] no medium_name column", flush=True)
        return

    counts = df["medium_name"].value_counts().head(top_n).index
    sub    = df[df["medium_name"].isin(counts)].copy()

    grp = (
        sub.groupby(["medium_name", "STANCE"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STANCES, fill_value=0)
    )
    totals = grp.sum(axis=1)
    pct    = grp.div(totals, axis=0) * 100
    pct    = pct.sort_values("CRITIC", ascending=True)

    fig, ax = plt.subplots(figsize=(12, max(8, len(pct) * 0.4)))
    y     = np.arange(len(pct))
    left  = np.zeros(len(pct))

    for stance in STANCES:
        ax.barh(y, pct[stance], left=left, color=STANCE_COLORS[stance],
                label=STANCE_LABELS[stance], height=0.7)
        left += pct[stance].values

    ax.set_yticks(y)
    ax.set_yticklabels(pct.index, fontsize=8)
    ax.set_xlabel("Share (%)")
    ax.set_title(f"Stance distribution by journal (top {top_n} by volume, sorted by Critic %)")
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
    ax.legend(loc="lower right")

    # Annotate total n
    for i, journal in enumerate(pct.index):
        n = int(totals[journal])
        ax.text(101, i, f"n={n:,}", va="center", fontsize=7, color="#555")

    save(fig, outdir / "03_by_journal.png")

    out = pct.copy()
    out["total"] = totals
    out.round(2).to_csv(outdir / "stats" / "03_journal_proportions.csv")


# ── Figure 04a — Category × time ──────────────────────────────────────────────

def fig_04a_category_temporal(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 04a] Category × temporal", flush=True)
    cats_present = [c for c in MAIN_CATEGORIES if c in df["primary_category"].unique()]

    grp = (
        df.groupby(["year", "primary_category", "STANCE"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=STANCES, fill_value=0)
    )

    n_cats = len(cats_present)
    fig, axes = plt.subplots(n_cats, 1, figsize=(11, 4 * n_cats), sharex=True)
    if n_cats == 1:
        axes = [axes]

    rows = []
    for i, cat in enumerate(cats_present):
        if cat not in grp.index.get_level_values("primary_category"):
            continue
        sub  = grp.xs(cat, level="primary_category")
        tots = sub.sum(axis=1)
        pct  = sub.div(tots, axis=0) * 100

        for stance in STANCES:
            if stance in pct.columns:
                axes[i].plot(pct.index, pct[stance], marker="o",
                             label=STANCE_LABELS[stance], color=STANCE_COLORS[stance], linewidth=2)

        axes[i].set_title(f"{cat}")
        axes[i].set_ylabel("Share (%)")
        axes[i].yaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        axes[i].legend(fontsize=8)

        for yr, row in pct.iterrows():
            for stance in STANCES:
                rows.append({"category": cat, "year": yr, "stance": stance,
                             "proportion_pct": round(row.get(stance, 0), 2)})

    axes[-1].set_xlabel("Year")
    fig.suptitle("Stance evolution by keyword category", fontsize=14, y=1.01)
    save(fig, outdir / "04a_category_temporal.png")
    pd.DataFrame(rows).to_csv(outdir / "stats" / "04a_category_temporal.csv", index=False)


# ── Figure 04b — Category × journal ───────────────────────────────────────────

def fig_04b_category_journal(df: pd.DataFrame, outdir: Path, top_n: int = 15) -> None:
    print("[fig 04b] Category × journal", flush=True)
    if "medium_name" not in df.columns:
        return

    top_journals = df["medium_name"].value_counts().head(top_n).index
    sub = df[df["medium_name"].isin(top_journals)].copy()

    cats_present = [c for c in MAIN_CATEGORIES if c in sub["primary_category"].unique()]
    n_cats = len(cats_present)
    fig, axes = plt.subplots(1, n_cats, figsize=(6 * n_cats, 8), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for i, cat in enumerate(cats_present):
        sub_cat = sub[sub["primary_category"] == cat]
        grp = (
            sub_cat.groupby(["medium_name", "STANCE"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=STANCES, fill_value=0)
        )
        tots = grp.sum(axis=1)
        pct  = grp.div(tots, axis=0) * 100
        pct  = pct.sort_values("CRITIC", ascending=True)

        y    = np.arange(len(pct))
        left = np.zeros(len(pct))
        for stance in STANCES:
            axes[i].barh(y, pct[stance], left=left, color=STANCE_COLORS[stance],
                         label=STANCE_LABELS[stance], height=0.7)
            left += pct[stance].values

        axes[i].set_yticks(y)
        axes[i].set_yticklabels(pct.index, fontsize=7)
        axes[i].set_title(cat, fontsize=9)
        axes[i].xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        if i == 0:
            axes[i].legend(fontsize=7)

    fig.suptitle(f"Stance by journal per category (top {top_n} journals)", fontsize=13)
    save(fig, outdir / "04b_category_by_journal.png")


# ── Figure 04c — Category × language ──────────────────────────────────────────

def fig_04c_category_language(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 04c] Category × language", flush=True)
    if "language" not in df.columns:
        return

    langs = sorted([l for l in df["language"].unique() if l in ("fr", "de")])
    cats_present = [c for c in MAIN_CATEGORIES if c in df["primary_category"].unique()]

    rows = []
    fig, axes = plt.subplots(1, len(langs), figsize=(8 * len(langs), max(6, len(cats_present) * 0.5)), sharey=True)
    if len(langs) == 1:
        axes = [axes]

    for j, lang in enumerate(langs):
        sub  = df[df["language"] == lang]
        grp  = (
            sub.groupby(["primary_category", "STANCE"])
            .size()
            .unstack(fill_value=0)
            .reindex(columns=STANCES, fill_value=0)
        )
        grp  = grp.reindex(cats_present, fill_value=0)
        tots = grp.sum(axis=1)
        pct  = grp.div(tots, axis=0) * 100

        y    = np.arange(len(pct))
        left = np.zeros(len(pct))
        for stance in STANCES:
            axes[j].barh(y, pct[stance], left=left, color=STANCE_COLORS[stance],
                         label=STANCE_LABELS[stance], height=0.7)
            left += pct[stance].values

        axes[j].set_yticks(y)
        axes[j].set_yticklabels(pct.index, fontsize=9)
        axes[j].set_title(f"Language: {lang.upper()}")
        axes[j].xaxis.set_major_formatter(mticker.FuncFormatter(pct_fmt))
        axes[j].legend(fontsize=8)

        for cat in cats_present:
            for stance in STANCES:
                rows.append({"language": lang, "category": cat, "stance": stance,
                             "proportion_pct": round(pct.loc[cat, stance] if cat in pct.index else 0, 2)})

    fig.suptitle("Stance by category and language", fontsize=13)
    save(fig, outdir / "04c_category_by_language.png")
    pd.DataFrame(rows).to_csv(outdir / "stats" / "04c_category_language.csv", index=False)


# ── Figure 05 — Critic curve + stacked keyword categories ─────────────────────

def fig_05_critic_stacked(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 05] Critic stacked by category", flush=True)
    critics = df[df["STANCE"] == "CRITIC"].copy()

    cats_present = [c for c in MAIN_CATEGORIES if c in critics["primary_category"].unique()]

    pivot = (
        critics.groupby(["year", "primary_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=cats_present, fill_value=0)
    )
    total_critic = pivot.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [CATEGORY_COLORS.get(c, "#aaa") for c in cats_present]
    ax.stackplot(
        pivot.index,
        [pivot[c] for c in cats_present],
        labels=cats_present,
        colors=colors,
        alpha=0.85,
    )
    ax.plot(pivot.index, total_critic, color="black", linewidth=2.5,
            marker="o", markersize=5, label="Total CRITIC", zorder=5)

    ax.set_title("CRITIC sentences over time — breakdown by keyword category", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of CRITIC sentences")
    ax.legend(loc="upper left", fontsize=8, ncol=2)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save(fig, outdir / "05_critic_stacked_categories.png")
    pivot["TOTAL"] = total_critic
    pivot.to_csv(outdir / "stats" / "05_critic_stacked_categories.csv")


# ── Figure 06 — Critic curve + stacked federal departments only ────────────────

def fig_06_critic_departments(df: pd.DataFrame, outdir: Path) -> None:
    print("[fig 06] Critic stacked by department", flush=True)
    critics = df[(df["STANCE"] == "CRITIC") & (df["primary_category"].isin(DEPT_CATEGORIES))].copy()

    if critics.empty:
        print("  [skip] no critic rows for departments", flush=True)
        return

    depts_present = [c for c in DEPT_CATEGORIES if c in critics["primary_category"].unique()]

    pivot = (
        critics.groupby(["year", "primary_category"])
        .size()
        .unstack(fill_value=0)
        .reindex(columns=depts_present, fill_value=0)
    )
    total_dept_critic = pivot.sum(axis=1)

    fig, ax = plt.subplots(figsize=(12, 6))

    colors = [CATEGORY_COLORS.get(c, "#aaa") for c in depts_present]
    ax.stackplot(
        pivot.index,
        [pivot[c] for c in depts_present],
        labels=[c.replace("Dept: ", "") for c in depts_present],
        colors=colors,
        alpha=0.85,
    )
    ax.plot(pivot.index, total_dept_critic, color="black", linewidth=2.5,
            marker="o", markersize=5, label="Total dept. CRITIC", zorder=5)

    ax.set_title("CRITIC sentences targeting federal departments — breakdown by department", fontsize=13)
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of CRITIC sentences")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{int(x):,}"))

    save(fig, outdir / "06_critic_stacked_departments.png")
    pivot["TOTAL"] = total_dept_critic
    pivot.to_csv(outdir / "stats" / "06_critic_stacked_departments.csv")


# ── Summary stats ──────────────────────────────────────────────────────────────

def save_global_stats(df: pd.DataFrame, outdir: Path) -> None:
    total = len(df)
    print(f"\n[stats] Total sentences analysed : {total:,}")
    for stance in STANCES:
        n = (df["STANCE"] == stance).sum()
        print(f"  {stance:<22}: {n:,} ({n/total*100:.1f}%)")

    overall = stance_proportions(df)
    overall.to_csv(outdir / "stats" / "00_global_overview.csv")

    cat_counts = df["primary_category"].value_counts()
    cat_counts.to_csv(outdir / "stats" / "00_category_counts.csv", header=["count"])


# ── Main ───────────────────────────────────────────────────────────────────────

def main() -> int:
    ap = argparse.ArgumentParser(description="Stance analysis on merged critic output")
    ap.add_argument("--input",   required=True,  help="Merged critic CSV or parquet")
    ap.add_argument("--outdir",  required=True,  help="Output directory for figures and stats")
    ap.add_argument("--top_journals", type=int, default=25, help="Top N journals to show (default 25)")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    (outdir / "stats").mkdir(parents=True, exist_ok=True)

    df = load(args.input)
    save_global_stats(df, outdir)

    fig_01_temporal(df, outdir)
    fig_02_language(df, outdir)
    fig_03_journal(df, outdir, top_n=args.top_journals)
    fig_04a_category_temporal(df, outdir)
    fig_04b_category_journal(df, outdir, top_n=15)
    fig_04c_category_language(df, outdir)
    fig_05_critic_stacked(df, outdir)
    fig_06_critic_departments(df, outdir)

    print(f"\n[analysis] Done. Results in {outdir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
