# scripts/run_sentence_cutting.py
"""
Sentence cutting step between sbatch_run_article.sh and sbatch_run_critic.sh.

Input  : output of run_article_pipeline.py (.csv or .parquet)
           must have columns: 'swiss', and a text column (default 'text', fallback 'lead')
Output : critic_base.csv ready to be consumed by run_critic_pipeline.py
           columns: sentence_id, sentence, matched_keywords, + all article metadata
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd


# ── Keywords (same as SentancesCutting.ipynb) ─────────────────────────────────
KW_SENT = [
    'Bürokratie', 'Berner Verwaltung', 'Papierkrieg', 'Verwaltung',
    'Bundesverwaltung', 'Beamtenapparat', 'Amtsschimmel', 'Regulierungsdichte',
    'Behörden', 'Bürokraten', 'Beamte', 'Staatsangestellte',
    'Bureaucratie', 'Administration publique', 'Administration fédérale',
    'Appareil administratif', 'Appareil étatique', "Appareil de l'État",
    'Autorités administratives', "Services de l'État", 'Services publics',
    'Fonction publique', 'Pouvoir administratif', 'Autorités cantonales',
    'Administration centrale', 'Départements fédéraux', 'Offices fédéraux',
    "Organes de l'État", 'Technocratie', 'Bureaucrates', 'Fonctionnaires',
    "Employés de l'État",
    'VBS', 'DDPS',
    'Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport',
    'Département fédéral de la défense, de la protection de la population et des sports',
    'EDA', 'DFAE',
    'Eidgenössische Departement für auswärtige Angelegenheiten',
    'Département fédéral des affaires étrangères',
    'UVEK', 'DETEC',
    'Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation',
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    'EJPD', 'DFJP',
    'Eidgenössische Justiz- und Polizeidepartement',
    'Département fédéral de justice et police',
    'EDI', 'DFI',
    'Eidgenössische Departement des Innern',
    "Département fédéral de l'intérieur",
    'EFD', 'DFF',
    'Eidgenössische Finanzdepartement',
    'Département fédéral des finances',
    'WBF', 'DEFR',
    'Eidgenössische Departement für Wirtschaft, Bildung und Forschung',
    "Département fédéral de l'économie, de la formation et de la recherche",
]


def build_kw_pattern(keywords):
    patterns = []
    for k in keywords:
        if k.isupper() and len(k) <= 4:
            patterns.append(rf'\b{re.escape(k)}\b')
        else:
            patterns.append(re.escape(k))
    return re.compile('|'.join(patterns), flags=re.IGNORECASE)


# ── Smart sentence splitter (same as SentancesCutting.ipynb) ─────────────────
_PLACEHOLDER = '\x00'

_ABBREVS = sorted([
    'Prof', 'Dr', 'Hr', 'Fr', 'Mr', 'Mrs', 'Ms', 'Mme', 'M',
    'St', 'Kt', 'Art', 'Abs', 'Ziff', 'lit', 'Bst', 'Nr', 'No',
    'bzw', 'resp', 'etc', 'usw', 'ua', 'ca', 'vgl', 'ggf',
    'inkl', 'exkl', 'evtl', 'insb', 'sog', 'mind', 'max', 'min',
    'Co', 'Cie',
    'Jan', 'Feb', 'Mär', 'Apr', 'Jun', 'Jul', 'Aug', 'Sep', 'Okt', 'Nov', 'Dez',
    'cf', 'vs', 'fig', 'Fig', 'Bd', 'Jg', 'ff', 'op',
], key=len, reverse=True)

_abbrev_re   = re.compile(r'\b(' + '|'.join(re.escape(a) for a in _ABBREVS) + r')\.(\s)', re.IGNORECASE)
_ordinal_re  = re.compile(r'(\d+)\.(\s)')
_initial_re  = re.compile(r'\b([A-ZÜÄÖ])\.(\s+[A-ZÜÄÖÉÀÂÊ])')
_multidot_re = re.compile(r'\b([a-zA-Züäö]\.)+([a-zA-Züäö])\.(?=\s)')


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    t = text
    t = _multidot_re.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), t)
    t = _abbrev_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _ordinal_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _initial_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.replace(_PLACEHOLDER, '.').strip() for p in parts if p.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cut articles into keyword-filtered sentences for run_critic_pipeline.py"
    )
    ap.add_argument("--input", required=True,
                    help="Article pipeline output (.csv or .parquet)")
    ap.add_argument("--output", required=True,
                    help="Output path for critic_base (.csv)")
    ap.add_argument("--text_col", default=None,
                    help="Column to split into sentences. Auto-detected if omitted (prefers 'text', falls back to 'lead').")
    ap.add_argument("--swiss_col", default="swiss",
                    help="Column containing YES/NO swiss flag (default: swiss)")
    args = ap.parse_args()

    # ── Load ──────────────────────────────────────────────────────────────────
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    df = pd.read_parquet(input_path) if input_path.suffix == ".parquet" else pd.read_csv(input_path)
    print(f"[sentence_cutting] Loaded {len(df):,} rows from {input_path}")

    # ── Detect text column ───────────────────────────────────────────────────
    if args.text_col:
        text_col = args.text_col
    elif "text" in df.columns:
        text_col = "text"
    elif "lead" in df.columns:
        text_col = "lead"
    else:
        print(f"[ERROR] No 'text' or 'lead' column found. Columns: {list(df.columns)}", file=sys.stderr)
        return 1
    print(f"[sentence_cutting] Using text column: '{text_col}'")

    if args.swiss_col not in df.columns:
        print(f"[ERROR] Swiss column '{args.swiss_col}' not found. Columns: {list(df.columns)}", file=sys.stderr)
        return 1

    # ── Filter swiss == YES ───────────────────────────────────────────────────
    mask_swiss = df[args.swiss_col].astype(str).str.strip().str.upper() == "YES"
    df_swiss = df[mask_swiss].copy()
    print(f"[sentence_cutting] Rows swiss=YES: {len(df_swiss):,} / {len(df):,}")

    # ── Split sentences ───────────────────────────────────────────────────────
    kw_pattern = build_kw_pattern(KW_SENT)

    df_swiss["sentence"] = df_swiss[text_col].apply(split_sentences)
    df_exp = df_swiss.explode("sentence", ignore_index=True)
    df_exp["sentence"] = df_exp["sentence"].astype(str).str.strip()
    df_exp = df_exp[df_exp["sentence"].ne("") & df_exp["sentence"].ne("nan")]

    # ── Keyword filter ────────────────────────────────────────────────────────
    mask_kw = df_exp["sentence"].str.contains(kw_pattern, na=False)
    df_exp = df_exp[mask_kw].copy()
    print(f"[sentence_cutting] Sentences kept after keyword filter: {len(df_exp):,}")

    df_exp["matched_keywords"] = df_exp["sentence"].str.findall(kw_pattern).apply(
        lambda lst: ", ".join(sorted(set(x.strip() for x in lst if isinstance(x, str))))
    )

    # ── Drop source text column, add sentence_id ─────────────────────────────
    df_exp = df_exp.drop(columns=[text_col], errors="ignore")
    df_exp.insert(0, "sentence_id", range(1, len(df_exp) + 1))

    # ── Save ──────────────────────────────────────────────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_exp.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"[sentence_cutting] Saved {len(df_exp):,} sentences → {output_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
