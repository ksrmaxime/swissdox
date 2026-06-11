# 03_extract_sentences.py
"""
Sentence extraction step between sbatch_02_classify_articles.sh and sbatch_04_classify_stance.sh.

Input  : output of 02_classify_articles.py (.csv or .parquet)
           must have columns: 'swiss', and a text column (default 'text', fallback 'lead')
Output : critic_base.csv ready to be consumed by 04_classify_stance.py
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


def process_chunk(chunk: pd.DataFrame, text_col: str, swiss_col: str, kw_pattern) -> list[dict]:
    """Process one chunk of articles, return list of sentence dicts. Never holds exploded df in memory."""
    meta_cols = [c for c in chunk.columns if c != text_col]
    rows = []
    for _, article in chunk.iterrows():
        if str(article[swiss_col]).strip().upper() != "YES":
            continue
        sentences = split_sentences(article[text_col])
        for sent in sentences:
            sent = sent.strip()
            if not sent or sent == "nan":
                continue
            if not kw_pattern.search(sent):
                continue
            matched = ", ".join(sorted(set(
                x.strip() for x in kw_pattern.findall(sent) if isinstance(x, str)
            )))
            row = {c: article[c] for c in meta_cols}
            row["sentence"] = sent
            row["matched_keywords"] = matched
            rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cut articles into keyword-filtered sentences for 04_classify_stance.py"
    )
    ap.add_argument("--input", required=True,
                    help="Article pipeline output (.csv or .parquet)")
    ap.add_argument("--output", required=True,
                    help="Output path for critic_base (.csv)")
    ap.add_argument("--text_col", default=None,
                    help="Column to split into sentences. Auto-detected if omitted (prefers 'text', falls back to 'lead').")
    ap.add_argument("--swiss_col", default="swiss",
                    help="Column containing YES/NO swiss flag (default: swiss)")
    ap.add_argument("--chunk_size", type=int, default=2000,
                    help="Nombre d'articles traités par chunk (default: 2000)")
    args = ap.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Input file not found: {input_path}", file=sys.stderr)
        return 1

    # ── Load uniquement pour inspecter les colonnes ───────────────────────────
    if input_path.suffix == ".parquet":
        df_full = pd.read_parquet(input_path)
    else:
        df_full = pd.read_csv(input_path)

    print(f"[sentence_cutting] Loaded {len(df_full):,} rows from {input_path}")

    if args.text_col:
        text_col = args.text_col
    elif "text" in df_full.columns:
        text_col = "text"
    elif "lead" in df_full.columns:
        text_col = "lead"
    else:
        print(f"[ERROR] No 'text' or 'lead' column found. Columns: {list(df_full.columns)}", file=sys.stderr)
        return 1
    print(f"[sentence_cutting] Using text column: '{text_col}'")

    if args.swiss_col not in df_full.columns:
        print(f"[ERROR] Swiss column '{args.swiss_col}' not found. Columns: {list(df_full.columns)}", file=sys.stderr)
        return 1

    n_total = len(df_full)
    n_swiss = int((df_full[args.swiss_col].astype(str).str.strip().str.upper() == "YES").sum())
    print(f"[sentence_cutting] Rows swiss=YES: {n_swiss:,} / {n_total:,}")

    kw_pattern = build_kw_pattern(KW_SENT)

    # ── Traitement par chunks + écriture incrémentale ─────────────────────────
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentence_id = 1
    total_sentences = 0
    first_chunk = True

    for chunk_start in range(0, n_total, args.chunk_size):
        chunk = df_full.iloc[chunk_start:chunk_start + args.chunk_size]
        rows = process_chunk(chunk, text_col, args.swiss_col, kw_pattern)

        if not rows:
            print(f"[sentence_cutting] Chunk {chunk_start//args.chunk_size + 1}: 0 phrases", flush=True)
            continue

        chunk_df = pd.DataFrame(rows)
        chunk_df.insert(0, "sentence_id", range(sentence_id, sentence_id + len(chunk_df)))
        sentence_id += len(chunk_df)
        total_sentences += len(chunk_df)

        chunk_df.to_csv(
            output_path,
            mode="w" if first_chunk else "a",
            header=first_chunk,
            index=False,
            encoding="utf-8-sig",
        )
        first_chunk = False

        print(f"[sentence_cutting] Chunk {chunk_start//args.chunk_size + 1} "
              f"({chunk_start+1}-{min(chunk_start+args.chunk_size, n_total)}/{n_total}) "
              f"→ {len(rows)} phrases | total: {total_sentences:,}", flush=True)

    print(f"[sentence_cutting] Saved {total_sentences:,} sentences → {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
