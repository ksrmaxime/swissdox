# 03_extract_sentences.py
"""
Sentence extraction step between sbatch_01_download.sh and sbatch_04_classify_stance.sh.

Input  : output of 01_download.py (.csv or .parquet)
           must have a text column (default 'text', fallback 'lead')
Output : criticism_base.csv ready to be consumed by 04_classify_stance.py
           columns: sentence_id, sentence, matched_keywords, type, sub_type,
           keyword_abbrev, + all article metadata

One output row = one sentence + one canonical keyword. A sentence that
matches several *distinct* entities (e.g. "EDA" and "Direktion für
Völkerrecht") is duplicated, once per entity — that duplication is expected
and is what makes the file usable for per-entity stance/populism classification
downstream. A sentence that matches several *aliases of the same entity*
(e.g. "Seco" and "Staatssekretariat für Wirtschaft") produces a single row,
keeping only the longest (spelled-out) alias — see ENTITY_GROUPS below.

Keywords used for sentence cutting are the same ones used to build the
run1 Swissdox query (src/download_src.py), minus the DE_LEVEL/FR_LEVEL terms
(fédéral, cantonal, Suisse, ...) which only serve to filter articles at query
time and are too generic to use for sentence-level matching.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.download_src import (
    DE_TERMS, FR_TERMS, DEPARTMENTS, ADMIN_UNITS,
    DEPARTMENTS_GROUPS, ADMIN_UNITS_GROUPS,
    classify_keyword,
)

# ── Keywords: generic DE/FR bureaucracy terms + departments + admin units ────
# (kept in sync with the run1 query — see src/download_src.py)
KW_SENT = DE_TERMS + FR_TERMS + DEPARTMENTS + ADMIN_UNITS

# Groups of surface forms (abbreviation + full name, DE + FR) that denote the
# SAME real-world entity. DE_TERMS/FR_TERMS have no aliases and are left out
# on purpose: they become their own singleton group further down.
ENTITY_GROUPS = DEPARTMENTS_GROUPS + ADMIN_UNITS_GROUPS


def build_kw_pattern(keywords):
    # Longest-first: without this, a short keyword that is a literal prefix
    # of a longer one (e.g. "Administration" / "Administration fédérale")
    # always wins the alternation and the longer, more specific term never
    # gets a chance to match.
    patterns = []
    for k in sorted(keywords, key=len, reverse=True):
        if k.isupper() and len(k) <= 4:
            patterns.append(rf'\b{re.escape(k)}\b')
        else:
            patterns.append(re.escape(k))
    return re.compile('|'.join(patterns), flags=re.IGNORECASE)


def build_keyword_maps(kw_all, entity_groups):
    """Map each lower-cased alias to (group_id, canonical spelling).

    Aliases not part of any entity_groups tuple (DE_TERMS/FR_TERMS) become
    their own singleton group, keyed by themselves.
    """
    alias_to_group = {}
    alias_canonical = {}
    for gid, group in enumerate(entity_groups):
        for alias in group:
            key = alias.lower()
            alias_to_group[key] = gid
            alias_canonical[key] = alias

    next_gid = len(entity_groups)
    for kw in kw_all:
        key = kw.lower()
        if key not in alias_to_group:
            alias_to_group[key] = next_gid
            alias_canonical[key] = kw
            next_gid += 1

    return alias_to_group, alias_canonical


def resolve_keywords(sent: str, kw_pattern, alias_to_group, alias_canonical) -> list[str]:
    """Return the canonical keyword(s) matched in `sent`, one per distinct entity."""
    raw_matches = [m for m in kw_pattern.findall(sent) if isinstance(m, str)]
    if not raw_matches:
        return []

    by_group: dict[int, set[str]] = {}
    for raw in raw_matches:
        key = raw.lower()
        gid = alias_to_group.get(key, key)
        canonical = alias_canonical.get(key, raw)
        by_group.setdefault(gid, set()).add(canonical)

    # Within a group, keep only the longest (most spelled-out) alias found.
    return sorted(max(aliases, key=len) for aliases in by_group.values())


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


def _split_on_punctuation(t: str) -> list[str]:
    t = _multidot_re.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), t)
    t = _abbrev_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _ordinal_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _initial_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    parts = re.split(r'(?<=[.!?])\s+', t)
    return [p.replace(_PLACEHOLDER, '.').strip() for p in parts if p.strip()]


def split_sentences(text: str) -> list[str]:
    if not isinstance(text, str) or not text.strip():
        return []
    # Line breaks (paragraph breaks, bullet/link lists, ...) are always hard
    # boundaries. Without this, a block of newline-separated list items with
    # no terminal punctuation gets glued by the punctuation-only split below
    # into one giant fake "sentence" the first time a period/! shows up.
    sentences = []
    for line in text.split('\n'):
        line = line.strip()
        if not line:
            continue
        sentences.extend(_split_on_punctuation(line))
    return sentences


def process_chunk(chunk: pd.DataFrame, text_col: str, kw_pattern, alias_to_group, alias_canonical) -> list[dict]:
    """Process one chunk of articles, return list of sentence dicts. Never holds exploded df in memory.

    One row per (sentence, canonical keyword) pair: a sentence matching
    several distinct entities is duplicated once per entity; a sentence
    matching several aliases of the same entity produces a single row.
    """
    meta_cols = [c for c in chunk.columns if c != text_col]
    rows = []
    for _, article in chunk.iterrows():
        sentences = split_sentences(article[text_col])
        for sent in sentences:
            sent = sent.strip()
            if not sent or sent == "nan":
                continue
            keywords = resolve_keywords(sent, kw_pattern, alias_to_group, alias_canonical)
            for keyword in keywords:
                kw_type, kw_sub_type, kw_abbrev = classify_keyword(keyword)
                row = {c: article[c] for c in meta_cols}
                row["sentence"] = sent
                row["matched_keywords"] = keyword
                row["type"] = kw_type
                row["sub_type"] = kw_sub_type
                row["keyword_abbrev"] = kw_abbrev
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cut articles into keyword-filtered sentences for 04_classify_stance.py"
    )
    ap.add_argument("--input", required=True,
                    help="Article pipeline output (.csv or .parquet)")
    ap.add_argument("--output_base", required=True,
                    help="Output base path for criticism_base (.csv is appended)")
    ap.add_argument("--text_col", default=None,
                    help="Column to split into sentences. Auto-detected if omitted (prefers 'text', falls back to 'lead').")
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

    n_total = len(df_full)

    kw_pattern = build_kw_pattern(KW_SENT)
    alias_to_group, alias_canonical = build_keyword_maps(KW_SENT, ENTITY_GROUPS)

    # ── Traitement par chunks + écriture incrémentale ─────────────────────────
    output_path = Path(f"{args.output_base}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentence_id = 1
    total_sentences = 0
    first_chunk = True

    for chunk_start in range(0, n_total, args.chunk_size):
        chunk = df_full.iloc[chunk_start:chunk_start + args.chunk_size]
        rows = process_chunk(chunk, text_col, kw_pattern, alias_to_group, alias_canonical)

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
