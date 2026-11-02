# step2_sentence_cutting.py
"""
Sentence extraction step between sbatch_step1_download.sh and sbatch_step3a_stance_array.sh.

Input  : output of step1_download.py (.csv or .parquet)
           must have a text column (default 'text', fallback 'lead')
Output : criticism_base.csv ready to be consumed by step3a_stance_array.py
           columns: sentence_id, sentence, matched_keywords, type, sub_type,
           keyword_abbrev, unit_level, unit_status, department_en,
           unit_name_en, unit_abbrev_en, + all article metadata

One output row = one sentence + one canonical keyword. A sentence that
matches several *distinct* entities (e.g. "EDA" and "Direktion für
Völkerrecht") is duplicated, once per entity — that duplication is expected
and is what makes the file usable for per-entity stance classification
downstream. A sentence that matches several *aliases of the same entity*
(e.g. "SECO" and "Staatssekretariat für Wirtschaft", or two declensions of the
same name) produces a single row, keeping only the longest alias.

Keywords are exactly the ones used to build the Swissdox query
(src/download_src.py + src/federal_units_table.py).
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
    GENERIC_TERMS, FEDERAL_KEYWORD_FORMS,
    classify_keyword, keyword_group, keyword_unit_metadata, norm_key,
)

# ── Mots-clés : mêmes listes que la requête Swissdox (src/download_src.py) ───
#   - termes génériques DE/FR : insensibles à la casse, sans limite de mot
#     (comportement historique, ex. "Bürokratieabbau" est capté)
#   - noms d'unités fédérales (toutes déclinaisons) : insensibles à la casse,
#     mot entier
#   - abréviations en MAJUSCULES (EDA, SEM, DR, BAR, ...) : SENSIBLES à la casse,
#     mot entier — sinon "DR" capterait "Dr.", "BAR" "bar", "BIT" "bit", ...
#   - abréviations en casse mixte (Fedpol, swisstopo, armasuisse, ...) :
#     insensibles à la casse, mot entier
#   - ' et ’ sont interchangeables partout.


def _to_regex(term: str) -> str:
    pat = re.escape(term).replace("\\ ", r"\s+")
    return re.sub(r"['’]", "['’]", pat)


def build_keyword_index():
    """Retourne (pattern compilé, dict norm_key(surface) -> forme canonique)."""
    entries = []            # (surface, regex)
    surface_to_canonical = {}

    for term in GENERIC_TERMS:
        surface_to_canonical[norm_key(term)] = term
        entries.append((term, f"(?i:{_to_regex(term)})"))

    for kf in FEDERAL_KEYWORD_FORMS.values():
        for surface in kf.surfaces:
            surface_to_canonical[norm_key(surface)] = kf.canonical
            core = rf"(?<!\w){_to_regex(surface)}(?!\w)"
            case_sensitive = kf.is_abbrev and surface.upper() == surface
            entries.append((surface, core if case_sensitive else f"(?i:{core})"))

    # Plus long d'abord : "Administration fédérale des finances" doit l'emporter
    # sur "Administration fédérale".
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    pattern = re.compile("|".join(rx for _, rx in entries))
    return pattern, surface_to_canonical


def resolve_keywords(sent: str, kw_pattern, surface_to_canonical) -> list[str]:
    """Mot(s)-clé(s) canonique(s) trouvés dans `sent`, un par entité distincte."""
    by_group: dict[tuple, set[str]] = {}
    for m in kw_pattern.finditer(sent):
        canonical = surface_to_canonical.get(norm_key(m.group(0)), m.group(0))
        by_group.setdefault(keyword_group(canonical), set()).add(canonical)

    # Dans une même entité, on garde l'alias le plus long (le plus explicite).
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


def process_chunk(chunk: pd.DataFrame, text_col: str, kw_pattern, surface_to_canonical) -> list[dict]:
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
            keywords = resolve_keywords(sent, kw_pattern, surface_to_canonical)
            for keyword in keywords:
                kw_type, kw_sub_type, kw_abbrev = classify_keyword(keyword)
                row = {c: article[c] for c in meta_cols}
                row["sentence"] = sent
                row["matched_keywords"] = keyword
                row["type"] = kw_type
                row["sub_type"] = kw_sub_type
                row["keyword_abbrev"] = kw_abbrev
                row.update(keyword_unit_metadata(keyword))
                rows.append(row)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Cut articles into keyword-filtered sentences for step3a_stance_array.py"
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

    kw_pattern, surface_to_canonical = build_keyword_index()

    # ── Traitement par chunks + écriture incrémentale ─────────────────────────
    output_path = Path(f"{args.output_base}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentence_id = 1
    total_sentences = 0
    first_chunk = True

    for chunk_start in range(0, n_total, args.chunk_size):
        chunk = df_full.iloc[chunk_start:chunk_start + args.chunk_size]
        rows = process_chunk(chunk, text_col, kw_pattern, surface_to_canonical)

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
