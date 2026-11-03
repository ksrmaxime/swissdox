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
(liste complète dans src/keywords_list.py).
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
    GENERIC_TERM_FORMS, FEDERAL_KEYWORD_FORMS,
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

    for canonical, surfaces in GENERIC_TERM_FORMS.items():
        for surface in surfaces:
            surface_to_canonical[norm_key(surface)] = canonical
            entries.append((surface, f"(?i:{_to_regex(surface)})"))

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
        if norm_key(canonical) in _IGNORED_KEYS:
            continue   # le terme est reconnu (il "consomme" le texte) mais ne crée pas de ligne
        by_group.setdefault(keyword_group(canonical), set()).add(canonical)

    # Dans une même entité, on garde l'alias le plus long (le plus explicite).
    return sorted(max(aliases, key=len) for aliases in by_group.values())


# ── Mots-clés ignorés au sentence cutting ────────────────────────────────────
# Ces termes restent dans la requête Swissdox (step 1) mais ne créent aucune
# ligne ici, car ils ne permettent pas d'identifier une unité fédérale :
#   - Secrétariat général / Generalsekretariat : nom complet commun aux 7
#     départements (les abréviations SG-DFAE, GS-EDA, ... restent actives) ;
#   - Secrétariat d'Etat / Staatssekretariat / SEE / STS : noms et abréviations
#     alternatifs du Secrétariat d'Etat du DFAE (les noms complets et
#     SEE-DFAE / STS-EDA restent actifs) ;
#   - unités de l'armée (OLD), dans les deux langues.
# Les comparaisons se font sur la forme de référence (toutes les déclinaisons
# et les deux apostrophes sont donc ignorées aussi).
STEP2_IGNORED_KEYWORDS = [
    "Secrétariat général", "Generalsekretariat",
    "Secrétariat d'Etat", "Staatssekretariat", "SEE", "STS",
    "Forces aériennes", "Luftwaffe",
    "Forces terrestres", "Heer",
    "Etat-major général", "Generalstab",
]
_IGNORED_KEYS = {norm_key(k) for k in STEP2_IGNORED_KEYWORDS}
_unknown_ignored = _IGNORED_KEYS - set(FEDERAL_KEYWORD_FORMS)
if _unknown_ignored:
    raise AssertionError(f"STEP2_IGNORED_KEYWORDS absents de keywords_list.py : {_unknown_ignored}")


# ── Légendes et crédits photo ────────────────────────────────────────────────
# 1) retirés À L'INTÉRIEUR des phrases (avant la recherche des mots-clés, pour
#    qu'un crédit "Name / EPA" ne déclenche pas le mot-clé EPA) :
#      "(Bild: Name / EPA)", "© EPA/Name ...", "Name Name / EPA / Keystone", "via EPA"
# 2) phrase entière supprimée si ce n'est qu'une légende / ligne de source :
#      commence par "Bild:", "Foto:", "Photo:", "Quelle:", "Source:" ... (<= 15 mots)
#      ou contient "/" en <= 8 mots sans ponctuation finale ("Alexander Kühni / VBS")
_PHOTO_LBL = r"(?:Bild|Bilder|Foto|Fotos|Photo|Photos|Image|Images|Illustration|Grafik|Karte)"
_LINE_LBL = r"(?:Bild|Bilder|Foto|Fotos|Photo|Photos|Image|Images|Illustration|Grafik|Karte|Quelle|Quellen|Source|Sources)"
_AGENCY = r"(?:EPA|Keystone|KEYSTONE|Keytone|Reuters|REUTERS|AFP|dpa|DPA|Imago|IMAGO|AP|Getty(?:\s+Images)?|freshfocus|Freshfocus)"
_W = r"[^\s/()]+"
_PAREN_CREDIT = re.compile(r"\(\s*" + _PHOTO_LBL + r"\s*:[^)]*(?:\)|$)", re.I)
_COPYRIGHT = re.compile(r"©.*$")
_AGENCY_CHAIN = re.compile(
    r"(?:" + _W + r"\s+){0,2}" + _W + r"\s*/\s*" + _AGENCY + r"(?:\s*/\s*" + _W + r")*"
    r"|" + _AGENCY + r"\s*/\s*" + _W + r"(?:\s*/\s*" + _W + r")*"
    r"|\bvia\s+" + _AGENCY + r"\b|\b" + _AGENCY + r"\s+via\b"
)
_LEAD_LABEL = re.compile(r"^[\s–—\-(]*" + _LINE_LBL + r"\s*:", re.I)


def strip_credits(sent: str) -> str:
    sent = _PAREN_CREDIT.sub(" ", sent)
    sent = _COPYRIGHT.sub(" ", sent)
    sent = _AGENCY_CHAIN.sub(" ", sent)
    return re.sub(r"\s+", " ", sent).strip()


def is_caption(sent: str) -> bool:
    core = sent.strip(" ()[]–—-.,;:➤■")
    if not core:
        return True
    n_words = len(core.split())
    if _LEAD_LABEL.match(sent) and n_words <= 15:
        return True
    if "/" in core and n_words <= 8 and not re.search(r"[.!?]$", sent.rstrip(" )")):
        return True
    return False


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


def _protect_ordinal(m: re.Match) -> str:
    """'3. Juli' : le point n'est pas une fin de phrase (ordinal allemand).
    Exception : une année (1800-2099) suivie d'un point termine la phrase
    ('... im Jahr 2014. Der Bundesrat ...')."""
    num = m.group(1)
    if len(num) == 4 and 1800 <= int(num) <= 2099:
        return m.group(0)
    return num + _PLACEHOLDER + m.group(2)
_initial_re  = re.compile(r'\b([A-ZÜÄÖ])\.(\s+[A-ZÜÄÖÉÀÂÊ])')
_multidot_re = re.compile(r'\b([a-zA-Züäö]\.)+([a-zA-Züäö])\.(?=\s)')


def _split_on_punctuation(t: str) -> list[str]:
    t = _multidot_re.sub(lambda m: m.group(0).replace('.', _PLACEHOLDER), t)
    t = _abbrev_re.sub(lambda m: m.group(1) + _PLACEHOLDER + m.group(2), t)
    t = _ordinal_re.sub(_protect_ordinal, t)
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


def process_chunk(chunk: pd.DataFrame, text_col: str, kw_pattern, surface_to_canonical,
                  journal_col, seen: set, stats: dict) -> list[dict]:
    """Process one chunk of articles, return list of sentence dicts. Never holds exploded df in memory.

    One row per (sentence, canonical keyword) pair: a sentence matching
    several distinct entities is duplicated once per entity; a sentence
    matching several aliases of the same entity produces a single row.

    - crédits photo retirés de la phrase, légendes supprimées (strip_credits / is_caption)
    - doublons : une phrase exactement identique, dans le même journal
      (même medium_code : NZZ et NZZO sont deux journaux distincts) et pour le
      même mot-clé n'est gardée qu'une fois (la 1re rencontrée). `seen` est
      partagé entre les chunks.
    """
    meta_cols = [c for c in chunk.columns if c != text_col]
    rows = []
    for _, article in chunk.iterrows():
        sentences = split_sentences(article[text_col])
        for sent in sentences:
            sent = strip_credits(sent)
            if not sent or sent == "nan":
                continue
            if is_caption(sent):
                stats["captions"] += 1
                continue
            keywords = resolve_keywords(sent, kw_pattern, surface_to_canonical)
            journal = article[journal_col] if journal_col else None
            for keyword in keywords:
                key = (sent, journal, keyword)
                if key in seen:
                    stats["duplicates"] += 1
                    continue
                seen.add(key)
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

    # Doublons : identité du journal = medium_code (version papier et online ont
    # des codes différents, ex. NZZ / NZZO). Articles triés par date pour garder
    # la 1re parution.
    journal_col = "medium_code" if "medium_code" in df_full.columns else (
        "medium_name" if "medium_name" in df_full.columns else None)
    if journal_col is None:
        print("[sentence_cutting] [WARN] ni medium_code ni medium_name : doublons non filtrés par journal")
    if "pubtime" in df_full.columns:
        df_full = df_full.sort_values("pubtime", kind="stable", na_position="last").reset_index(drop=True)
    seen: set = set()
    stats = {"captions": 0, "duplicates": 0}

    # ── Traitement par chunks + écriture incrémentale ─────────────────────────
    output_path = Path(f"{args.output_base}.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    sentence_id = 1
    total_sentences = 0
    first_chunk = True

    for chunk_start in range(0, n_total, args.chunk_size):
        chunk = df_full.iloc[chunk_start:chunk_start + args.chunk_size]
        rows = process_chunk(chunk, text_col, kw_pattern, surface_to_canonical, journal_col, seen, stats)

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
    print(f"[sentence_cutting] Légendes supprimées : {stats['captions']:,} | "
          f"doublons supprimés (même phrase, même journal) : {stats['duplicates']:,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
