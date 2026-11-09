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
import multiprocessing as mp
import os
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
#   - abréviations : SENSIBLES à la casse, mot entier. Seules les graphies
#     listées dans keywords_list.py sont acceptées (EDA ; SECO et Seco ;
#     Fedpol, fedpol et FEDPOL ...) — sinon "DR" capterait "Dr.", "BAR" "bar", ...
#   - abréviations ambiguës : règles de contexte (STEP2_ABBREV_RULES)
#   - ' et ’ sont interchangeables partout.


def _to_regex(term: str) -> str:
    pat = re.escape(term).replace("\\ ", r"\s+")
    return re.sub(r"['’]", "['’]", pat)


def build_keyword_index():
    """Retourne (pattern compilé, dict norm_key(surface) -> forme canonique)."""
    entries = []            # (surface, regex)
    surface_to_canonical = {}

    whole_word = {norm_key(t) for t in STEP2_GENERIC_WHOLE_WORD}
    for canonical, surfaces in GENERIC_TERM_FORMS.items():
        for surface in surfaces:
            surface_to_canonical[norm_key(surface)] = canonical
            core = _to_regex(surface)
            if norm_key(canonical) in whole_word:
                core = rf"(?<!\w){core}(?!\w)"
            entries.append((surface, f"(?i:{core})"))

    for kf in FEDERAL_KEYWORD_FORMS.values():
        for surface in kf.surfaces:
            surface_to_canonical[norm_key(surface)] = kf.canonical
            core = rf"(?<!\w){_to_regex(surface)}(?!\w)"
            case_sensitive = kf.is_abbrev   # chaque graphie d'abréviation est listée explicitement
            entries.append((surface, core if case_sensitive else f"(?i:{core})"))

    # Plus long d'abord : "Administration fédérale des finances" doit l'emporter
    # sur "Administration fédérale".
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    pattern = re.compile("|".join(rx for _, rx in entries))
    return pattern, surface_to_canonical


def resolve_keywords(sent: str, kw_pattern, surface_to_canonical, lang=None) -> list[str]:
    """Mot(s)-clé(s) canonique(s) trouvés dans `sent`, un par entité distincte."""
    by_group: dict[tuple, dict[str, int]] = {}
    for m in kw_pattern.finditer(sent):
        canonical = surface_to_canonical.get(norm_key(m.group(0)), m.group(0))
        if norm_key(canonical) in _IGNORED_KEYS:
            continue   # le terme est reconnu (il "consomme" le texte) mais ne crée pas de ligne
        if not _passes_context(canonical, sent, lang):
            continue   # abréviation ambiguë hors contexte (STEP2_ABBREV_RULES)
        group = by_group.setdefault(keyword_group(canonical), {})
        group.setdefault(canonical, m.start())   # position de la 1re occurrence

    # Dans une même entité, on garde l'alias le plus long (le plus explicite) ;
    # à longueur égale (ex. « BAZL-OFAC »), celui qui apparaît en premier.
    # Ce départage rend le résultat identique d'un run à l'autre.
    return sorted(
        max(aliases, key=lambda a: (len(a), -aliases[a])) for aliases in by_group.values()
    )


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


# ── Règles de contexte pour les abréviations ambiguës ────────────────────────
# Une abréviation listée ici n'est retenue que si la phrase remplit la règle.
#   "require" : la phrase doit contenir un de ces mots (sinon : ignorée).
#               Pour les abréviations presque jamais utilisées pour l'unité
#               fédérale dans la presse (DC = Washington DC / DC Comics / DC-10,
#               DR = crédit photo « droits réservés » / TV danoise, EPA = agence
#               US / oméga-3, ...).
#   "exclude" : la phrase ne doit contenir aucun de ces motifs.
#   "lang"    : l'abréviation n'est retenue que dans les articles de cette
#               langue (ex. BBL = abréviation allemande ; en français, « BBL »
#               désigne presque toujours le Béjart Ballet Lausanne).
# Clé = abréviation de référence (les variantes « Seco », « Astra » ... suivent
# la règle de leur abréviation). (?i:...) = insensible à la casse.
STEP2_ABBREV_RULES = {
    "DC":    {"require": r"(?i:consulaire|konsular|Auslandschweizer|Suisses? de l.étranger|\bEDA\b|\bDFAE\b)"},
    "DR":    {"require": r"(?i:ressources|Ressourcen|Aussennetz|réseau extérieur|\bEDA\b|\bDFAE\b)"},
    "DV":    {"require": r"(?i:Völkerrecht|droit international|\bEDA\b|\bDFAE\b)"},
    "KD":    {"require": r"(?i:konsular|consulaire|Auslandschweizer|\bEDA\b)"},
    "OA":    {"require": r"(?i:Oberauditor|auditeur en chef|Militärjustiz|justice militaire)"},
    "AFS":   {"require": r"(?i:Archiv|archives)"},
    "EPA":   {"require": r"(?i:Personalamt|Bundespersonal|personnel de la Confédération|Bundesangestellt|Lohn|Löhne|salaire|\bEFD\b|\bDFF\b)"},
    "AFC":   {"lang": "fr",
              "exclude": r"AFC\s+[A-Z][a-z]|(?i:Sunderland|Bournemouth|Ajax|Wimbledon|Fussball|football|Fifa|Uefa|asiati|Asien|Champions League|Ligue des champions|Asian Cup|M23|Congo|Kongo|America First|cantonal)"},
    "BIT":   {"lang": "de", "exclude": r"(?i:travail|chômage|Arbeitsorganisation|\bILO\b|\bOIT\b|Bureau international)"},
    "BWL":   {"lang": "de", "exclude": r"BWL-|VWL|(?i:studi|Betriebswirtschaft|Universität|Hochschule|Bachelor|Master)"},
    "BBL":   {"lang": "de", "exclude": r"(?i:Béjart|Ballet|Butt|Brazilian)"},
    "ARE":   {"exclude": r"(?i:kantonal|\bKanton|cantonal|\bcanton)"},
    "ASTRA": {"exclude": r"(?i:Opel|Vauxhall|Satellit|satellite|\bSES\b|Zeneca)"},
    # Ajouts après le run complet 2000-2025 :
    "BAR":   {"require": r"(?i:archiv|Akten|Dossier|Dokument|document|Bundesarchiv)"},   # sinon : écurie de F1 BAR Honda, Barbade (BAR), bars
    "BFM":   {"exclude": r"BFM\s*TV|BFMTV|BFM\s+(?:Business|Paris|Lyon)|«BFM|(?i:chaîne|télévision)"},   # chaîne française BFM TV
    "BAK":   {"exclude": r"\b(?:[Dd]ie|der)\s+BAK\b(?!-)|BAK\s*(?:Basel|Economics)|BAKBASEL|BAK-Konjunktur|(?i:Konjunktur|Ecoplan|Forschungsinstitut|Wirtschaftsforsch|institut de recherche|recherches économiques|bâlois|BAK-Experten|BAK-Ökonom|Prognose|prévision|prophezei|Ökonom|économiste|Wachstum|croissance|\bBIP\b|\bPIB\b)"},   # institut BAK Economics (« die BAK » ; l'office est « das BAK »)
    "BSV":   {"exclude": r"BSV\s+(?:Bern|Muri|Stans|Wacker|Kriens|Borba)|(?i:Handball)|\b\d{1,2}:\d{1,2}\b"},   # club de handball BSV Bern
    "METAS": {"exclude": r"Metas\s+[A-ZÄÖÜ]|(?i:Zuckerberg|Facebook|Instagram|WhatsApp)"},   # « Metas Llama » = génitif de Meta
}
_ABBREV_RULES = {
    norm_key(k): {kind: (rx if kind == "lang" else re.compile(rx)) for kind, rx in rule.items()}
    for k, rule in STEP2_ABBREV_RULES.items()
}
_unknown_rules = set(_ABBREV_RULES) - set(FEDERAL_KEYWORD_FORMS)
if _unknown_rules:
    raise AssertionError(f"STEP2_ABBREV_RULES absents de keywords_list.py : {_unknown_rules}")


def _passes_context(canonical: str, sent: str, lang=None) -> bool:
    rule = _ABBREV_RULES.get(norm_key(canonical))
    if rule is None:
        return True
    if "lang" in rule and str(lang).strip().lower() != rule["lang"]:
        return False
    if "require" in rule and not rule["require"].search(sent):
        return False
    if "exclude" in rule and rule["exclude"].search(sent):
        return False
    return True


# ── Termes génériques exigeant un mot entier ──────────────────────────────────
# Par défaut, un terme générique est aussi trouvé à l'intérieur d'un mot composé
# (« Bürokratieabbau »). Pour ceux-ci, seul le mot seul compte : « Beamte » et
# « Beamten » oui, « Polizeibeamte », « Chefbeamten » … non.
STEP2_GENERIC_WHOLE_WORD = ["Beamte"]

# ── Phrases supprimées ────────────────────────────────────────────────────────
STEP2_MAX_SENTENCE_CHARS = 1000      # au-delà : tableaux de chiffres, code de graphiques
_HTML_ENTITY = re.compile(r"&(?:[A-Za-z]{2,8}|#[xX]?[0-9A-Fa-f]{1,6});")   # « &gt; », « &#x00A0; » … restés dans le texte

# ── Termes partagés entre deux départements : attribution selon la date ───────
# L'Office fédéral des assurances privées dépendait du DFJP jusqu'au 30 juin
# 2003, puis du DFF à partir du 1er juillet 2003. Format : liste de
# (date limite incluse "AAAA-MM-JJ", département), la 1re qui convient gagne.
STEP2_DATE_RULES = {
    "Bundesamt für Privatversicherungen":    [("2003-06-30", "EJPD/DFJP"), ("9999-12-31", "EFD/DFF")],
    "Office fédéral des assurances privées": [("2003-06-30", "EJPD/DFJP"), ("9999-12-31", "EFD/DFF")],
}
_DATE_RULES = {norm_key(k): v for k, v in STEP2_DATE_RULES.items()}
_unknown_dates = set(_DATE_RULES) - set(FEDERAL_KEYWORD_FORMS)
if _unknown_dates:
    raise AssertionError(f"STEP2_DATE_RULES absents de keywords_list.py : {_unknown_dates}")


def department_by_date(keyword: str, pubtime) -> str | None:
    rule = _DATE_RULES.get(norm_key(keyword))
    if rule is None or pubtime is None or str(pubtime) in ("", "nan", "NaT", "None"):
        return None
    day = str(pubtime)[:10]
    for limit, dept in rule:
        if day <= limit:
            return dept
    return None


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


def extract_chunk(chunk: pd.DataFrame, text_col: str, kw_pattern, surface_to_canonical) -> tuple[list[dict], dict]:
    """Découpe un paquet d'articles et retourne (lignes, compteurs des phrases supprimées).

    One row per (sentence, canonical keyword) pair: a sentence matching
    several distinct entities is duplicated once per entity; a sentence
    matching several aliases of the same entity produces a single row.
    Crédits photo retirés de la phrase, légendes supprimées (strip_credits / is_caption).
    Les doublons ne sont PAS filtrés ici (voir drop_duplicates_in_order) : cette
    fonction peut donc tourner en parallèle sur plusieurs CPU.
    """
    meta_cols = [c for c in chunk.columns if c != text_col]
    rows = []
    counts = {"captions": 0, "too_long": 0, "html_entity": 0}
    for _, article in chunk.iterrows():
        sentences = split_sentences(article[text_col])
        lang = article["language"] if "language" in article.index else None
        for sent in sentences:
            sent = strip_credits(sent)
            if not sent or sent == "nan":
                continue
            if is_caption(sent):
                counts["captions"] += 1
                continue
            keywords = resolve_keywords(sent, kw_pattern, surface_to_canonical, lang)
            if not keywords:
                continue
            if len(sent) > STEP2_MAX_SENTENCE_CHARS:
                counts["too_long"] += 1
                continue
            if _HTML_ENTITY.search(sent):
                counts["html_entity"] += 1
                continue
            for keyword in keywords:
                kw_type, kw_sub_type, kw_abbrev = classify_keyword(keyword)
                dept = None
                if kw_sub_type == "ambiguous":
                    dept = department_by_date(keyword, article.get("pubtime"))
                    if dept:
                        kw_sub_type = dept
                row = {c: article[c] for c in meta_cols}
                row["sentence"] = sent
                row["matched_keywords"] = keyword
                row["type"] = kw_type
                row["sub_type"] = kw_sub_type
                row["keyword_abbrev"] = kw_abbrev
                row.update(keyword_unit_metadata(keyword, department=dept))
                rows.append(row)
    return rows, counts


def drop_duplicates_in_order(rows: list[dict], journal_col, seen: set) -> tuple[list[dict], int]:
    """Doublons : une phrase exactement identique, dans le même journal (même
    medium_code : NZZ et NZZO sont deux journaux distincts) et pour le même
    mot-clé n'est gardée qu'une fois (la 1re rencontrée, les articles étant
    traités par date). `seen` est partagé entre tous les paquets."""
    kept, n_dup = [], 0
    for row in rows:
        key = (row["sentence"], row.get(journal_col) if journal_col else None, row["matched_keywords"])
        if key in seen:
            n_dup += 1
            continue
        seen.add(key)
        kept.append(row)
    return kept, n_dup


def process_chunk(chunk: pd.DataFrame, text_col: str, kw_pattern, surface_to_canonical,
                  journal_col, seen: set, stats: dict) -> list[dict]:
    """Version en un seul appel (extraction + doublons), conservée pour les tests."""
    rows, counts = extract_chunk(chunk, text_col, kw_pattern, surface_to_canonical)
    rows, n_dup = drop_duplicates_in_order(rows, journal_col, seen)
    for k, v in counts.items():
        stats[k] = stats.get(k, 0) + v
    stats["duplicates"] += n_dup
    return rows


# ── Parallélisation : chaque processus construit son propre index de mots-clés ─
_WORKER_INDEX = None


def _init_worker():
    global _WORKER_INDEX
    _WORKER_INDEX = build_keyword_index()


def _extract_in_worker(job):
    chunk, text_col = job
    pattern, mapping = _WORKER_INDEX
    return extract_chunk(chunk, text_col, pattern, mapping)


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
    ap.add_argument("--workers", type=int,
                    default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
                    help="Nombre de processus en parallèle (défaut : SLURM_CPUS_PER_TASK, sinon 1)")
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

    # Les paquets sont découpés en parallèle, mais leurs résultats arrivent DANS
    # L'ORDRE (imap) : doublons et sentence_id sont gérés ici, séquentiellement,
    # exactement comme en version mono-processus.
    starts = list(range(0, n_total, args.chunk_size))
    jobs = ((df_full.iloc[st:st + args.chunk_size], text_col) for st in starts)
    workers = max(1, args.workers)
    print(f"[sentence_cutting] {len(starts)} paquets de {args.chunk_size} articles, {workers} processus", flush=True)
    if workers > 1:
        pool = mp.Pool(workers, initializer=_init_worker)
        results = pool.imap(_extract_in_worker, jobs)
    else:
        pool = None
        results = (extract_chunk(c, t, kw_pattern, surface_to_canonical) for c, t in jobs)

    for chunk_start, (rows, counts) in zip(starts, results):
        for k, v in counts.items():
            stats[k] = stats.get(k, 0) + v
        rows, n_dup = drop_duplicates_in_order(rows, journal_col, seen)
        stats["duplicates"] += n_dup

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

    if pool is not None:
        pool.close()
        pool.join()
    print(f"[sentence_cutting] Saved {total_sentences:,} sentences → {output_path}")
    print(f"[sentence_cutting] Légendes supprimées : {stats.get('captions', 0):,} | "
          f"doublons supprimés (même phrase, même journal) : {stats['duplicates']:,} | "
          f"phrases > {STEP2_MAX_SENTENCE_CHARS} car. : {stats.get('too_long', 0):,} | "
          f"phrases avec entité HTML : {stats.get('html_entity', 0):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
