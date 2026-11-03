from __future__ import annotations

import html
import os
import re
import time
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import requests
import yaml
from dotenv import load_dotenv


# -----------------------------
# Swissdox config (query builder)
# -----------------------------
DEFAULT_API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
DEFAULT_COLUMNS = [
    "id", "pubtime", "medium_code", "medium_name", "rubric", "regional",
    "doctype", "doctype_description", "language", "char_count", "dateline",
    "head", "subhead", "content_id", "content",
]

# -----------------------------
# Mots-clés : tout est défini en toutes lettres dans src/keywords_list.py
# (génériques + unités fédérales, déclinaisons allemandes comprises).
# Ce module ne fait que :
#   - ajouter l'autre apostrophe (' <-> ’) aux termes qui en contiennent une ;
#   - indexer les formes pour la requête Swissdox et pour le step 2.
# -----------------------------
try:  # src/ sur le sys.path (pip install -e .)
    from keywords_list import GENERIC_KEYWORDS, FEDERAL_UNITS as _FEDERAL_UNITS_LIST  # noqa: E402
except ImportError:  # import via le package "src"
    from src.keywords_list import GENERIC_KEYWORDS, FEDERAL_UNITS as _FEDERAL_UNITS_LIST  # noqa: E402

_APOSTROPHES = ("'", "’")


def _norm_spaces(s: str) -> str:
    return re.sub(r"\s+", " ", s or "").strip()


def norm_key(s: str) -> str:
    """Clé de comparaison : minuscules, apostrophes unifiées, espaces normalisés."""
    return _norm_spaces(s).replace("’", "'").lower()


def apostrophe_variants(term: str) -> List[str]:
    if not any(a in term for a in _APOSTROPHES):
        return [term]
    base = term.replace("’", "'")
    return [base, base.replace("'", "’")]


# ── Génériques ────────────────────────────────────────────────────────────────
GENERIC_SUBTYPES: Dict[str, str] = {base: sub for base, (sub, _) in GENERIC_KEYWORDS.items()}
GENERIC_TERM_FORMS: Dict[str, List[str]] = {
    base: list(dict.fromkeys(a for f in forms for a in apostrophe_variants(_norm_spaces(f))))
    for base, (_, forms) in GENERIC_KEYWORDS.items()
}
GENERIC_TERMS: List[str] = list(GENERIC_TERM_FORMS)            # formes de base
GENERIC_QUERY_TERMS: List[str] = [v for vs in GENERIC_TERM_FORMS.values() for v in vs]


# ── Unités fédérales ──────────────────────────────────────────────────────────
@dataclass
class FederalUnit:
    unit_id: int
    level: str            # "department" | "office"
    status: str           # "current" | "OLD"
    department_en: str    # code EN du département parent (FDFA, FDHA, ...)
    department: str       # label DE/FR du département parent (ex. "EDA/DFAE"), format historique de sub_type
    name_en: str
    abbrev_en: str
    abbrev_fr: str
    abbrev_de: str


@dataclass
class KeywordForm:
    canonical: str            # forme de référence (1re forme du nom)
    is_abbrev: bool
    lang: str                 # "fr" | "de"
    unit_ids: set             # unités désignées par ce terme (plusieurs si terme partagé)
    surfaces: set             # toutes les formes à chercher (déclinaisons + apostrophes)


def _build_federal_units():
    dept_label = {
        u["upper_level"]: f'{u["de_abbrev"]}/{u["fr_abbrev"]}'
        for u in _FEDERAL_UNITS_LIST if u["type"] == "Federal Department"
    }
    units: List[FederalUnit] = []
    forms: Dict[str, KeywordForm] = {}   # clé = norm_key(canonical)

    def add(variants: List[str], *, is_abbrev: bool, lang: str, uid: int) -> None:
        variants = [_norm_spaces(v) for v in variants if _norm_spaces(v)]
        if not variants:
            return
        canonical = variants[0]
        kf = forms.get(norm_key(canonical))
        if kf is None:
            kf = forms[norm_key(canonical)] = KeywordForm(canonical, is_abbrev, lang, set(), set())
        kf.unit_ids.add(uid)
        for v in variants:
            kf.surfaces.update(apostrophe_variants(v))

    for uid, u in enumerate(_FEDERAL_UNITS_LIST):
        units.append(FederalUnit(
            unit_id=uid,
            level="department" if "department" in u["type"].lower() else "office",
            status="OLD" if u["type"].upper().startswith("OLD") else "current",
            department_en=u["upper_level"],
            department=dept_label.get(u["upper_level"], u["upper_level"]),
            name_en=u["en_name"], abbrev_en=u["en_abbrev"],
            abbrev_fr=u["fr_abbrev"], abbrev_de=u["de_abbrev"],
        ))
        for variants in u["fr_names"]:          # [forme de référence, variantes casse/accents...]
            add(variants, is_abbrev=False, lang="fr", uid=uid)
        for declined in u["de_names"]:          # [forme de référence, déclinaisons...]
            add(declined, is_abbrev=False, lang="de", uid=uid)
        case_variants = u.get("abbrev_case_variants", {})
        for lang in ("fr", "de"):
            for ab in [u[f"{lang}_abbrev"]] + u[f"{lang}_alt_abbrevs"]:
                if ab:
                    add([ab] + case_variants.get(ab, []), is_abbrev=True, lang=lang, uid=uid)
    return units, forms


FEDERAL_UNITS, FEDERAL_KEYWORD_FORMS = _build_federal_units()
_UNIT_BY_ID = {u.unit_id: u for u in FEDERAL_UNITS}

# Liste plate de tous les termes spécifiques envoyés à Swissdox (dédoublonnée, ordre stable)
FEDERAL_UNIT_TERMS: List[str] = list(dict.fromkeys(
    s for kf in FEDERAL_KEYWORD_FORMS.values() for s in sorted(kf.surfaces)
))


# -----------------------------
# Classification d'un mot-clé (utilisée par scripts/step2_sentence_cutting.py)
# -----------------------------
def keyword_group(canonical: str):
    """Identifiant d'entité : deux alias de la même unité ont le même groupe."""
    kf = FEDERAL_KEYWORD_FORMS.get(norm_key(canonical))
    if kf is not None:
        return ("unit",) + tuple(sorted(kf.unit_ids))
    return ("generic", norm_key(canonical))


def classify_keyword(keyword: str) -> tuple[str, str, str]:
    """Return (type, sub_type, abbrev) for a canonical keyword string.

    type    : "generic" (GENERIC_KEYWORDS) or "specific" (unité fédérale)
    sub_type: "Public Admin", "bureaucracy", le label du département parent
              (ex. "VBS/DDPS"), ou "ambiguous" si le terme désigne plusieurs
              unités de départements différents (ex. "Generalsekretariat")
    abbrev  : abréviation de l'unité dans la langue du terme (sinon l'autre
              langue, sinon le terme lui-même)
    """
    kf = FEDERAL_KEYWORD_FORMS.get(norm_key(keyword))
    if kf is not None:
        units = [_UNIT_BY_ID[i] for i in sorted(kf.unit_ids)]
        depts = sorted({u.department for u in units})
        sub_type = depts[0] if len(depts) == 1 else "ambiguous"
        if kf.is_abbrev:
            abbrev = kf.canonical
        elif len(units) == 1:
            u = units[0]
            first, second = (u.abbrev_fr, u.abbrev_de) if kf.lang == "fr" else (u.abbrev_de, u.abbrev_fr)
            abbrev = first or second or kf.canonical
        else:
            abbrev = kf.canonical
        return "specific", sub_type, abbrev
    if keyword in GENERIC_SUBTYPES:
        return "generic", GENERIC_SUBTYPES[keyword], keyword
    return "unknown", "unknown", keyword


def keyword_unit_metadata(keyword: str) -> Dict[str, Any]:
    """Métadonnées d'analyse (dont colonnes EN) de l'unité désignée par le mot-clé.

    Si le terme désigne plusieurs unités, les valeurs sont jointes par " | ".
    Pour un terme générique, toutes les valeurs sont vides.
    """
    kf = FEDERAL_KEYWORD_FORMS.get(norm_key(keyword))
    keys = ["unit_level", "unit_status", "department_en", "unit_name_en", "unit_abbrev_en"]
    if kf is None:
        return {k: pd.NA for k in keys}
    units = [_UNIT_BY_ID[i] for i in sorted(kf.unit_ids)]

    def join(vals):
        vals = list(dict.fromkeys(v for v in vals if v))
        return " | ".join(vals) if vals else pd.NA

    return {
        "unit_level": join(u.level for u in units),
        "unit_status": join(u.status for u in units),
        "department_en": join(u.department_en for u in units),
        "unit_name_en": join(u.name_en for u in units),
        "unit_abbrev_en": join(u.abbrev_en for u in units),
    }


def build_query_payload(
    *,
    start_date: str,
    end_date: str,
    languages: List[str],
    sources: List[str],
    max_results: int,
    columns: List[str],
    query_name: str,
    comment: str,
    expiration_date: str,
    version: str = "1.2",
) -> Dict[str, Any]:
    query_block = {
        "sources": sources,
        "dates": [{"from": start_date, "to": end_date}],
        "languages": languages,
        "content": {
            "OR": [
                {"OR": GENERIC_QUERY_TERMS},
                {"OR": FEDERAL_UNIT_TERMS},
            ]
        },
    }
    yaml_payload = {
        "query": query_block,
        "result": {"format": "TSV", "maxResults": max_results, "columns": columns},
        "version": version,
    }
    return {
        "yaml_payload": yaml_payload,
        "meta": {"name": query_name, "comment": comment, "expirationDate": expiration_date},
    }


# -----------------------------
# Swissdox API client
# -----------------------------
@dataclass
class SwissdoxClient:
    api_key: str
    api_secret: str
    base_url: str = DEFAULT_API_BASE_URL
    timeout: int = 120

    def __post_init__(self) -> None:
        self.query_url = f"{self.base_url}/query"
        self.status_url = f"{self.base_url}/status"
        self.headers = {"X-API-Key": self.api_key, "X-API-Secret": self.api_secret}
        self.session = requests.Session()

    def submit_query(
        self,
        yaml_payload: Dict[str, Any],
        *,
        name: str,
        comment: str,
        expiration_date: str,
        test: bool = False,
    ) -> str:
        yaml_query = yaml.safe_dump(yaml_payload, sort_keys=False, allow_unicode=True)
        data = {
            "query": yaml_query,
            "name": name,
            "comment": comment,
            "expirationDate": expiration_date,
        }
        if test:
            data["test"] = "1"

        r = self.session.post(self.query_url, headers=self.headers, data=data, timeout=self.timeout)
        if r.status_code >= 400:
            raise RuntimeError(f"Swissdox /query failed ({r.status_code}): {r.text[:2000]}")

        resp = r.json()
        if resp.get("result") != "ok":
            raise RuntimeError(f"Swissdox non-ok response: {resp}")

        qid = resp.get("queryId") or resp.get("id")
        if not qid:
            raise RuntimeError(f"Swissdox missing queryId: {resp}")
        return str(qid)

    def wait_for_download_url(
        self,
        query_id: str,
        *,
        max_wait_s: int = 12 * 60 * 60,
        poll_every_s: int = 30,
    ) -> str:
        deadline = time.time() + max_wait_s
        while time.time() < deadline:
            rs = self.session.get(self.status_url, headers=self.headers, timeout=self.timeout)
            rs.raise_for_status()
            status_list = rs.json()

            job = next((j for j in status_list if str(j.get("id")) == str(query_id)), None)
            if not job:
                time.sleep(poll_every_s)
                continue

            if job.get("error"):
                raise RuntimeError(f"Swissdox job error: {job['error']}")

            url = job.get("downloadUrl")
            if url:
                return self._normalize_download_url(url)

            time.sleep(poll_every_s)

        raise TimeoutError(f"Swissdox: no downloadUrl after {max_wait_s}s (query_id={query_id})")

    def download_tsv_xz(self, download_url: str) -> pd.DataFrame:
        r = self.session.get(download_url, headers=self.headers, timeout=self.timeout)
        r.raise_for_status()
        return pd.read_csv(BytesIO(r.content), sep="\t", compression="xz")

    def _normalize_download_url(self, download_url: str) -> str:
        if download_url.startswith("http"):
            return download_url
        if download_url.startswith("/"):
            return f"{self.base_url}{download_url}"
        return f"{self.base_url}/download/{download_url}"


# -----------------------------
# Text cleaning
# -----------------------------
_WS_RE = re.compile(r"[ \t\r\f\v]+")
_MULTI_NL_RE = re.compile(r"\n{2,}")


def clean_text(s: str) -> str:
    if not isinstance(s, str):
        return ""
    s = html.unescape(s)
    s = s.replace("\r", " ").replace("\n", " ").replace("\t", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s.strip(' "“”„\'')


def clean_xml_keep_paragraphs(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = html.unescape(s)

    s = re.sub(r"(?i)<br\s*/?>", "\n", s)
    s = re.sub(r"(?i)</p\s*>", "\n\n", s)
    s = re.sub(r"(?i)<p[^>]*>", "", s)
    s = re.sub(r"(?i)</div\s*>", "\n\n", s)
    s = re.sub(r"(?i)<div[^>]*>", "", s)

    s = re.sub(r"<[^>]+>", " ", s)

    lines = s.splitlines()
    cleaned_lines = []
    for line in lines:
        line = _WS_RE.sub(" ", line).strip()
        cleaned_lines.append(line)

    s = "\n".join(cleaned_lines)
    s = re.sub(r"\n[ \t]+", "\n", s)
    s = _MULTI_NL_RE.sub("\n\n", s).strip()

    return s


def extract_first_paragraph(s: str) -> str:
    if not isinstance(s, str):
        return ""

    s = s.strip()
    if not s:
        return ""

    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", s) if p.strip()]
    if paragraphs:
        return paragraphs[0]

    return clean_text(s)


def clean_articles_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"].astype(str), errors="coerce", utc=True).dt.date

    for c in ["medium_name", "rubric", "dateline", "head", "subhead"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)

    if "content" in df.columns:
        df["content"] = df["content"].apply(clean_xml_keep_paragraphs)

    return df


# -----------------------------
# Article-level output:
# one row = one article with title + lead + full text
# -----------------------------
def build_article_leads(df_articles: pd.DataFrame, *, content_col: str = "content") -> pd.DataFrame:
    if content_col not in df_articles.columns:
        raise ValueError(f"Missing '{content_col}' column")

    if "content_id" in df_articles.columns:
        article_id = df_articles["content_id"].astype(str)
    elif "id" in df_articles.columns:
        article_id = df_articles["id"].astype(str)
    else:
        article_id = df_articles.index.astype(str)

    title = df_articles["head"].fillna("").astype(str) if "head" in df_articles.columns else ""
    lead = df_articles[content_col].fillna("").astype(str).apply(extract_first_paragraph)
    text = df_articles[content_col].fillna("").astype(str)

    out = pd.DataFrame({
        "article_id": article_id,
        "title": title,
        "lead": lead,
        "text": text,
    })

    meta_cols = [
        "pubtime",
        "medium_code",
        "medium_name",
        "rubric",
        "regional",
        "doctype",
        "doctype_description",
        "language",
        "char_count",
        "dateline",
    ]

    for col in meta_cols:
        if col in df_articles.columns:
            out[col] = df_articles[col]

    return out


# -----------------------------
# Top-level pipeline
# -----------------------------
def run_pipeline(
    *,
    start_date: str,
    end_date: str,
    languages: List[str],
    sources: List[str],
    max_results: int,
    expiration_date: str,
    query_name: str,
    comment: str,
    out_dir: Path,
    test: bool = False,
    max_wait_s: int = 12 * 60 * 60,
) -> Dict[str, Path]:
    load_dotenv()
    api_key = os.getenv("SWISSDOX_API_KEY")
    api_secret = os.getenv("SWISSDOX_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing SWISSDOX_API_KEY / SWISSDOX_API_SECRET in .env")

    payload = build_query_payload(
        start_date=start_date,
        end_date=end_date,
        languages=languages,
        sources=sources,
        max_results=max_results,
        columns=DEFAULT_COLUMNS,
        query_name=query_name,
        comment=comment,
        expiration_date=expiration_date,
    )

    client = SwissdoxClient(api_key=api_key, api_secret=api_secret)
    qid = client.submit_query(
        payload["yaml_payload"],
        name=payload["meta"]["name"],
        comment=payload["meta"]["comment"],
        expiration_date=payload["meta"]["expirationDate"],
        test=test,
    )
    print(f"[Swissdox] queryId={qid}")

    url = client.wait_for_download_url(qid, max_wait_s=max_wait_s)
    print(f"[Swissdox] downloadUrl={url}")

    df_raw = client.download_tsv_xz(url)
    print(f"[Swissdox] raw shape={df_raw.shape}")

    df_articles = clean_articles_df(df_raw)
    print(f"[Swissdox] cleaned articles shape={df_articles.shape}")

    df_leads = build_article_leads(df_articles)
    print(f"[Swissdox] article-level output shape={df_leads.shape}")

    out_dir.mkdir(parents=True, exist_ok=True)

    p_articles_parquet = out_dir / "swissdox_articles_lead.parquet"
    p_articles_csv = out_dir / "swissdox_articles_lead.csv"

    df_leads.to_parquet(p_articles_parquet, index=False)
    df_leads.to_csv(p_articles_csv, index=False)

    return {
        "articles_lead_parquet": p_articles_parquet,
        "articles_lead_csv": p_articles_csv,
    }