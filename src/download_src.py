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
# Swissdox config (keywords + query builder)
# -----------------------------
DEFAULT_API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
DEFAULT_COLUMNS = [
    "id", "pubtime", "medium_code", "medium_name", "rubric", "regional",
    "doctype", "doctype_description", "language", "char_count", "dateline",
    "head", "subhead", "content_id", "content",
]

DE_TERMS = [
    "Bürokratie", "Bürokraten", "Verwaltung", "öffentlicher Dienst",
]
DE_LEVEL = ["Bund", "Bundes", "Kanton", "kantonal", "Schweiz"]

FR_TERMS = [
    "Bureaucratie", "Administration", "Administration fédérale", "Services publics", "Bureaucrates",
]
FR_LEVEL = ["fédéral", "federal", "federale", "cantonal", "cantonale", "Suisse"]

DEPARTMENTS = [
    "VBS", "DDPS", "Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport",
    "Département fédéral de la défense, de la protection de la population et des sports",
    "EDA", "DFAE", "Eidgenössische Departement für auswärtige Angelegenheiten",
    "Département fédéral des affaires étrangères",
    "UVEK", "DETEC", "Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    "EJPD", "DFJP", "Eidgenössische Justiz- und Polizeidepartement",
    "Département fédéral de justice et police",
    "EDI", "DFI", "Eidgenössische Departement des Innern", "Département fédéral de l'intérieur",
    "EFD", "DFF", "Eidgenössische Finanzdepartement", "Département fédéral des finances",
    "WBF", "DEFR", "Eidgenössische Departement für Wirtschaft, Bildung und Forschung",
    "Département fédéral de l'économie, de la formation et de la recherche",
]

# Administrative units of each department (FR + DE names and abbreviations)
# NOTE: 2-letter abbreviations (DV, DC, KD, DR, OA) and ambiguous short ones
# (BAR, BAG, BIT, BWL) are intentionally omitted — they are common standalone
# words in FR/DE press and would generate false positives.
ADMIN_UNITS = [
    # ── DFAE / EDA ──────────────────────────────────────────────────────────
    "SG-DFAE", "GS-EDA",
    "Direction du droit international public", "DDIP",
    "Direktion für Völkerrecht",
    "Direction consulaire",
    "Konsularische Direktion",
    "Direction du développement et de la coopération", "DDC",
    "Direktion für Entwicklung und Zusammenarbeit", "DEZA",
    "Direction des ressources",
    "Direktion für Ressourcen",
    # ── DFI / EDI ───────────────────────────────────────────────────────────
    "SG-DFI", "GS-EDI",
    "Bureau fédéral de l'égalité entre femmes et hommes", "BFEG",
    "Eidgenössisches Büro für die Gleichstellung von Frau und Mann", "EBG",
    "Office fédéral de la culture", "OFC",
    "Bundesamt für Kultur", "BAK",
    "Archives fédérales suisses", "AFS",
    "Schweizerisches Bundesarchiv",
    "Office fédéral de météorologie et de climatologie", "MétéoSuisse",
    "Bundesamt für Meteorologie und Klimatologie", "MeteoSchweiz",
    "Office fédéral de la santé publique", "OFSP",
    "Bundesamt für Gesundheit", "BAG",
    "Office fédéral de la sécurité alimentaire et des affaires vétérinaires", "OSAV",
    "Bundesamt für Lebensmittelsicherheit und Veterinärwesen", "BLV",
    "Office fédéral de la statistique", "OFS",
    "Bundesamt für Statistik", "BFS",
    "Office fédéral des assurances sociales", "OFAS",
    "Bundesamt für Sozialversicherungen", "BSV",
    # ── DFJP / EJPD ─────────────────────────────────────────────────────────
    "SG-DFJP", "GS-EJPD",
    "Secrétariat d'État aux migrations", "SEM",
    "Staatssekretariat für Migration",
    "Office fédéral de la justice", "OFJ",
    "Bundesamt für Justiz",
    "Office fédéral de la police", "fedpol",
    "Bundesamt für Polizei",
    "Service Surveillance de la correspondance par poste et télécommunication", "Service SCPT",
    "Dienst Überwachung Post- und Fernmeldeverkehr", "ÜPF",
    # ── DDPS / VBS ──────────────────────────────────────────────────────────
    "SG-DDPS", "GS-VBS",
    "Office fédéral de la protection de la population", "OFPP",
    "Bundesamt für Bevölkerungsschutz", "BABS",
    "Office fédéral de l'armement", "armasuisse",
    "Bundesamt für Rüstung",
    "Office fédéral de topographie", "swisstopo",
    "Bundesamt für Landestopografie",
    "Office fédéral du sport", "OFSPO",
    "Bundesamt für Sport", "BASPO",
    "Office fédéral de la cybersécurité", "OFCS",
    "Bundesamt für Cybersicherheit", "BACS",
    "Secrétariat d'État à la politique de sécurité", "SEPOS",
    "Staatssekretariat für Sicherheitspolitik",
    "Armée suisse", "Schweizer Armee",
    "Service de renseignement de la Confédération", "SRC",
    "Nachrichtendienst des Bundes", "NDB",
    "Office de l'auditeur en chef", "OAC",
    "Oberauditorat",
    # ── DFF / EFD ───────────────────────────────────────────────────────────
    "SG-DFF", "GS-EFD",
    "Secrétariat d'État aux questions financières internationales", "SFI",
    "Staatssekretariat für internationale Finanzfragen", "SIF",
    "Administration fédérale des finances", "AFF",
    "Eidgenössische Finanzverwaltung", "EFV",
    "Office fédéral du personnel", "OFPER",
    "Eidgenössisches Personalamt", "EPA",
    "Administration fédérale des contributions", "AFC",
    "Eidgenössische Steuerverwaltung", "ESTV",
    "Office fédéral de la douane et de la sécurité des frontières", "OFDF",
    "Bundesamt für Zoll und Grenzsicherheit", "BAZG",
    "Office fédéral de l'informatique et de la télécommunication", "OFIT",
    "Bundesamt für Informatik und Telekommunikation",
    "Office fédéral des constructions et de la logistique", "OFCL",
    "Bundesamt für Bauten und Logistik", "BBL",
    # ── DEFR / WBF ──────────────────────────────────────────────────────────
    "SG-DEFR", "GS-WBF",
    "Secrétariat d'État à l'économie", "SECO",
    "Staatssekretariat für Wirtschaft",
    "Secrétariat d'État à la formation, à la recherche et à l'innovation", "SEFRI",
    "Staatssekretariat für Bildung, Forschung und Innovation", "SBFI",
    "Office fédéral de l'agriculture", "OFAG",
    "Bundesamt für Landwirtschaft", "BLW",
    "Office fédéral pour l'approvisionnement économique du pays", "OFAE",
    "Bundesamt für wirtschaftliche Landesversorgung",
    "Office fédéral du logement", "OFL",
    "Bundesamt für Wohnungswesen", "BWO",
    "Office fédéral du service civil", "CIVI",
    "Bundesamt für Zivildienst", "ZIVI",
    # ── DETEC / UVEK ────────────────────────────────────────────────────────
    "SG-DETEC", "GS-UVEK",
    "Office fédéral des transports", "OFT",
    "Bundesamt für Verkehr", "BAV",
    "Office fédéral de l'aviation civile", "OFAC",
    "Bundesamt für Zivilluftfahrt", "BAZL",
    "Office fédéral de l'énergie", "OFEN",
    "Bundesamt für Energie", "BFE",
    "Office fédéral des routes", "OFROU",
    "Bundesamt für Strassen", "ASTRA",
    "Office fédéral de la communication", "OFCOM",
    "Bundesamt für Kommunikation", "BAKOM",
    "Office fédéral de l'environnement", "OFEV",
    "Bundesamt für Umwelt", "BAFU",
    "Office fédéral du développement territorial", "ARE",
    "Bundesamt für Raumentwicklung",
]


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
                {"AND": [{"OR": DE_TERMS}, {"OR": DE_LEVEL}]},
                {"AND": [{"OR": FR_TERMS}, {"OR": FR_LEVEL}]},
                {"OR": DEPARTMENTS},
                {"OR": ADMIN_UNITS},
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