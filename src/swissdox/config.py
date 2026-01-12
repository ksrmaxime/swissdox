# src/swissdox/config.py
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any


DEFAULT_API_BASE_URL = "https://swissdox.linguistik.uzh.ch/api"
DEFAULT_COLUMNS = [
    "id","pubtime","medium_code","medium_name","rubric","regional",
    "doctype","doctype_description","language","char_count","dateline",
    "head","subhead","content_id","content",
]


# ---- Terms (from your block) ----
DE_TERMS = [
    "Bürokratie","Berner Verwaltung","Papierkrieg","Verwaltung","Bundesverwaltung",
    "Beamtenapparat","Amtsschimmel","Regulierungsdichte","Behörden","Bürokraten",
    "Beamte","Staatsangestellte",
]
DE_LEVEL = ["Bund", "Bundes", "Kanton", "kantonal", "Schweiz"]

FR_TERMS = [
    "Bureaucratie","Administration publique","Administration fédérale","Appareil administratif",
    "Appareil étatique","Appareil de l’État","Autorités administratives","Services de l’État",
    "Services publics","Fonction publique","Pouvoir administratif","Autorités cantonales",
    "Administration centrale","Départements fédéraux","Offices fédéraux","Organes de l’État",
    "Technocratie","Bureaucrates","Fonctionnaires","Employés de l'État",
]
FR_LEVEL = ["fédéral", "federal", "federale", "cantonal", "cantonale", "Suisse"]

DEPARTMENTS = [
    "VBS","DDPS","Eidgenössische Departement für Verteidigung, Bevölkerungsschutz und Sport",
    "Département fédéral de la défense, de la protection de la population et des sports",
    "EDA","DFAE","Eidgenössische Departement für auswärtige Angelegenheiten",
    "Département fédéral des affaires étrangères",
    "UVEK","DETEC","Eidgenössische Departement für Umwelt, Verkehr, Energie und Kommunikation",
    "Département fédéral de l'environnement, des transports, de l'énergie et de la communication",
    "EJPD","DFJP","Eidgenössische Justiz- und Polizeidepartement","Département fédéral de justice et police",
    "EDI","DFI","Eidgenössische Departement des Innern","Département fédéral de l'intérieur",
    "EFD","DFF","Eidgenössische Finanzdepartement","Département fédéral des finances",
    "WBF","DEFR","Eidgenössische Departement für Wirtschaft, Bildung und Forschung",
    "Département fédéral de l'économie, de la formation et de la recherche",
]


@dataclass(frozen=True)
class SwissdoxQueryParams:
    start_date: str
    end_date: str
    languages: List[str]
    sources: List[str]
    max_results: int = 20000
    columns: List[str] = None  # filled in __post_init__ like behavior avoided


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
        "meta": {
            "name": query_name,
            "comment": comment,
            "expirationDate": expiration_date,
        },
    }


def default_query_name(base: str = "BuerokratieVerwaltung_2025") -> str:
    return f"{base}_{datetime.now():%Y%m%d_%H%M%S}"