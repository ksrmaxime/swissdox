# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect the rare articles that are entirely unrelated to Switzerland.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """Does this article title and lead contain any reference to Switzerland?

A Swiss reference includes: Switzerland itself, Swiss institutions (e.g. Bundesrat, EDA, SECO, SNB, Parlament, Kantone, Bundesgericht), Swiss officials, Swiss companies, Swiss laws, or any event/decision involving Switzerland — even if the main topic is foreign.

Output "NO" only if you find ZERO Swiss references. If unsure, output "NO".

Examples:
---
Title: "EDA hält Palästina-Infos zurück – «zum Schutz des Bundesrates»"
Lead: "Ein EDA-Gutachten zur Anerkennung Palästinas bleibt seit Juni unter Verschluss."
{{"non_swiss": "NO", "justification": "EDA and Bundesrat are Swiss federal institutions"}}
---
Title: "Trump signs sweeping immigration order"
Lead: "The US president signed an executive order restricting entry from several countries."
{{"non_swiss": "YES", "justification": "Purely about US politics, no Swiss reference"}}
---
Title: "Gaza: Hunderte Tote bei israelischen Angriffen"
Lead: "Bei israelischen Luftangriffen auf Gaza sind laut palästinensischen Behörden über 200 Menschen ums Leben gekommen."
{{"non_swiss": "YES", "justification": "Purely about the Gaza conflict, no Swiss reference"}}
---

Now classify:
Title: {title}
Lead: {lead}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title")).strip()
    lead = "" if pd.isna(row.get("lead")) else str(row.get("lead")).strip()
    return USER_TEMPLATE.format(title=title, lead=lead)