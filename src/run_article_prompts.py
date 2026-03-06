# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "You must follow the requested output format exactly.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You will classify ONE article based only on its title and lead (not the full text).

Return ONLY this strict JSON (no extra keys, no markdown):
{{
  "justification": "",
  "non_swiss": "YES|NO",
}}

Rules:

A) non_swiss
- YES only if the article is clearly about a non-Swiss country/context (e.g., USA, France, EU institutions) AND has no Swiss anchor.
- NO otherwise.

B) justification
  - justification must be a concise explanation of the non_swiss classification, based only on the title and lead.

Title:
{title}

Lead:
{lead}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title")).strip()
    lead = "" if pd.isna(row.get("lead")) else str(row.get("lead")).strip()
    return USER_TEMPLATE.format(title=title, lead=lead)