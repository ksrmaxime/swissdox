# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether an article mentions a Swiss context.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given the title and lead of an article from a Swiss newspaper.

You have to answer the following question:
Does the title or the lead mention at least one Swiss-related context (Switzerland as a country, places, institutions, persons, events, etc.) ?
One objective mention of a relevant keyword in the title or the lead is sufficient, no need to find elements in both or to find multiple elements.
Important, the keyword should be concretely related not hypothetical.

YES = Swiss context mentioned
NO  = no Swiss context mentioned

Return ONLY this strict JSON:
{{
  "justification": "<concise explanation based only on title or lead>",
  "swiss": "YES|NO"
}}

Title:
{title}

Lead:
{lead}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    title = "" if pd.isna(row.get("title")) else str(row.get("title")).strip()
    lead = "" if pd.isna(row.get("lead")) else str(row.get("lead")).strip()
    return USER_TEMPLATE.format(title=title, lead=lead)