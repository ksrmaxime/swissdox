# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "You must follow the requested output format exactly.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """ Here is an article title and lead from a swiss newspaper.
You job is to detect when the article talks about a non-swiss related country or context.
Classify the article as non_swiss YES or NO, and provide a justification for your classification.

YES only if the article talks only about a non-Swiss country/context without any connection to Switzerland.
NO otherwise.

IMPORTANT: If the article talks about a non-Swiss country/context but also has a connection to Switzerland, then the article is NOT non-Swiss and should be classified as NO.

Justification must be a concise explanation of the non_swiss classification, based only on the title and lead.


Return ONLY this strict JSON (no extra keys, no markdown):
{{
  "justification": "",
  "non_swiss": "YES|NO",
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