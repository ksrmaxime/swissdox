# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect the rare articles that are entirely unrelated to Switzerland.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given the title and lead of an article from a Swiss newspaper.

Context of the task:
This dataset has already been prefiltered from Swiss newspapers using Swiss-related and public-affairs keywords.
Most articles are expected to be related to Switzerland in some way.
Your job is ONLY to identify the rare cases that are entirely about a foreign context and have no link to Switzerland.

Classification rule:
- Output "YES" only when you are certain that the article doesn't have any implication, mention, effect, comparison, or link to Switzerland.
- Output "NO" if there is ANY link to Switzerland.
- If unsure, output "NO".

Important clarifications:
Mentioning a foreign country does NOT make the article non-Swiss. Event the smallest clue like, swiss currancy, companies, people, entity, city name etc... that would related to switzerland are enough to classify the article as "NO".
link to Switzerland is not limited to "talking about the country", it can be a comparison, an effect, an implication, a mention of swiss entities, people, companies, cities, currency, etc...

Return ONLY this strict JSON:
{{
  "justification": "<concise explanation based only on title and lead>",
  "non_swiss": "YES|NO"
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