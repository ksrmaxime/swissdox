# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect the rare articles that have absolutely no Swiss context mentioned.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given the title and lead of an article from a Swiss newspaper.

Context of the task:
This dataset has already been prefiltered from Swiss newspapers using Swiss-related and public-affairs keywords.
Most articles are expected to be related to Switzerland in some way.
Your job is ONLY to identify the rare false positives: articles that are entirely about a foreign country or foreign context and have no meaningful link to Switzerland.

Definition of Swiss-related:
An article IS Swiss-related if it has ANY meaningful connection to Switzerland, including but not limited to:
- Switzerland as a country (mention of Suisse, Schweiz, Svizzera, Svizra, Swiss, Helvetia, or any of its cantons or cities etc...)
- Swiss politics, administration, parliament, courts, parties, laws, or public institutions (mention of EDA, FDFA, Bundesrat, Bundesversammlung, Bundesgericht, Bundeskanzlei, or any Swiss political party etc...)
- Swiss actors, officials, companies, organizations, experts, or residents (mention of UBS, Nestlé, Novartis, Swiss banks, Swiss universities, or any Swiss person etc...)
- events, decisions, debates, or developments affecting Switzerland
- Swiss reactions to foreign events
- Swiss participation in international affairs (sent of swiss aid to a foreign country, swiss involvement in an international organization, swiss diplomacy, swiss trade relations, swiss tourism, swiss sports events etc...)
- comparisons with Switzerland
- consequences, implications, or relevance for Switzerland

Classification rule:
- Output "YES" only if the article is clearly and exclusively about a non-Swiss country or foreign context, with NO connection to Switzerland at all.
- Output "NO" if there is ANY Swiss connection, even indirect (like mention of a swiss administration), partial, comparative, diplomatic, economic, political, legal, or contextual.
- If unsure, output "NO".

Important clarifications:
- Mentioning a foreign country does NOT make the article non-Swiss.
- An article about international affairs is still Swiss-related if Switzerland is involved, mentioned, affected, compared, or implicated.
- An article about a foreign politician, foreign administration, or foreign bureaucracy is still "NO" if the title or lead gives any link to Switzerland.
- When there is a swiss administration mentioned in the title or lead, the article is Swiss-related, even if the rest of the article is about a foreign context.
- Use only the title and lead. Do not infer facts not present in them.

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