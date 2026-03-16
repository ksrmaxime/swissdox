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
Your job is ONLY to identify the rare false positives: articles that are entirely about a foreign country or foreign context and have no meaningful link to Switzerland.

Definition of Swiss-related:
An article IS Swiss-related if it has ANY meaningful connection to Switzerland, including but not limited to:
- Switzerland as a country
- Swiss politics, administration, parliament, courts, parties, laws, or public institutions
- Swiss actors, officials, companies, organizations, experts, or residents
- events, decisions, debates, or developments affecting Switzerland
- Swiss reactions to foreign events
- Swiss participation in international affairs
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

Step-by-step reasoning (follow this order strictly):
1. List any Swiss connections found in the title and lead (Swiss institutions, officials, companies, political bodies, laws, events, etc.).
2. If you found at least one Swiss connection in step 1, the answer MUST be "NO".
3. Only if you found zero Swiss connections in step 1, the answer MAY be "YES".
4. Your "non_swiss" field MUST match the conclusion from steps 1-3. Never output a "non_swiss" value that contradicts your justification.

Return ONLY this strict JSON:
{{
  "justification": "<concise explanation: list Swiss connections found, or state none found, then state your conclusion>",
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