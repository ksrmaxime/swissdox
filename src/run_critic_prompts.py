# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence contains a critic or a praise towards a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

You have to answer the following question:
Does this sentence contain a clear and explicit critic or praise directed AT a public administration (e.g. a government, a ministry, a public agency, a municipality, a public institution, etc.)?

CRITIC            = the public administration is BLAMED or ACCUSED for something it did or failed to do — the fault or negative judgement is imputed TO the administration itself (e.g. "the ministry wasted funds", "the government failed to act", "the agency is incompetent").
PRAISE            = the public administration is explicitly credited or praised for something it did (e.g. "the government handled the crisis well").
NEUTRAL_STATEMENT = anything else: factual reporting, the administration is a victim or passive subject, a third party is criticised, a proposal is described without evaluative tone, or it is unclear who bears the criticism.

Key distinction — the administration must be the RESPONSIBLE PARTY being judged, not merely mentioned:
- "DDoS attacks targeted Swiss banks" → the administration is a VICTIM, not blamed → NEUTRAL_STATEMENT
- "The government proposes reducing travel privileges" → describes a proposal, no blame → NEUTRAL_STATEMENT
- "The government neglected cybersecurity, leaving banks vulnerable" → blame IS imputed to the administration → CRITIC

Tie-break: if uncertain or if the target is not a public administration, choose NEUTRAL_STATEMENT.

Return ONLY this strict JSON:
{{
  "justification": "<concise one-sentence explanation>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
