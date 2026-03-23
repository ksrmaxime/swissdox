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
Does this sentence contain a critic or a praise towards a public administration (e.g. a government, a ministry, a public agency, a municipality, a public institution, etc.)?

CRITIC           = the sentence expresses a negative judgment, blame, disapproval, or criticism directed at a public administration
PRAISE           = the sentence expresses a positive judgment, approval, commendation, or praise directed at a public administration
NEUTRAL_STATEMENT = the sentence does not express a clear positive or negative stance towards a public administration (factual statement, no public administration involved, ambiguous, etc.)

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
