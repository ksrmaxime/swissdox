# run3_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: assess the sentiment toward PUBLIC ADMINISTRATION in the given sentence.\n"
    "Return EXACTLY ONE token and nothing else:\n"
    "1\n"
    "0\n"
    "-1\n"
)

USER_TEMPLATE = """B. sp (sentiment toward public administration: 1 / 0 / -1)

Assess the sentiment toward PUBLIC ADMINISTRATION or its actions.

Definition:
Public administration = administrative bodies/agencies/offices, civil servants, bureaucracy, administrative procedures.
NOT public administration = political parties or elected politicians in general, unless the sentence explicitly evaluates administrative functioning.

Labels:
-  1 = positive: Supportive/appreciative of public administration or its actions.
- -1 = negative: Criticism/blame/dissatisfaction toward public administration or its actions.
-  0 = neutral: descriptive, factual, vague, or no clear evaluation of public administration.

Important:
- Focus ONLY on the evaluation of public administration.
- Ignore general political conflict unless administrative functioning is explicitly targeted.
- If the sentence does not evaluate public administration, assign 0.
- If uncertain, assign 0.

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)