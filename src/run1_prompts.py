# run1_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: decide whether the sentence is a criticism, a congratulation or a neutral statement.\n"
    "Return EXACTLY ONE token and nothing else:\n"
    "CRITIC\n"
    "or\n"
    "CONGRATULATION\n"
    "or\n"
    "NEUTRAL\n"
)

USER_TEMPLATE = """Decide if the sentence is is a criticism, a congratulation or a neutral statement.

Return CRITIC if the sentence expresses criticism, complains, or is negative.
Return CONGRATULATION if the sentence expresses praise, appreciation, or positive sentiment.
Return NEUTRAL if the sentence is , factual and does not express criticism or praise.

Important tie-break:
- If uncertain, choose NEUTRAL.

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)
