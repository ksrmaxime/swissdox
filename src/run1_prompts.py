# run1_prompts.py
from __future__ import annotations

import pandas as pd

# Objectif: détecter si une phrase parle de la Suisse (au sens "contenu substantiel lié à la Suisse")
# Sortie attendue: UN SEUL TOKEN: YES ou NO

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: decide whether the given sentence is related to Swiss bureaucracy or foreign bureaucracy.\n"
    "Answer with EXACTLY ONE token and nothing else:\n"
    "YES\n"
    "or\n"
    "NO\n"
)

USER_TEMPLATE = """
The sentence is coming from a swiss newspapers, so it's often about Switzerland, answer = YES
But sometimes, the sentence talks about foreign countries or foreign bureaucracy, even if it comes from a swiss newspaper, answer = NO

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)
