# run1_prompts.py
from __future__ import annotations

import pandas as pd

# Objectif: détecter si une phrase parle de la Suisse (au sens "contenu substantiel lié à la Suisse")
# Sortie attendue: UN SEUL TOKEN: YES ou NO

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: decide whether the given sentence is related to Switzerland.\n"
    "Answer with EXACTLY ONE token and nothing else:\n"
    "YES\n"
    "or\n"
    "NO\n"
)

USER_TEMPLATE = """
The sentence is coming from a swiss newspapers, but sometime the media article is talking about what is happening with other countries bureaucracy.

Decision rules :
The sentence should be considered as talking about Switzerland when the topic is an internal swiss related topic or when it's a foreign affaires directly impacting switzerland.
The only case where it would not be considered swiss related, is when it's an internal bureaucracy topic from another country like for exemple shutdown of the US government, strike in the french administration or Brussels internal discussion etc...

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)
