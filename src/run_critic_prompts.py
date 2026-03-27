# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

Your task is to determine whether the sentence conveys that someone (X) holds a public administration (Y) responsible for something negative or positive.

The key question is NOT "is this sentence written in a critical tone?" but rather:
"Does this sentence express — directly or through its factual framing — that someone (a journalist, a quoted person, a citizen, a speaker) is attributing blame, failure, or wrongdoing TO a public administration?"

CRITIC = The sentence conveys that someone (X) attributes blame, failure, or wrongdoing to a public administration (Y). X can be:
  • An explicitly named person or group expressing criticism
  • A quoted speaker demanding accountability or implying non-compliance
  • The author/journalist, whose choice of framing implies the administration is at fault (e.g. reporting that something is being concealed, that authorities failed at something, or that treatment is inconsistent or arbitrary)

PRAISE = The sentence conveys that someone (X) credits or praises a public administration (Y) for something it did or achieved.

NEUTRAL_STATEMENT = The sentence does not convey that any identifiable actor is holding the administration responsible. This includes:
  • The administration is a victim or target of external actors — no one is blaming the administration itself
  • A proposal, plan, or measure is described without the author or any quoted person passing judgement on it
  • The administration's actions or statements are reported neutrally, without framing that implies fault
  • A third party (not the administration) is the one being blamed or criticised

Reasoning approach:
1. Who, if anyone, is the critic or praiser (X)? Could be named, quoted, or implicit (the journalist's framing).
2. Who is the target administration (Y)?
3. Is X attributing something negative (→ CRITIC) or positive (→ PRAISE) to Y, or is no such attribution present (→ NEUTRAL_STATEMENT)?
4. If no clear attribution to an administration can be identified → NEUTRAL_STATEMENT.

Return ONLY this strict JSON:
{{
  "justification": "<one sentence identifying X, Y, and the nature of the attribution — or why none exists>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
