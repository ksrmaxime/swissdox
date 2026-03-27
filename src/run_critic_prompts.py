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
Does this sentence contain a critic or praise towards a public administration (e.g. a government, a ministry, a public agency, a municipality, a public institution, etc.)?

CRITIC            = the public administration bears responsibility for a negative outcome, failure, or wrongdoing — either explicitly stated OR logically implied by the factual framing. This includes:
  • Direct blame or accusation for something the administration did or failed to do
  • Factual framing that implies wrongdoing without stating it outright — e.g. that information is being withheld or concealed, that authorities are not following their own rules or the law, that past efforts by authorities have failed, or that official treatment of people is inconsistent or arbitrary
  • Statements from third parties demanding compliance or accountability from the administration (implying current non-compliance or failure)

PRAISE            = the public administration is explicitly credited or praised for something it did or achieved.

NEUTRAL_STATEMENT = anything else, including:
  • The administration is a victim or target of external actors (attacks, crimes, external pressure)
  • Factual description of a proposal, plan, or measure without evaluative tone
  • Neutral reporting of what the administration said, decided, or published
  • Description of a situation or event where no blame or implied wrongdoing is attributable to the administration
  • A third party (not the administration) is the one being criticised or blamed

Key diagnostic questions — ask them in order:
1. Is the administration the TARGET of an external action (attack, accusation by others, external event)? → lean NEUTRAL_STATEMENT
2. Is a proposal or decision described without the author expressing any judgement? → lean NEUTRAL_STATEMENT
3. Is blame or wrongdoing attributed TO the administration, whether directly or by logical implication of the facts presented? → lean CRITIC
4. Still uncertain? → choose NEUTRAL_STATEMENT

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
