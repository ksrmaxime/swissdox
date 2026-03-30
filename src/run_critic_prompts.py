# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

Your task is to determine whether the sentence contains a negative or positive evaluation of a public administration (e.g. a government, a ministry, a public agency, a court, a municipality, etc.), attributed to an identifiable voice.

The identifiable voice (X) can be:
  • A named person or group expressing a position
  • A quoted or paraphrased speaker demanding accountability, implying non-compliance, or recounting a negative personal experience with officials
  • The journalist or author, through their own word choice — including language of secrecy/opacity, dysfunction, inconsistency, obstruction, past failure, or internal disarray — applied to the administration anywhere in the sentence

CRITIC = The sentence contains a negative evaluation of an administration (Y) attributable to an identifiable voice (X). This includes:
  • Direct blame or accusation for something the administration did or failed to do
  • An expert, public figure, or citizen calling for change in institutional practice (implying current practice is inadequate)
  • A speaker implying authorities are not meeting their obligations or failed in the past
  • Personal testimony of intrusive, arbitrary, or harmful treatment by officials
  • Journalist word choice that implies opacity (withheld, concealed, kept secret, not disclosed), dysfunction (key people gone, even insiders doubt), obstruction, stubbornness, inconsistency in how people are treated, or internal loss of confidence in a project

PRAISE = The sentence contains a positive evaluation of an administration (Y) attributable to an identifiable voice (X).

NEUTRAL_STATEMENT = The sentence does not contain an evaluation attributable to any identifiable voice. This includes:
  • The administration is a victim or target of external actors — no one is blaming the administration itself
  • The administration is the active subject making a proposal, issuing a statement, or giving its own reasons — and no external voice is evaluating or challenging this
  • Neutral reporting of what the administration did or decided, using plain non-evaluative language
  • A third party (not the administration) is the one being evaluated
  • Legal or administrative proceedings described as procedural facts: a private party filing a complaint or appeal against an administrative decision, a court ordering an administration to review a case — these are institutional processes, not criticisms
  • A politician or candidate making a conditional promise about future action — this does not attribute blame to any current administration
  • The administration is defending or explaining itself in response to external criticism — unless it admits fault

Reasoning approach — work through these steps:
1. Is the sentence describing a legal/procedural fact (complaint filed, appeal lodged, court ordering review) or a conditional future promise? → lean NEUTRAL_STATEMENT
2. Scan the entire sentence, including all subordinate clauses and modifiers: is there any evaluative language (positive or negative) applied to the administration?
   - If no evaluative language anywhere → NEUTRAL_STATEMENT
3. Who is the voice (X) behind that evaluation — named, quoted, or implicit in the journalist's own word choice?
   - If the only voice is the administration itself (proposing, explaining, defending) and no external voice evaluates it → NEUTRAL_STATEMENT
4. Is the target of the evaluation (Y) a public administration?
   - If the target is a private entity or a third party → NEUTRAL_STATEMENT
5. Is the evaluation negative → CRITIC, or positive → PRAISE?
6. If still uncertain → NEUTRAL_STATEMENT

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: identify X, Y, the evaluative content, and where in the sentence it appears — or explain why no evaluation is present>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
