# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

Your task is to determine whether the sentence contains a negative or positive evaluation of a public administration (e.g. a government, a ministry, a public agency, a municipality, a public institution, etc.), attributed to an identifiable voice.

The identifiable voice (X) criticising or praising the administration (Y) can be:
  • A named person or group expressing a position
  • A quoted or paraphrased speaker demanding accountability, expressing dissatisfaction, or implying the administration is failing
  • The journalist or author, when their own word choice — including in embedded clauses, modifiers, or appositional phrases — frames the administration as responsible for a failure, obstacle, or wrongdoing
  • Anyone recounting a personal negative or positive experience with officials or bureaucracy

CRITIC = The sentence contains a negative evaluation of an administration (Y) attributable to an identifiable voice (X). The critical content can appear anywhere in the sentence — in the main clause, a subordinate clause, a modifier, or an embedded phrase. Signals include: attributing failure, incompetence, obstruction, dishonesty, arbitrariness, stubbornness, or non-compliance to the administration; characterising it as an obstacle or cause of problems; implying it is acting against the public interest or against rules.

PRAISE = The sentence contains a positive evaluation of an administration (Y) attributable to an identifiable voice (X).

NEUTRAL_STATEMENT = The sentence does not contain an evaluation attributable to any identifiable voice. This includes:
  • The administration is a victim of external actors — no one is blaming the administration itself
  • The administration is the active subject making a proposal, issuing a statement, or giving its own reasons — and no external voice is evaluating or challenging this
  • Neutral reporting of what an administration did or decided, using plain non-evaluative language
  • A third party (not the administration) is the one being evaluated
  • The administration is defending or explaining itself in response to external criticism — unless it admits fault
  • A factual situation is described where the administration is incidentally mentioned without being held responsible

Reasoning approach — work through these steps:
1. Scan the entire sentence, including all subordinate clauses and modifiers: is there any evaluative language (positive or negative) applied to the administration anywhere in the sentence?
   - If no evaluative language anywhere → NEUTRAL_STATEMENT
2. Who is the voice (X) behind that evaluation — named, quoted, or implicit in the journalist's own word choice?
   - If the only voice is the administration itself (proposing, explaining, defending) and no external voice evaluates it → NEUTRAL_STATEMENT
3. Is the target of the evaluation (Y) a public administration?
   - If the target is a private entity, a foreign government unrelated to Swiss/local administration, or a third party → NEUTRAL_STATEMENT
4. Is the evaluation negative → CRITIC, or positive → PRAISE?
5. If still uncertain → NEUTRAL_STATEMENT

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
