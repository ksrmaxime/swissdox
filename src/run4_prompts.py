# run4_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: detect populist rhetoric in the given sentence.\n"
    "Return EXACTLY ONE token and nothing else:\n"
    "1\n"
    "0\n"
    "-1\n"
)

USER_TEMPLATE = """C. p (populism: 1 / 0 / -1)

Decide whether the sentence contains populist rhetoric.

Labels:
-  1 = Populist: the sentence contains clear populist rhetoric such as:
  - explicit "the people" vs "the elite/establishment" framing, OR
  - anti-institution legitimacy attacks (courts, administration, "the system") portrayed as corrupt/illegitimate, OR
  - scapegoating an out-group framed as an enemy of "the people".
- -1 = Not Populist: no populist rhetoric.
-  0 = Somehow Populist: the sentence could be populist but doesn't clearly fall into people vs elite framing.

Important:
- Focus on rhetoric/framing, not just criticism.
- Strong populism usually includes "people vs elite" or delegitimization of institutions as corrupt/illegitimate.
- If the sentence is only ordinary policy disagreement, assign -1.
- If uncertain between 0 and -1, assign 0.

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)