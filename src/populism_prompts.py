# populism_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text analysis system.\n"
    "Your task is to identify whether the criticism uses populist rhetoric.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence that has already been classified as a CRITICISM directed at a public administration.

Perform the following task:

━━━ POPULISM CLASSIFICATION ━━━
Assess whether this criticism uses populist rhetoric based on the following definition:

  POPULIST RHETORIC — "The People" vs. "The Elite":
  Populism claims that the "true" will of the people is being undermined by a corrupt,
  out-of-touch elite, which can include politicians, media, or corporations.
  Key markers:
    • Explicit or implicit invocation of "the people", "ordinary citizens", "the public", or a
      synonym as a virtuous, homogeneous group
    • The targeted entity is framed as part of a self-serving, corrupt, or detached elite
    • A moral opposition is constructed between the people (pure/legitimate) and the elite (corrupt/illegitimate)
    • The criticism implies that the elite is acting against the interests or will of the people

  Choose ONE of the three labels:
    "Not Populist"      — No populist rhetoric. The criticism targets specific conduct, failures,
                          or decisions without invoking a people-vs-elite frame.
    "Somehow Populist"  — Some populist elements are present (e.g., the elite framing is implied
                          or secondary), but the sentence does not fully commit to a populist narrative.
    "Clearly Populist"  — Strong populist rhetoric: the people-vs-elite opposition is explicit and
                          central to the criticism.

━━━ DECISION STEPS ━━━
1. Identify what specific entity (person, institution, group) is being criticised.
2. Does the sentence invoke "the people" or their equivalent as a moral reference point?
   • No → "Not Populist"
3. Is the targeted entity framed as an elite acting against the people's interests?
   • No → "Not Populist" or "Somehow Populist" if weakly implied
4. Is the people-vs-elite opposition explicit and central?
   • Yes → "Clearly Populist"
   • Partially → "Somehow Populist"
   • No → "Not Populist"

Return ONLY this strict JSON:
{{
  "target_entity": "<entity exactly as written in the text, or null>",
  "populism": "Not Populist|Somehow Populist|Clearly Populist",
  "justification": "<one sentence: name the targeted entity and the key populist signal, or confirm its absence>"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
