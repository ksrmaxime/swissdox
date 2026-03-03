# src/run_all_prompts.py
from __future__ import annotations

import pandas as pd

TOPICS = [
    "Foreign Affairs",
    "Culture",
    "Health",
    "Social",
    "Justice",
    "Migration",
    "Defence",
    "Sport",
    "Finance",
    "Economy",
    "Education",
    "Research",
    "Environment",
    "Transports",
    "Energy",
    "Communication",
    "Other",
]

STANCES = ["CRITICISM", "PRAISE", "NEUTRAL"]

DEPARTMENTS = ["DFAE/EDA", "DFI/EDI", "DFJP/EJPD", "DDPS/VBS", "DFF/EFD", "DEFR/WBF", "DETEC/UVEK"]

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "You must follow the requested output format exactly.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You will classify ONE sentence with this schema.

Return ONLY this strict JSON (no extra keys, no explanations, no markdown):
{{
  "non_swiss": "YES|NO",
  "stance": "CRITICISM|PRAISE|NEUTRAL",
  "dept": "DFAE/EDA|DFI/EDI|DFJP/EJPD|DDPS/VBS|DFF/EFD|DEFR/WBF|DETEC/UVEK|null",
  "topic": "<one of the allowed topic strings>|null",
  "populism": 1|0|-1
}}

Rules:

A) non_swiss (does the sentence mentions another country/context than Switzerland ? YES/NO)
- YES only if the sentence is clearly about a non-Swiss country/context (e.g., USA, France, EU institutions) AND has no Swiss anchor.
- NO otherwise.
Tie-break: if uncertain, return non_swiss="NO" (prefer recall).

B) stance (about public administration/government action in tone)
- CRITICISM: complaint, blame, demand, accusation, negative judgement (explicit or clearly implied).
- PRAISE: compliment, approval, success framing, gratitude, positive judgement.
- NEUTRAL: descriptive/factual, no clear evaluative tone, or unclear.
Tie-break: if uncertain, choose NEUTRAL.

C) dept vs topic (conditional)
- If stance is CRITICISM or PRAISE:
  - dept MUST be exactly one of: DFAE/EDA, DFI/EDI, DFJP/EJPD, DDPS/VBS, DFF/EFD, DEFR/WBF, DETEC/UVEK.
  - topic MUST be null.
- If stance is NEUTRAL:
  - topic MUST be exactly one of the allowed topics (spelling must match).
  - dept MUST be null.

D) populism
Only if stance="CRITICISM":
-  1 = Clearly populist: explicit people vs elite framing; delegitimizing institutions as corrupt/illegitimate; scapegoating an out-group as enemy of "the people".
-  0 = Somewhat populist: hints of anti-elite / anti-system rhetoric but not fully explicit.
- -1 = Not populist.
If stance is not CRITICISM: populism MUST be null.
Tie-break: if uncertain between 0 and -1, choose -1.

Allowed topics (topic):
- "Foreign Affairs" / "Culture" / "Health" / "Social" / "Justice" / "Migration" / "Defence" / "Sport"
- "Finance" / "Economy" / "Education" / "Research" / "Environment" / "Transports" / "Energy" / "Communication" / "Other"
If unsure, choose "Other".

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)