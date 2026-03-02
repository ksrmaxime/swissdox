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

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "You must follow the requested output format exactly.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You will classify ONE sentence on FOUR dimensions.

Return ONLY this strict JSON (no extra keys, no explanations, no markdown):
{{
  "swiss": "YES|NO",
  "topic": "<one of the allowed topic strings>",
  "sp": 1|0|-1,
  "p": 1|0|-1
}}

A) swiss (related to Switzerland: YES/NO)
- YES if Switzerland is explicitly mentioned OR clearly implied by Swiss-specific context.
- NO only if clearly about other countries/contexts with no Swiss anchor.
Strong Swiss cues (=> YES):
- words: Schweiz, schweizerisch, Schweizer, Eidgenossenschaft, Bund, Bundesrat, Parlament, Nationalrat, Ständerat
- places: Kanton, Gemeinden, Zürich, Bern, Genf, Basel, Lausanne, Tessin, Wallis, etc.
- institutions/acronyms: SBB/CFF/FFS, SRF/RTS/SSR, SNB, BAG/OFSP, SECO, SEM, Fedpol, ASTRA/OFROU, BAZL, UVEK/DETEC
- Swiss politics: SVP/UDC, SP/PS, FDP/PLR, Die Mitte/Centre, Grüne/Verts, Initiative, Referendum, Abstimmung/Votation
- currency: CHF
Tie-break: if uncertain, swiss="YES" (prefer recall).

B) topic (single label; spelling must match exactly)
Choose EXACTLY ONE topic from:
- "Foreign Affairs" / "Culture" / "Health" / "Social" / "Justice" / "Migration" / "Defence" / "Sport"
- "Finance" / "Economy" / "Education" / "Research" / "Environment" / "Transports" / "Energy" / "Communication" / "Other"
Guidance:
- Foreign Affairs: international trade, cooperation, representation, diplomacy.
- Culture: culture, art, music, museums.
- Health: health, doctors, hospitals.
- Social: social affairs, elderly people, pensions, aid.
- Justice: law, courts, police, justice system.
- Migration: asylum, migration, immigration, foreigners, work permits.
- Defence: defence, army, civil protection.
- Sport: sport, clubs, promotion of sport.
- Finance: budget, taxation.
- Economy: economy, business, money, banks, trade, markets.
- Education: schools, university, high school, education, teachers.
- Research: research, innovation, science, labs, technology.
- Environment: ecology, climate change, nature protection.
- Transports: roads, trains, cars, planes, mobility.
- Energy: energy consumption, nuclear energy, electricity, oil, coal.
- Communication: TV, radio, internet networks, antennas.
If unsure, choose "Other".

C) sp (sentiment toward public administration: 1 / 0 / -1)
Assess the sentiment toward PUBLIC ADMINISTRATION or its actions.
Public administration = administrative bodies/agencies/offices, civil servants, bureaucracy, administrative procedures.
NOT public administration = parties/elected politicians in general, unless the sentence explicitly evaluates administrative functioning.
-  1 = positive (supportive/appreciative of public administration or its actions)
- -1 = negative (criticism/blame/dissatisfaction toward public administration or its actions)
-  0 = neutral (descriptive, factual, vague, or no clear evaluation of public administration)
If uncertain, assign 0.

D) p (populism: 1 / 0 / -1)
-  1 = Populist: clear populist rhetoric such as:
  - explicit "the people" vs "the elite/establishment" framing, OR
  - anti-institution legitimacy attacks (courts, administration, "the system") portrayed as corrupt/illegitimate, OR
  - scapegoating an out-group framed as an enemy of "the people".
- -1 = Not Populist: no populist rhetoric.
-  0 = Somehow Populist: could be populist but not clearly people-vs-elite.
If uncertain between 0 and -1, assign -1.

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)