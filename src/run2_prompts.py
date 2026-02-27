# run2_prompts.py
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
    "Task: assign EXACTLY ONE topic label to the given sentence.\n"
    "Return ONLY the topic string, with exact spelling, and nothing else.\n"
)

USER_TEMPLATE = """Choose EXACTLY ONE topic from the following list (spelling must match exactly):
- "Foreign Affairs"
- "Culture"
- "Health"
- "Social"
- "Justice"
- "Migration"
- "Defence"
- "Sport"
- "Finance"
- "Economy"
- "Education"
- "Research"
- "Environment"
- "Transports"
- "Energy"
- "Communication"
- "Other"

Topic guidance:
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
- Other

The topic should be the main topic of the sentence even if not related to public administration.
If unsure, choose "Other".

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)