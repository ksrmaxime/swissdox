# run1_prompts.py
from __future__ import annotations
import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Task: decide whether the sentence is related to Switzerland.\n"
    "Return EXACTLY ONE token and nothing else:\n"
    "YES\n"
    "or\n"
    "NO\n"
)

USER_TEMPLATE = """Decide if the sentence is related to Switzerland.

Return YES if Switzerland is explicitly mentioned OR clearly implied by Swiss-specific context.
Return NO only if the sentence is clearly about other countries/contexts with no Swiss anchor, or fully generic with no identifiable Swiss link.

Strong Swiss cues (=> YES):
- words: Schweiz, schweizerisch, Schweizer, Eidgenossenschaft, Bund, Bundesrat, Parlament, Nationalrat, Ständerat
- places: Kanton, Gemeinden, Zürich, Bern, Genf, Basel, Lausanne, Tessin, Wallis, etc.
- institutions/acronyms: SBB/CFF/FFS, SRF/RTS/SSR, SNB, BAG/OFSP, SECO, SEM, Fedpol, ASTRA/OFROU, BAZL, UVEK/DETEC
- Swiss politics: SVP/UDC, SP/PS, FDP/PLR, Die Mitte/Centre, Grüne/Verts, Initiative, Referendum, Abstimmung/Votation
- currency: CHF

Important tie-break:
- If uncertain, choose YES (we prefer recall).

Sentence:
{sentence}
"""

def build_user_prompt(row: pd.Series, text_col: str) -> str:
    txt = "" if pd.isna(row[text_col]) else str(row[text_col]).strip()
    return USER_TEMPLATE.format(sentence=txt)
