# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

Classify whether the sentence contains a negative or positive EVALUATION — not merely a DESCRIPTION — of a public administration (a government, ministry, agency, court, municipality, etc.) attributed to an identifiable voice.

═══ THE CORE DISTINCTION ═══
A sentence can DESCRIBE an administration's action (neutral) or EVALUATE it (critic/praise).
Describing what officials did — even dramatic or consequential actions (arrests, sanctions, enforcement, rulings, denials) — is NOT a criticism unless the sentence also contains language that judges the action as wrong, unjust, excessive, incompetent, opaque, or arbitrary.

═══ CRITIC ═══
The sentence contains at least one evaluative signal — even in a subordinate clause, modifier, or embedded phrase — that attributes fault, failure, injustice, or inadequacy to a public administration (Y) through an identifiable voice (X).

Identifiable voices include:
  • A named or quoted person demanding accountability, expressing dissatisfaction, implying non-compliance, or recounting harmful treatment by officials in a first-person account
  • The journalist/author, when their own word choice carries evaluative meaning about the administration

Evaluative signals in the journalist's language (these are CRITIC even when phrased as facts):
  • Opacity / secrecy: kept secret, withheld, concealed, under wraps, not disclosed, "unter Verschluss", "sous le sceau du secret"
  • Stubbornness / poor judgment embedded in a clause: "had long clung to", "despite all warnings", "continued to insist despite"
  • Dysfunction / loss of confidence: "even insiders no longer believe", "key figures have left", "the project is in doubt"
  • Explicit failure / absurdity labels: "billion-franc flop", "absurd that", "regulatory standstill", "stumbling block"
  • Unnecessary or harmful bureaucracy: "unnecessary bureaucracy without real benefit", "bureaucratic obstacle", "regulatory paralysis"
  • Inconsistent or arbitrary treatment: "outcomes vary by official", "different experiences depending on location or civil servant"
  • Past inability of authorities to act: "what authorities had previously failed to achieve", "what had not worked before"
  • An expert or public figure calling for a change in institutional practice (implying current practice is inadequate)

═══ PRAISE ═══
The sentence contains a positive evaluation of an administration (Y) by an identifiable voice (X).

═══ NEUTRAL_STATEMENT ═══
The sentence describes facts, actions, decisions, or processes without any evaluative signal. This includes:
  • Officials performing authority actions (arrests, threats, enforcement, rulings, sanctions) described factually without the author judging them
  • The administration is the grammatical subject making a proposal, giving its own reasons, or explaining its position — no external voice evaluates this
  • A third party challenging or opposing the administration through a legal process (filing a complaint, lodging an appeal) — this is a procedural fact, not a criticism
  • A politician or candidate making a conditional promise about future action — this does not attribute blame to any current administration
  • The administration defending or explaining itself in response to external criticism (unless it admits fault)
  • A foreign or non-local administration's actions (outside the scope of Swiss public administration)
  • Evaluative language in the sentence targets a non-administration entity (a private actor, a foreign government, an individual politician in a personal capacity) — the administration is not the target

═══ REASONING ═══
Step 1 — Quick exclusions: Is this a legal procedure, a conditional promise, a foreign administration, or evaluative language targeting a non-administration entity? → NEUTRAL_STATEMENT
Step 2 — Scan every clause and modifier: is there any evaluative signal (see CRITIC list above) applied to a public administration anywhere in the sentence?
  • No evaluative signal anywhere → NEUTRAL_STATEMENT
Step 3 — Is the evaluative signal from the journalist's own word choice, or from an identifiable speaker?
  • If the only voice is the administration itself (explaining, proposing, defending) and no outside voice evaluates it → NEUTRAL_STATEMENT
Step 4 — Is the evaluation negative → CRITIC, or positive → PRAISE?
Step 5 — Still uncertain → NEUTRAL_STATEMENT

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: name the evaluative signal and its source (X), or explain why no evaluation is present>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
