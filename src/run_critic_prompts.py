# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text.

Classify whether the sentence contains a JUDGMENT — not just a description — that a public administration (a government, ministry, agency, court, municipality, etc.) acted wrongly, failed, was inadequate, or (for PRAISE) acted well.

━━━ THE CENTRAL TEST ━━━
Before classifying, ask: "Does this sentence make a CLAIM that the administration DID SOMETHING WRONG or FAILED?"
— If the answer is clearly YES → lean CRITIC
— If the sentence merely describes what happened, what was decided, or how a situation unfolded → lean NEUTRAL_STATEMENT
A word with negative connotations does NOT automatically mean CRITIC. The sentence must carry a claim of wrongdoing or failure directed AT the administration.

━━━ CRITIC ━━━
The sentence contains a claim of wrongdoing, failure, or inadequacy directed at the administration (Y) from an identifiable voice (X):
  • A named or quoted person who: accuses the administration, demands it comply with rules (implying it doesn't), recounts harmful personal treatment by officials, or calls for a change in institutional practice (implying current practice is inadequate)
  • The journalist's own language — but ONLY when it explicitly labels the administration's behavior as failure, dysfunction, unjust, opaque, or arbitrary. Specifically:
    - Secrecy / opacity: "kept secret", "withheld", "under wraps", "nicht veröffentlicht / unter Verschluss gehalten"
    - Stubbornness embedded as a clause modifier: "had long clung to X despite warnings", "continued to insist despite all criticism"
    - Explicit failure or absurdity labels: "billion-franc flop", "absurd that X", "regulatory standstill that contradicts Y"
    - Describing bureaucracy as an obstacle or cause of harm: "stumbling block", "unnecessary bureaucracy without real benefit"
    - Noting inconsistency implying arbitrary treatment: "outcomes differ depending on the official or location"
    - Reference to the administration's past inability to fulfill obligations: "what the authorities had previously failed to achieve"

━━━ PRAISE ━━━
The sentence contains a positive judgment of the administration (Y) by an identifiable voice (X).

━━━ NEUTRAL_STATEMENT ━━━
Use NEUTRAL_STATEMENT whenever the sentence does not clearly pass the central test. In particular:

DESCRIBING AN ACTION IS NOT EVALUATING IT. Even if the action is dramatic or has negative consequences, reporting it factually is NEUTRAL:
  • Officials arresting, threatening, enforcing rules, or issuing rulings → described factually = NEUTRAL
  • A government shutdown or crisis described using its standard terminology → NEUTRAL

THE ADMINISTRATION SPEAKING FOR ITSELF IS NOT CRITICISM OF IT:
  • The administration giving its own reasons, explaining a decision, or stating a policy position → its own voice is not an external evaluator → NEUTRAL
  • An official making a normative statement about what policy should be ("must happen") → NEUTRAL

EXPERT AND LEGAL OBSERVATIONS ARE NOT EVALUATIONS OF THE ADMINISTRATION:
  • A legal scholar describing institutional attributes (discretionary power, procedural rights, scope of competence) → NEUTRAL
  • An expert assessing the chances of a lawsuit → evaluates the lawsuit, not the administration → NEUTRAL

OTHER NEUTRAL PATTERNS:
  • A third party challenging the administration in a legal process (filing a complaint, lodging an appeal) → procedural fact → NEUTRAL
  • A politician making a conditional promise about future action → no blame attributed to current administration → NEUTRAL
  • Evaluative language that targets something OTHER than the administration: a private actor, a foreign government, a phone call, a lawsuit → NEUTRAL
  • Actions of a foreign or non-local administration → NEUTRAL

━━━ DECISION STEPS ━━━
1. Is the sentence about a Swiss or local public administration? If it concerns a foreign government's internal situation → NEUTRAL_STATEMENT
2. Is there evaluative language (a judgment of wrong/failure/inadequacy) — including in subordinate clauses or modifiers?
   • No evaluative language → NEUTRAL_STATEMENT
3. Does the evaluative language target the administration's behavior or decisions (not a lawsuit, a phone call, a private actor)?
   • Target is not the administration → NEUTRAL_STATEMENT
4. Is the evaluative voice external to the administration (not the administration explaining itself)?
   • Only voice is the administration's own → NEUTRAL_STATEMENT
5. Is the judgment negative → CRITIC, or positive → PRAISE?
6. Still uncertain → NEUTRAL_STATEMENT

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: state the claim of wrongdoing/failure and its source — or confirm no such claim is present>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
