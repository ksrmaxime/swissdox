# stance_prompts.py
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
  • A direct negative label or adjective applied to authorities or their officials by an external voice (naive, incompetent, ineffective, unresponsive, etc.)
  • The journalist's own language — but ONLY when it explicitly labels the administration's behavior as failure, dysfunction, unjust, opaque, or arbitrary. Specifically:
    - Secrecy / opacity: "kept secret", "withheld", "under wraps", "unter Verschluss gehalten", "nicht veröffentlicht"
    - Stubbornness embedded in a clause: "had long clung to X", "continued to insist despite all criticism"
    - Explicit failure or absurdity labels applied to the administration: "billion-franc flop", "absurd that X", "regulatory standstill"
    - Bureaucracy described as an obstacle or cause of harm: "stumbling block", "unnecessary bureaucracy without real benefit"
    - Inconsistency implying arbitrary treatment: citizens receive different treatment depending on which official handles their case or where they live
    - Past inability of authorities to fulfill obligations: "what authorities had previously failed to achieve"
    - Internal dysfunction or loss of confidence: key members departing, or even insiders no longer believing in a project

━━━ PRAISE ━━━
The sentence contains a positive judgment of the administration (Y) by an identifiable voice (X) that is external to the administration.
  • This includes favorable comparisons to other administrations or countries ("better than in other countries")
  • NOTE: An administration announcing its own positive actions or a spokesperson explaining a decision is NOT praise — it is the administration's own voice, which is NEUTRAL

━━━ NEUTRAL_STATEMENT ━━━
Use NEUTRAL_STATEMENT whenever the sentence does not clearly pass the central test. In particular:

DESCRIBING AN ACTION IS NOT EVALUATING IT:
  • Officials arresting, threatening, enforcing rules, or issuing sanctions — described factually = NEUTRAL
  • A court issuing any ruling, including overturning a decision or ordering a review — this is normal judicial function, not self-criticism = NEUTRAL
  • A government shutdown or crisis described using its standard terminology = NEUTRAL

THE ADMINISTRATION SPEAKING FOR ITSELF IS NOT CRITICISM OF IT:
  • The administration giving its own reasons, explaining a decision, or defending its position — even if those reasons concern a sensitive matter — its own voice is not an external evaluator = NEUTRAL
  • Reported speech attributed directly to the administration ("heisst es von X", "sagte X", "laut X") = still the administration's own voice = NEUTRAL
  • An official making a normative statement about policy goals or requirements = NEUTRAL
  • The administration not yet having taken a position on something = absence of action, not a claim of failure = NEUTRAL

WRONG TARGET — evaluative language must target the administration's OWN conduct:
  • Evaluative language targeting a phone call, a meeting, a negotiation, or a diplomatic event = NEUTRAL
  • Evaluative language targeting a private individual, even if formerly associated with a public institution = NEUTRAL
  • Evaluative language targeting a foreign government's internal actions = NEUTRAL
  • A legal expert describing institutional attributes (discretionary power, scope of competence) = NEUTRAL
  • An expert assessing the chances of a lawsuit = evaluates the lawsuit, not the administration = NEUTRAL

OTHER NEUTRAL PATTERNS:
  • A third party challenging the administration in a legal process (filing a complaint, lodging an appeal) = procedural fact = NEUTRAL
  • A politician making a conditional promise about future action = no blame attributed to current administration = NEUTRAL
  • A private party requesting the administration to do something, without implying it has failed = NEUTRAL

━━━ DECISION STEPS ━━━
1. Is the only administration mentioned foreign (not Swiss or local)? → NEUTRAL_STATEMENT
2. Is there evaluative language anywhere in the sentence (including subordinate clauses and modifiers)?
   • No → NEUTRAL_STATEMENT
3. Does the evaluative language target the administration's own conduct — not a phone call, a ruling, a private actor, a foreign government?
   • No → NEUTRAL_STATEMENT
4. Is the evaluative voice external to the administration (not the administration's own words, even in reported speech)?
   • No → NEUTRAL_STATEMENT
5. Is the evaluation negative → CRITIC, or positive → PRAISE?
6. Still uncertain → NEUTRAL_STATEMENT

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: name the evaluative signal and its source, or confirm no such signal is present>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
