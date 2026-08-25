# stance_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at ONE SPECIFIC "
    "public-administration entity named by the user, not at any other entity that may also appear in the sentence.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence extracted from a text, and ONE specific entity mentioned in that sentence.

Target entity: {keyword}

Classify whether the sentence contains a JUDGMENT — not just a description — that {keyword} SPECIFICALLY acted wrongly, failed, was inadequate (CRITIC), or acted well (PRAISE). The sentence may also mention other administrations, entities, or actors: any evaluation of THOSE is irrelevant. Only a judgment about {keyword} itself counts.

━━━ THE CENTRAL TEST ━━━
Before classifying, ask: "Does this sentence make a CLAIM that {keyword} DID SOMETHING WRONG or FAILED?"
— If the answer is clearly YES → lean CRITIC
— If the sentence merely describes what {keyword} did/decided, or the evaluative language targets a different entity than {keyword} → lean NEUTRAL_STATEMENT
A word with negative connotations does NOT automatically mean CRITIC. The sentence must carry a claim of wrongdoing or failure directed AT {keyword} specifically.

━━━ CRITIC ━━━
The sentence contains a claim of wrongdoing, failure, or inadequacy directed at {keyword} (Y) from an identifiable voice (X):
  • A named or quoted person who: accuses {keyword}, demands it comply with rules (implying it doesn't), recounts harmful personal treatment by its officials, or calls for a change in its institutional practice (implying current practice is inadequate)
  • A direct negative label or adjective applied to {keyword} or its officials by an external voice (naive, incompetent, ineffective, unresponsive, etc.)
  • The journalist's own language — but ONLY when it explicitly labels {keyword}'s behavior as failure, dysfunction, unjust, opaque, or arbitrary. Specifically:
    - Secrecy / opacity: "kept secret", "withheld", "under wraps", "unter Verschluss gehalten", "nicht veröffentlicht"
    - Stubbornness embedded in a clause: "had long clung to X", "continued to insist despite all criticism"
    - Explicit failure or absurdity labels applied to {keyword}: "billion-franc flop", "absurd that X", "regulatory standstill"
    - Bureaucracy described as an obstacle or cause of harm: "stumbling block", "unnecessary bureaucracy without real benefit"
    - Inconsistency implying arbitrary treatment: citizens receive different treatment depending on which official handles their case or where they live
    - Past inability of {keyword} to fulfill its obligations: "what authorities had previously failed to achieve"
    - Internal dysfunction or loss of confidence within {keyword}: key members departing, or even insiders no longer believing in a project

━━━ PRAISE ━━━
The sentence contains a positive judgment of {keyword} (Y) by an identifiable voice (X) that is external to {keyword}.
  • This includes favorable comparisons to other administrations or countries ("better than in other countries")
  • NOTE: {keyword} announcing its own positive actions, or a spokesperson of {keyword} explaining a decision, is NOT praise — it is {keyword}'s own voice, which is NEUTRAL

━━━ NEUTRAL_STATEMENT ━━━
Use NEUTRAL_STATEMENT whenever the sentence does not clearly pass the central test. In particular:

DESCRIBING AN ACTION IS NOT EVALUATING IT:
  • {keyword} arresting, threatening, enforcing rules, or issuing sanctions — described factually = NEUTRAL
  • {keyword} issuing any ruling, including overturning a decision or ordering a review — this is normal judicial function, not self-criticism = NEUTRAL
  • A government shutdown or crisis involving {keyword} described using its standard terminology = NEUTRAL

{keyword} SPEAKING FOR ITSELF IS NOT CRITICISM OF IT:
  • {keyword} giving its own reasons, explaining a decision, or defending its position — even if those reasons concern a sensitive matter — its own voice is not an external evaluator = NEUTRAL
  • Reported speech attributed directly to {keyword} ("heisst es von X", "sagte X", "laut X") = still {keyword}'s own voice = NEUTRAL
  • An official of {keyword} making a normative statement about policy goals or requirements = NEUTRAL
  • {keyword} not yet having taken a position on something = absence of action, not a claim of failure = NEUTRAL

WRONG TARGET — evaluative language must target {keyword}'s OWN conduct, not any other entity in the sentence:
  • Evaluative language targeting a phone call, a meeting, a negotiation, or a diplomatic event = NEUTRAL
  • Evaluative language targeting a private individual, even if formerly associated with {keyword} = NEUTRAL
  • Evaluative language targeting a foreign government's internal actions (when {keyword} is not that government) = NEUTRAL
  • Evaluative language targeting a DIFFERENT administration or entity than {keyword}, even in the same sentence = NEUTRAL (for {keyword})
  • A legal expert describing institutional attributes (discretionary power, scope of competence) = NEUTRAL
  • An expert assessing the chances of a lawsuit = evaluates the lawsuit, not {keyword} = NEUTRAL

OTHER NEUTRAL PATTERNS:
  • A third party challenging {keyword} in a legal process (filing a complaint, lodging an appeal) = procedural fact = NEUTRAL
  • A politician making a conditional promise about future action = no blame attributed to {keyword}'s current conduct = NEUTRAL
  • A private party requesting {keyword} to do something, without implying it has failed = NEUTRAL

━━━ DECISION STEPS ━━━
1. Is {keyword} itself a foreign administration (not Swiss or local)? → NEUTRAL_STATEMENT
2. Is there evaluative language anywhere in the sentence (including subordinate clauses and modifiers) that targets {keyword} specifically — not some other entity mentioned in the sentence?
   • No → NEUTRAL_STATEMENT
3. Does that evaluative language target {keyword}'s own conduct — not a phone call, a ruling, a private actor, a foreign or other government?
   • No → NEUTRAL_STATEMENT
4. Is the evaluative voice external to {keyword} (not {keyword}'s own words, even in reported speech)?
   • No → NEUTRAL_STATEMENT
5. Is the evaluation of {keyword} negative → CRITIC, or positive → PRAISE?
6. Still uncertain → NEUTRAL_STATEMENT

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: name the evaluative signal about {keyword} and its source, or confirm no such signal targets {keyword}>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Target entity:
{keyword}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str, keyword_col: str = "matched_keywords") -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    keyword = "" if pd.isna(row.get(keyword_col)) else str(row.get(keyword_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence, keyword=keyword)
