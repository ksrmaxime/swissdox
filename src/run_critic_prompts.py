# run_critic_prompts.py
from __future__ import annotations

import pandas as pd

SYSTEM_PROMPT = (
    "You are a STRICT text classification system.\n"
    "Your task is to detect whether a sentence conveys a criticism or a praise directed at a public administration.\n"
    "Return ONLY valid JSON and nothing else.\n"
)

USER_TEMPLATE = """You are given a sentence from a news article. Classify it using the decision tree below.

══════════════════════════════════════════
STEP 1 — SCOPE CHECK
══════════════════════════════════════════
Is there a Swiss or local public administration involved (government, ministry, agency, court, municipality, cantonal authority, etc.)?
If the only administrations mentioned are foreign (Turkish, Israeli, American, etc.) → NEUTRAL_STATEMENT.

══════════════════════════════════════════
STEP 2 — WHOSE VOICE?
══════════════════════════════════════════
Identify who is speaking:
  (A) The administration itself — it explains, proposes, defends, or justifies its own actions
  (B) An external voice — a journalist, a quoted person, an expert, a citizen, a politician outside the administration

If the only voice is (A) and no external party passes judgment on it → NEUTRAL_STATEMENT.

══════════════════════════════════════════
STEP 3 — WHAT IS THE ACTUAL TARGET?
══════════════════════════════════════════
The judgment must land on the administration's OWN behavior or decisions — not on:
  • A phone call, a negotiation, a meeting
  • A lawsuit, an appeal, a court procedure
  • A private actor, a foreign government, an individual politician in their personal capacity

If the evaluative language targets something other than the administration's conduct → NEUTRAL_STATEMENT.

══════════════════════════════════════════
STEP 4 — IS THERE A REAL JUDGMENT?
══════════════════════════════════════════
A judgment means: an external voice (B) states or implies that the administration acted WRONGLY, FAILED, or acted WELL.

Counts as CRITIC (external voice judging administration negatively):
  ✓ Accusing the administration of wrongdoing, incompetence, or dishonesty
  ✓ Demanding the administration comply with rules or laws (implies it currently doesn't)
  ✓ Recounting personal harmful treatment by officials
  ✓ Calling for change in current institutional practice (implies it is inadequate)
  ✓ Journalist language that explicitly labels: secrecy ("unter Verschluss gehalten", "kept secret"), internal dysfunction ("even insiders no longer believe in the project", "key people have left"), stubbornness as a flaw ("had long clung to X despite warnings"), unnecessary bureaucracy as obstacle, inconsistent/arbitrary treatment of citizens ("outcomes vary depending on the official"), past failure of authorities to act ("what had not worked before")
  ✓ An adjective in the journalist's voice that directly labels the administration as naive, incompetent, or dysfunctional

Does NOT count as CRITIC:
  ✗ An official describing a policy, a plan, or a future goal (even if ambitious or normative)
  ✗ A court issuing a ruling or ordering a review (this is the court doing its job, not criticizing itself)
  ✗ Authorities taking enforcement action (arrests, threats, sanctions) described factually
  ✗ An administration defending or explaining itself in response to criticism
  ✗ A legal scholar stating an institutional attribute (discretionary power, scope of competence)
  ✗ A conditional promise by a politician about future action
  ✗ Standard terminology for a known phenomenon (e.g. "paralysis" for a government shutdown)
  ✗ A request or demand by a private party for the administration to do something (without implying it has failed)

Counts as PRAISE (external voice judging administration positively):
  ✓ Explicit credit or approval of administration's actions
  ✓ Favorable comparison to other administrations ("better than in other countries")

Does NOT count as PRAISE:
  ✗ The administration announcing its own positive actions or decisions
  ✗ A spokesperson explaining a decision in neutral terms

If no judgment present → NEUTRAL_STATEMENT.

══════════════════════════════════════════
STEP 5 — FINAL CHECK
══════════════════════════════════════════
Still uncertain? → NEUTRAL_STATEMENT.

Return ONLY this strict JSON:
{{
  "justification": "<one sentence: name the external voice, the administration targeted, and the judgment — or confirm why no judgment is present>",
  "stance": "CRITIC|PRAISE|NEUTRAL_STATEMENT"
}}

Sentence:
{sentence}
"""


def build_user_prompt(row: pd.Series, text_col: str) -> str:
    sentence = "" if pd.isna(row.get(text_col)) else str(row.get(text_col)).strip()
    return USER_TEMPLATE.format(sentence=sentence)
