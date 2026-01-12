from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import pandas as pd

from swissdox.llm_client import LlamaServerClient


TOPICS = [
    "Foreign Affairs","Culture","Health","Social","Justice","Migration","Defence","Sport",
    "Finance","Economy","Education","Research","Environment","Transports","Energy","Communication","Other",
]
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
POPULISM = ["YES", "NO"]

DEFAULT_RES = {
    "topic": "Other",
    "sentiment_public_admin": "NEUTRAL",
    "sentiment_justification": "",
    "populism": "NO",
    "populism_justification": "",
}

SYSTEM_PROMPT = (
    "You are a strict text classification system.\n"
    "Return ONLY a valid JSON object, with no extra text.\n"
    "Follow the output schema EXACTLY.\n"
)


@dataclass(frozen=True)
class LlmRunConfig:
    temperature: float = 0.0
    top_p: float = 1.0
    max_tokens: int = 180
    max_chars_input: int = 2000

    n_workers: int = 2
    max_retries: int = 2
    max_in_flight_mult: int = 5

    checkpoint_every: int = 50
    print_every: int = 25


LABEL_COLS = [
    "topic_llm",
    "sentiment_public_admin_llm",
    "sentiment_justification_llm",
    "populism_llm",
    "populism_justification_llm",
]


def build_user_prompt(title: str, subhead: str, body: str) -> str:
    return f"""
Analyze the following Swiss news article (title/subtitle/body). You must follow ALL rules.

TASKS

1) Topic recognition
Choose ONE AND ONLY ONE topic from this list:
{TOPICS}

Topic definitions:
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
- Other: not related to these topics OR mainly about another country than Switzerland.

2) Sentiment analysis TOWARDS PUBLIC ADMINISTRATION
Choose ONE label: {SENTIMENTS}
Important:
- Evaluate the tone towards public administration
- Public administration = administrative bodies/agencies/offices, civil servants, bureaucracy, administrative procedures.
- NOT public administration = parties/elected politicians in general, unless the text explicitly evaluates administrative functioning.
- If there is no clear evaluation of public administration, choose NEUTRAL.

Justification rule (STRICT):
- If sentiment_public_admin is POSITIVE or NEGATIVE, you MUST provide a short justification in sentiment_justification.
- If sentiment_public_admin is NEUTRAL, sentiment_justification MUST be an empty string: "".

3) Populism detection
Choose ONE label: {POPULISM}
Definition:
Populism rhetoric such as:
- "the people" vs "the elite/establishment" framing, OR
- anti-institution legitimacy attacks (courts, administration, "system") portrayed as corrupt/illegitimate, OR
- scapegoating an out-group framed as an enemy of "the people".
If not clear, choose NO.

Justification rule (STRICT):
- If populism is YES, you MUST provide a short justification in populism_justification.
- If populism is NO, populism_justification MUST be an empty string: "".

OUTPUT FORMAT (STRICT)
Return ONLY this JSON schema (no additional keys, no markdown):
{{
  "topic": "<one of the topics exactly as written>",
  "sentiment_public_admin": "<POSITIVE|NEGATIVE|NEUTRAL>",
  "sentiment_justification": "<string, or empty string if NEUTRAL>",
  "populism": "<YES|NO>",
  "populism_justification": "<string, or empty string if NO>"
}}

Justification style constraints:
- Max 2 sentences each.
- Must refer to concrete cues in the text.
- Do NOT add quotes longer than 12 words.

ARTICLE
TITLE: {title}
SUBTITLE: {subhead}
BODY: {body}
""".strip()


def clamp_text(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    s = re.sub(r"\s+", " ", s).strip()
    if len(s) <= max_chars:
        return s
    head = s[: int(max_chars * 0.7)]
    tail = s[-int(max_chars * 0.3):]
    return head + " ... " + tail


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not isinstance(text, str):
        return None
    text = text.strip()

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                obj = json.loads(line)
                return obj if isinstance(obj, dict) else None
            except Exception:
                pass

    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def normalize_and_validate(obj: Dict[str, Any]) -> Dict[str, Any]:
    topic = str(obj.get("topic", "")).strip()
    sentiment = str(obj.get("sentiment_public_admin", "")).strip().upper()
    sentiment_just = "" if obj.get("sentiment_justification") is None else str(obj.get("sentiment_justification", "")).strip()

    populism = str(obj.get("populism", "")).strip().upper()
    populism_just = "" if obj.get("populism_justification") is None else str(obj.get("populism_justification", "")).strip()

    if topic not in TOPICS:
        topic = "Other"
    if sentiment not in SENTIMENTS:
        sentiment = "NEUTRAL"
    if populism not in POPULISM:
        populism = "NO"

    if sentiment == "NEUTRAL":
        sentiment_just = ""
    if populism == "NO":
        populism_just = ""

    return {
        "topic": topic,
        "sentiment_public_admin": sentiment,
        "sentiment_justification": sentiment_just,
        "populism": populism,
        "populism_justification": populism_just,
    }


def atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)


def llm_classify_one(
    client: LlamaServerClient,
    model_id: str,
    cfg: LlmRunConfig,
    *,
    title: str,
    subhead: str,
    content: str,
) -> Dict[str, Any]:
    body = clamp_text(content or "", cfg.max_chars_input)
    user_prompt = build_user_prompt(title or "", subhead or "", body)

    for attempt in range(cfg.max_retries + 1):
        try:
            text = client.chat_completion(
                model=model_id,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                temperature=cfg.temperature,
                top_p=cfg.top_p,
                max_tokens=cfg.max_tokens,
            )
            obj = extract_json_object(text)
            if obj is not None:
                return normalize_and_validate(obj)
        except Exception:
            time.sleep(0.4 * (attempt + 1))

    return dict(DEFAULT_RES)


def classify_dataframe_checkpointed(
    df_articles: pd.DataFrame,
    *,
    client: LlamaServerClient,
    model_id: str,
    output_parquet: Path,
    cfg: LlmRunConfig,
    limit: Optional[int] = None,
) -> pd.DataFrame:
    # Resume
    if output_parquet.exists():
        df = pd.read_parquet(output_parquet)
        print(f"✅ Resuming from checkpoint: {output_parquet} shape={df.shape}")
    else:
        df = df_articles.copy()

    if limit is not None:
        df = df.head(limit).copy()

    for c in LABEL_COLS:
        if c not in df.columns:
            df[c] = pd.NA

    pending = [i for i in df.index if pd.isna(df.at[i, "topic_llm"])]
    total_pending = len(pending)
    total_all = len(df)
    already_done = total_all - total_pending
    print(f"📌 Pending: {total_pending} | Already done: {already_done} | Total: {total_all}")

    if total_pending == 0:
        return df

    def payload_for_idx(idx) -> Tuple[str, str, str]:
        row = df.loc[idx]
        return (str(row.get("head", "") or ""), str(row.get("subhead", "") or ""), str(row.get("content", "") or ""))

    max_in_flight = max(1, cfg.n_workers * cfg.max_in_flight_mult)
    done_since_save = 0
    done_total = already_done

    def checkpoint_save():
        atomic_save_parquet(df, output_parquet)
        print(f"💾 checkpoint saved -> {output_parquet}")

    with ThreadPoolExecutor(max_workers=cfg.n_workers) as ex:
        it = iter(pending)
        future_to_idx: Dict[Any, Any] = {}

        for _ in range(min(max_in_flight, total_pending)):
            idx = next(it, None)
            if idx is None:
                break
            t, sh, c = payload_for_idx(idx)
            future_to_idx[ex.submit(llm_classify_one, client, model_id, cfg, title=t, subhead=sh, content=c)] = idx

        while future_to_idx:
            done_set, _ = wait(future_to_idx.keys(), return_when=FIRST_COMPLETED)

            for fut in done_set:
                idx = future_to_idx.pop(fut)
                try:
                    res = fut.result()
                except Exception:
                    res = dict(DEFAULT_RES)

                df.at[idx, "topic_llm"] = res["topic"]
                df.at[idx, "sentiment_public_admin_llm"] = res["sentiment_public_admin"]
                df.at[idx, "sentiment_justification_llm"] = res["sentiment_justification"]
                df.at[idx, "populism_llm"] = res["populism"]
                df.at[idx, "populism_justification_llm"] = res["populism_justification"]

                done_since_save += 1
                done_total += 1

                if done_total % cfg.print_every == 0 or done_total == total_all:
                    print(f"… classified {done_total}/{total_all} (session: {done_since_save}/{total_pending})")

                if done_since_save % cfg.checkpoint_every == 0:
                    checkpoint_save()

                nxt = next(it, None)
                if nxt is not None:
                    t, sh, c = payload_for_idx(nxt)
                    future_to_idx[ex.submit(llm_classify_one, client, model_id, cfg, title=t, subhead=sh, content=c)] = nxt

    checkpoint_save()
    return df
