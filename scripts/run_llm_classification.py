# ============================================================
# 2) LLM CLASSIFICATION via llama-server (local OpenAI-compatible HTTP)
#    Adds columns: topic / sentiment / populism
#    + Overnight-safe: retries, checkpointing, resume
#    + AUTO: chunking + automatic process restart (real RAM reset)
# ============================================================

import json
import re
import time
import sys
import os
import subprocess
from pathlib import Path
from typing import Dict, Any, Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import pandas as pd

import sys
print("PYTHON EXECUTABLE:", sys.executable)
print("PYTHON VERSION:", sys.version)


# -------------------------------
# 0) SERVER CONFIG
# -------------------------------
LLAMA_BASE_URL = "http://127.0.0.1:8080"
CHAT_URL = f"{LLAMA_BASE_URL}/v1/chat/completions"
MODELS_URL = f"{LLAMA_BASE_URL}/v1/models"

REQ_TIMEOUT = 120
SESSION = requests.Session()

retry = Retry(
    total=5,
    backoff_factor=0.5,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET", "POST"],
    raise_on_status=False,
)
adapter = HTTPAdapter(max_retries=retry, pool_connections=50, pool_maxsize=50)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)


# -------------------------------
# 1) GENERATION CONFIG
# -------------------------------
TEMPERATURE = 0.0
TOP_P = 1.0
MAX_TOKENS = 180
MAX_CHARS_INPUT = 4000

N_WORKERS = 2
MAX_RETRIES = 2
MAX_IN_FLIGHT_MULT = 2


# -------------------------------
# 1b) CHECKPOINT / RESUME / AUTO-RUN
# -------------------------------
CSV_PATH = "df_articles.csv"

OUTPUT_PARQUET = Path("Swissdox_articles_labeled.parquet")
CHECKPOINT_EVERY = 25
PRINT_EVERY = 25

# ✅ Auto overnight settings (tune)
CHUNK_SIZE = 25          # how many articles per worker run (300-600 good)
WORKER_RETRIES = 3        # if worker crashes, parent retries
WORKER_BACKOFF_SEC = 30   # wait before retry


# -------------------------------
# 2) LABEL SPACE (STRICT)
# -------------------------------
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
SENTIMENTS = ["POSITIVE", "NEGATIVE", "NEUTRAL"]
POPULISM = ["YES", "NO"]


# -------------------------------
# 3) PROMPT (STRICT JSON output)
# -------------------------------
SYSTEM_PROMPT = (
    "You are a strict text classification system.\n"
    "Return ONLY a valid JSON object, with no extra text.\n"
    "Follow the output schema EXACTLY.\n"
)

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


# -------------------------------
# 4) HELPERS
# -------------------------------
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
    sentiment_just = obj.get("sentiment_justification", "")
    sentiment_just = "" if sentiment_just is None else str(sentiment_just).strip()

    populism = str(obj.get("populism", "")).strip().upper()
    populism_just = obj.get("populism_justification", "")
    populism_just = "" if populism_just is None else str(populism_just).strip()

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

DEFAULT_RES = {
    "topic": "Other",
    "sentiment_public_admin": "NEUTRAL",
    "sentiment_justification": "",
    "populism": "NO",
    "populism_justification": "",
}

LABEL_COLS = [
    "topic_llm",
    "sentiment_public_admin_llm",
    "sentiment_justification_llm",
    "populism_llm",
    "populism_justification_llm",
]


# -------------------------------
# 5) SERVER CALL
# -------------------------------
def get_server_model_id() -> Optional[str]:
    try:
        r = SESSION.get(MODELS_URL, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        models = data.get("data", [])
        if models and isinstance(models, list):
            return models[0].get("id")
    except Exception:
        pass
    return None

MODEL_ID = get_server_model_id() or "local-model"
print(f"✅ llama-server reachable, using model='{MODEL_ID}'")

def chat_completion(system_prompt: str, user_prompt: str) -> str:
    payload = {
        "model": MODEL_ID,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "max_tokens": MAX_TOKENS,
    }
    r = SESSION.post(CHAT_URL, json=payload, timeout=REQ_TIMEOUT)
    r.raise_for_status()
    j = r.json()
    return j["choices"][0]["message"]["content"]

def llm_classify_one(title: str, subhead: str, content: str) -> Dict[str, Any]:
    body = clamp_text(content or "", MAX_CHARS_INPUT)
    user_prompt = build_user_prompt(title or "", subhead or "", body)

    for attempt in range(MAX_RETRIES + 1):
        try:
            text = chat_completion(SYSTEM_PROMPT, user_prompt)
            obj = extract_json_object(text)
            if obj is not None:
                return normalize_and_validate(obj)
        except Exception:
            time.sleep(0.4 * (attempt + 1))
    return dict(DEFAULT_RES)


# -------------------------------
# 6) CHECKPOINT UTIL
# -------------------------------
def atomic_save_parquet(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(path)

def ensure_checkpoint_exists() -> None:
    if OUTPUT_PARQUET.exists():
        return
    df = pd.read_csv(CSV_PATH)
    for c in LABEL_COLS:
        df[c] = pd.NA
    atomic_save_parquet(df, OUTPUT_PARQUET)
    print(f"💾 Created initial checkpoint -> {OUTPUT_PARQUET}")


# -------------------------------
# 7) WORKER: process ONE chunk then exit
# -------------------------------
def worker_process_one_chunk() -> int:
    """
    Loads checkpoint parquet, finds pending rows, processes up to CHUNK_SIZE,
    saves checkpoint, then exits.
    Returns 0 if work done, 1 if nothing left.
    """
    df = pd.read_parquet(OUTPUT_PARQUET)

    # pending = rows where topic_llm is NA
    pending = [i for i in df.index if pd.isna(df.at[i, "topic_llm"])]
    total_pending = len(pending)
    total_all = len(df)
    already_done = total_all - total_pending

    if total_pending == 0:
        print("✅ Nothing left to do.")
        return 1

    chunk_indices = pending[:CHUNK_SIZE]
    print(f"🚀 Worker chunk: {len(chunk_indices)} rows | pending={total_pending} | done={already_done} | total={total_all}")

    def payload_for_idx(idx) -> Tuple[str, str, str]:
        row = df.loc[idx]
        return (
            str(row.get("head", "") or ""),
            str(row.get("subhead", "") or ""),
            str(row.get("content", "") or ""),
        )

    max_in_flight = max(1, N_WORKERS * MAX_IN_FLIGHT_MULT)
    done_since_save = 0
    done_total = already_done

    def checkpoint_save():
        atomic_save_parquet(df, OUTPUT_PARQUET)
        print(f"💾 checkpoint saved -> {OUTPUT_PARQUET}")

    with ThreadPoolExecutor(max_workers=N_WORKERS) as ex:
        it = iter(chunk_indices)
        future_to_idx: Dict[Any, Any] = {}

        # prime
        for _ in range(min(max_in_flight, len(chunk_indices))):
            idx = next(it, None)
            if idx is None:
                break
            t, sh, c = payload_for_idx(idx)
            future_to_idx[ex.submit(llm_classify_one, t, sh, c)] = idx

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
                df.at[idx, "sentiment_justification_llm"] = res.get("sentiment_justification", "")
                df.at[idx, "populism_llm"] = res["populism"]
                df.at[idx, "populism_justification_llm"] = res.get("populism_justification", "")

                done_since_save += 1
                done_total += 1

                if done_total % PRINT_EVERY == 0:
                    print(f"… classified {done_total}/{total_all} (this chunk: {done_since_save}/{len(chunk_indices)})")

                if done_since_save % CHECKPOINT_EVERY == 0:
                    checkpoint_save()

                # refill
                nxt = next(it, None)
                if nxt is not None:
                    t, sh, c = payload_for_idx(nxt)
                    future_to_idx[ex.submit(llm_classify_one, t, sh, c)] = nxt

    checkpoint_save()
    print("✅ Worker finished chunk. Exiting to free RAM.")
    return 0


# -------------------------------
# 8) PARENT: automatic overnight loop
# -------------------------------
def parent_run_overnight() -> None:
    ensure_checkpoint_exists()

    # Need script path for subprocess
    script_path = Path(__file__).resolve()

    print("🌙 Parent started. Will run workers until completion (automatic).")
    while True:
        # quick check pending before launching worker
        df = pd.read_parquet(OUTPUT_PARQUET)
        pending = int(df["topic_llm"].isna().sum()) if "topic_llm" in df.columns else len(df)
        if pending == 0:
            print("🎉 All done! No pending rows left.")
            break

        # launch worker with retries
        attempt = 0
        while True:
            attempt += 1
            print(f"🧩 Launching worker (attempt {attempt}/{WORKER_RETRIES})… pending={pending}")
            try:
                subprocess.run(
                    [sys.executable, str(script_path), "--worker"],
                    check=True,
                )
                break  # worker succeeded
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Worker crashed: {e}")
                if attempt >= WORKER_RETRIES:
                    raise
                print(f"⏳ Waiting {WORKER_BACKOFF_SEC}s then retry…")
                time.sleep(WORKER_BACKOFF_SEC)


# -------------------------------
# 9) ENTRYPOINT
# -------------------------------
if __name__ == "__main__":
    # Run as:
    #   python your_script.py          -> parent (automatic overnight)
    #   python your_script.py --worker -> processes one chunk then exits
    if "--worker" in sys.argv:
        # Worker mode
        ensure_checkpoint_exists()
        rc = worker_process_one_chunk()
        sys.exit(0 if rc == 0 else 0)
    else:
        # Parent mode (one command, fully automatic)
        parent_run_overnight()
