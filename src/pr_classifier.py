from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from client import TransformersClient, TransformersConfig, LLMClient
from pr_prompt import SYSTEM_MESSAGE, TASK_INSTRUCTIONS

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]", re.MULTILINE)


@dataclass(frozen=True)
class PRConfig:
    in_parquet: Path
    out_dir: Path
    scratch_dir: Path

    batch_size: int = 60
    max_tokens: int = 250
    temperature: float = 0.0

    max_retries: int = 3
    retry_sleep_s: float = 2.0

    resume: bool = True

    out_parquet_name: str = "swissdox_sentences_with_pr.parquet"
    out_csv_name: str = "swissdox_sentences_with_pr.csv"
    ckpt_name: str = "pr_checkpoint.parquet"


def _extract_json_array(text: str) -> str:
    m = _JSON_ARRAY_RE.search(text.strip())
    if not m:
        raise ValueError("No JSON array found in model output.")
    return m.group(0)


def _parse_output(text: str) -> List[Dict[str, Any]]:
    arr_txt = _extract_json_array(text)
    obj = json.loads(arr_txt)
    if not isinstance(obj, list):
        raise ValueError("Output is not a JSON array.")
    out: List[Dict[str, Any]] = []
    for el in obj:
        if not isinstance(el, dict) or "id" not in el or "pr" not in el:
            raise ValueError("Bad element schema (need id, pr).")
        pr = int(el["pr"])
        if pr not in (0, 1):
            raise ValueError("pr must be 0 or 1.")
        out.append({"id": str(el["id"]), "pr": pr})
    return out


def _build_messages(items: List[Tuple[str, str]]) -> List[Dict[str, str]]:
    # On garde la description EXACTE. On ajoute seulement l'input wrapper.
    lines = ["SENTENCES:"]
    for sid, sent in items:
        # input stable: id + texte
        lines.append(f'- id="{sid}" sentence="{sent}"')
    content = TASK_INSTRUCTIONS + "\n\n" + "\n".join(lines)
    return [
        {"role": "system", "content": SYSTEM_MESSAGE},
        {"role": "user", "content": content},
    ]


def _load_df(path: Path) -> pd.DataFrame:
    df = pd.read_parquet(path)
    if "sentence_id" not in df.columns or "sentence" not in df.columns:
        raise ValueError("Input parquet must contain sentence_id and sentence.")
    df = df.copy()
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["sentence"] = df["sentence"].fillna("").astype(str)
    return df


def _ckpt_path(cfg: PRConfig) -> Path:
    # checkpoint sur scratch (rapide), pour reprise
    return cfg.scratch_dir / cfg.ckpt_name


def _load_ckpt(cfg: PRConfig) -> Dict[str, int]:
    p = _ckpt_path(cfg)
    if not p.exists():
        return {}
    ck = pd.read_parquet(p)
    if not {"sentence_id", "pr"}.issubset(ck.columns):
        return {}
    return dict(zip(ck["sentence_id"].astype(str), ck["pr"].astype(int)))


def _save_ckpt(cfg: PRConfig, mapping: Dict[str, int]) -> None:
    p = _ckpt_path(cfg)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sentence_id": list(mapping.keys()), "pr": list(mapping.values())}).to_parquet(p, index=False)


def classify_pr(cfg: PRConfig, llm: LLMClient) -> Dict[str, Path]:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    cfg.scratch_dir.mkdir(parents=True, exist_ok=True)

    df = _load_df(cfg.in_parquet)

    done: Dict[str, int] = _load_ckpt(cfg) if cfg.resume else {}
    print(f"[PR] total={len(df)} already_done={len(done)}")

    # worklist (skip done)
    todo: List[Tuple[str, str]] = []
    for sid, sent in zip(df["sentence_id"].tolist(), df["sentence"].tolist()):
        if sid in done:
            continue
        if sent.strip():
            todo.append((sid, sent))

    results: Dict[str, int] = dict(done)
    print(f"[PR] to_do={len(todo)} batch_size={cfg.batch_size}")

    for bi in range(0, len(todo), cfg.batch_size):
        batch = todo[bi : bi + cfg.batch_size]
        messages = _build_messages(batch)

        last_err: Optional[Exception] = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                raw = llm.chat(
                    messages=messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )
                parsed = _parse_output(raw)
                got = {d["id"]: int(d["pr"]) for d in parsed}

                # coverage: si un id manque -> 0 (conservateur)
                for sid, _sent in batch:
                    results[sid] = int(got.get(sid, 0))

                last_err = None
                break
            except Exception as e:
                last_err = e
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_sleep_s * attempt)
                else:
                    for sid, _sent in batch:
                        results[sid] = 0

        if last_err is not None:
            print(f"[PR][WARN] batch_index={bi//cfg.batch_size} failed -> fallback 0. err={last_err}")

        # checkpoint toutes les ~10 batches
        if (bi // cfg.batch_size) % 10 == 0:
            _save_ckpt(cfg, results)
            print(f"[PR] checkpoint saved: {len(results)}")

    df["pr"] = df["sentence_id"].map(lambda x: int(results.get(str(x), 0)))

    out_parquet = cfg.out_dir / cfg.out_parquet_name
    out_csv = cfg.out_dir / cfg.out_csv_name
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # dernier checkpoint complet
    _save_ckpt(cfg, dict(zip(df["sentence_id"].astype(str), df["pr"].astype(int))))

    return {"out_parquet": out_parquet, "out_csv": out_csv, "checkpoint": _ckpt_path(cfg)}


def build_apertus_client(model_path: str, dtype: str = "bf16") -> TransformersClient:
    return TransformersClient(
        TransformersConfig(
            model_path=model_path,
            device="cuda",
            dtype=dtype,
            trust_remote_code=True,
        )
    )
