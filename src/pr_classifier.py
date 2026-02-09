from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from client import LLMClient, TransformersClient, TransformersConfig
from pr_prompt import SYSTEM_MESSAGE, TASK_INSTRUCTIONS

_JSON_ARRAY_RE = re.compile(r"\[[\s\S]*\]", re.MULTILINE)


@dataclass(frozen=True)
class PRConfig:
    in_parquet: Path
    out_dir: Path
    scratch_dir: Path

    # batching (GPU-optimized)
    items_per_prompt: int = 60        # sentences per prompt
    prompts_per_batch: int = 8        # prompts generated together on GPU

    # LLM
    max_tokens: int = 250
    temperature: float = 0.0

    # robustness
    max_retries: int = 3
    retry_sleep_s: float = 2.0

    # resume
    resume: bool = True

    # outputs
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
    # Keep TASK_INSTRUCTIONS EXACT. Only add input wrapper.
    lines = ["SENTENCES:"]
    for sid, sent in items:
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

    # build todo list
    todo: List[Tuple[str, str]] = []
    for sid, sent in zip(df["sentence_id"].tolist(), df["sentence"].tolist()):
        if sid in done:
            continue
        if sent.strip():
            todo.append((sid, sent))

    results: Dict[str, int] = dict(done)
    print(
        f"[PR] to_do={len(todo)} items_per_prompt={cfg.items_per_prompt} "
        f"prompts_per_batch={cfg.prompts_per_batch}"
    )

    # Split todo into prompts of items_per_prompt
    prompts: List[List[Tuple[str, str]]] = [
        todo[i : i + cfg.items_per_prompt]
        for i in range(0, len(todo), cfg.items_per_prompt)
    ]

    # Process groups of prompts_per_batch on GPU
    for gi in range(0, len(prompts), cfg.prompts_per_batch):
        prompt_group = prompts[gi : gi + cfg.prompts_per_batch]
        batch_messages = [_build_messages(p) for p in prompt_group]

        last_err: Optional[Exception] = None
        for attempt in range(1, cfg.max_retries + 1):
            try:
                # Need batched backend
                if not hasattr(llm, "chat_batch"):
                    raise RuntimeError("LLM client does not implement chat_batch().")

                outputs = llm.chat_batch(  # type: ignore[attr-defined]
                    batch_messages=batch_messages,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )

                # Parse each output
                for p_items, raw in zip(prompt_group, outputs):
                    parsed = _parse_output(raw)
                    got = {d["id"]: int(d["pr"]) for d in parsed}
                    for sid, _sent in p_items:
                        results[sid] = int(got.get(sid, 0))

                last_err = None
                break

            except Exception as e:
                last_err = e
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_sleep_s * attempt)
                else:
                    # fallback: conservative 0 for all items in this group
                    for p_items in prompt_group:
                        for sid, _sent in p_items:
                            results[sid] = 0

        if last_err is not None:
            print(f"[PR][WARN] group={gi//cfg.prompts_per_batch} failed -> fallback 0. err={last_err}")

        # checkpoint every ~10 groups
        if (gi // cfg.prompts_per_batch) % 10 == 0:
            _save_ckpt(cfg, results)
            print(f"[PR] checkpoint saved: {len(results)}")

    # merge back
    df["pr"] = df["sentence_id"].map(lambda x: int(results.get(str(x), 0)))

    out_parquet = cfg.out_dir / cfg.out_parquet_name
    out_csv = cfg.out_dir / cfg.out_csv_name
    df.to_parquet(out_parquet, index=False)
    df.to_csv(out_csv, index=False)

    # final checkpoint full
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