from __future__ import annotations
import json, re, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from llm_client import TransformersClient, TransformersConfig
from pr_prompt import SYSTEM_MESSAGE, TASK_INSTRUCTIONS

RE_ARR = re.compile(r"\[[\s\S]*\]")

@dataclass(frozen=True)
class Cfg:
    inp: Path
    outdir: Path
    scratch: Path
    model_path: str
    dtype: str = "bf16"
    items_per_prompt: int = 60
    prompts_per_batch: int = 8
    max_tokens: int = 250
    temperature: float = 0.0
    resume: bool = True

def _ckpt(p: Path) -> Path: return p / "pr_ckpt.parquet"

def _load_ckpt(p: Path) -> Dict[str, int]:
    f = _ckpt(p)
    if not f.exists(): return {}
    d = pd.read_parquet(f)
    if not {"sentence_id","pr"}.issubset(d.columns): return {}
    return dict(zip(d["sentence_id"].astype(str), d["pr"].astype(int)))

def _save_ckpt(p: Path, m: Dict[str,int]) -> None:
    p.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"sentence_id": list(m.keys()), "pr": list(m.values())}).to_parquet(_ckpt(p), index=False)

def _parse(txt: str) -> Dict[str,int]:
    m = RE_ARR.search(txt.strip())
    if not m: raise ValueError("no json array")
    arr = json.loads(m.group(0))
    out = {}
    for el in arr:
        sid = str(el["id"])
        pr = int(el["pr"])
        out[sid] = 1 if pr == 1 else 0
    return out

def _user_prompt(items: List[Tuple[str,str]]) -> str:
    # compact input, less tokens than sentence="..."
    lines = ["SENTENCES (id\\ttext):"] + [f"{sid}\t{s}" for sid,s in items]
    return TASK_INSTRUCTIONS + "\n\n" + "\n".join(lines)

def run(cfg: Cfg) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.scratch.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.inp)
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["sentence"] = df["sentence"].fillna("").astype(str)

    done = _load_ckpt(cfg.scratch) if cfg.resume else {}
    todo = [(sid, s) for sid, s in zip(df["sentence_id"], df["sentence"]) if sid not in done and s.strip()]
    res: Dict[str,int] = dict(done)

    client = TransformersClient(TransformersConfig(cfg.model_path, dtype=cfg.dtype))

    prompts = [todo[i:i+cfg.items_per_prompt] for i in range(0, len(todo), cfg.items_per_prompt)]
    t0 = time.time()

    for gi in range(0, len(prompts), cfg.prompts_per_batch):
        grp = prompts[gi:gi+cfg.prompts_per_batch]
        ups = [_user_prompt(p) for p in grp]

        # retry group (fallback pr=0 on failure)
        ok = False
        for att in range(3):
            try:
                outs = client.chat_many(SYSTEM_MESSAGE, ups, temperature=cfg.temperature, max_tokens=cfg.max_tokens)
                for items, raw in zip(grp, outs):
                    got = _parse(raw)
                    for sid,_ in items:
                        res[sid] = got.get(sid, 0)
                ok = True
                break
            except Exception:
                time.sleep(2*(att+1))
        if not ok:
            for items in grp:
                for sid,_ in items:
                    res[sid] = 0

        if (gi // cfg.prompts_per_batch) % 10 == 0:
            _save_ckpt(cfg.scratch, res)
            print(f"[PR] groups={gi//cfg.prompts_per_batch} done≈{min((gi*cfg.items_per_prompt), len(todo))}/{len(todo)} elapsed_s={time.time()-t0:.1f}")

    df["pr"] = df["sentence_id"].map(lambda x: int(res.get(str(x), 0)))
    out_parq = cfg.outdir / "swissdox_sentences_with_pr.parquet"
    out_csv  = cfg.outdir / "swissdox_sentences_with_pr.csv"
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)
    _save_ckpt(cfg.scratch, dict(zip(df["sentence_id"].astype(str), df["pr"].astype(int))))
    print(f"[DONE] {out_parq} | {out_csv} | ckpt={_ckpt(cfg.scratch)}")
