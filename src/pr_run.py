from __future__ import annotations
import json, time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import pandas as pd

from llm_client import TransformersClient, TransformersConfig
from pr_prompt import SYSTEM_MESSAGE, TASK_INSTRUCTIONS


TOPICS = {
    "Foreign Affairs", "Culture", "Health", "Social", "Justice", "Migration",
    "Defence", "Sport", "Finance", "Economy", "Education", "Research",
    "Environment", "Transports", "Energy", "Communication", "Other",
}

@dataclass(frozen=True)
class Cfg:
    inp: Path
    outdir: Path
    scratch: Path
    model_path: str
    dtype: str = "bf16"
    # IMPORTANT: smaller per-prompt batch, because your instructions are long
    items_per_prompt: int = 12
    prompts_per_batch: int = 6
    # IMPORTANT: output is bigger now (t + sp + p), so allow more tokens
    max_tokens: int = 900
    temperature: float = 0.0
    resume: bool = True


def _ckpt(p: Path) -> Path:
    return p / "tsp_ckpt.parquet"


def _faildir(p: Path) -> Path:
    return p / "tsp_failures"


def _load_ckpt(p: Path) -> Dict[str, Tuple[str, int, int]]:
    f = _ckpt(p)
    if not f.exists():
        return {}
    d = pd.read_parquet(f)
    need = {"sentence_id", "t", "sp", "p"}
    if not need.issubset(d.columns):
        return {}

    out: Dict[str, Tuple[str, int, int]] = {}
    for sid, t, sp, pp in zip(
        d["sentence_id"].astype(str),
        d["t"].astype(str),
        d["sp"].astype(int),
        d["p"].astype(int),
    ):
        if t not in TOPICS:
            t = "Other"
        sp = int(sp)
        if sp not in (-1, 0, 1):
            sp = 0
        pp = 1 if int(pp) == 1 else 0
        out[str(sid)] = (t, sp, pp)
    return out


def _save_ckpt(p: Path, m: Dict[str, Tuple[str, int, int]]) -> None:
    p.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(
        {
            "sentence_id": list(m.keys()),
            "t": [v[0] for v in m.values()],
            "sp": [int(v[1]) for v in m.values()],
            "p": [int(v[2]) for v in m.values()],
        }
    )
    df.to_parquet(_ckpt(p), index=False)


def _extract_json_array(text: str) -> list:
    """
    Robustly extract a JSON array from model output, even if extra text exists.
    Strategy: try to parse from every '[' to a matching ']' by incremental scan.
    """
    s = text.strip()

    # Fast path: if it already looks like a JSON array
    if s.startswith("["):
        try:
            return json.loads(s)
        except Exception:
            pass

    starts = [i for i, ch in enumerate(s) if ch == "["]
    if not starts:
        raise ValueError("no '[' found")

    # Try progressively larger slices starting from each '['
    for st in starts:
        for ed in range(len(s) - 1, st, -1):
            if s[ed] != "]":
                continue
            chunk = s[st:ed + 1]
            try:
                arr = json.loads(chunk)
                if isinstance(arr, list):
                    return arr
            except Exception:
                continue

    raise ValueError("could not parse any JSON array")


def _parse(txt: str) -> Dict[str, Tuple[str, int, int]]:
    arr = _extract_json_array(txt)

    out: Dict[str, Tuple[str, int, int]] = {}
    for el in arr:
        sid = str(el["id"])

        t = str(el["t"])
        if t not in TOPICS:
            t = "Other"

        sp = int(el["sp"])
        if sp not in (-1, 0, 1):
            sp = 0

        pp = int(el["p"])
        pp = 1 if pp == 1 else 0

        out[sid] = (t, sp, pp)

    return out


def _user_prompt(items: List[Tuple[str, str]]) -> str:
    lines = ["SENTENCES (id\\ttext):"] + [f"{sid}\t{s}" for sid, s in items]
    return TASK_INSTRUCTIONS + "\n\n" + "\n".join(lines)


def _write_failure(scratch: Path, tag: str, user_prompt: str, raw: str) -> None:
    d = _faildir(scratch)
    d.mkdir(parents=True, exist_ok=True)
    (d / f"{tag}_user_prompt.txt").write_text(user_prompt, encoding="utf-8")
    (d / f"{tag}_raw_output.txt").write_text(raw, encoding="utf-8")


def run(cfg: Cfg) -> None:
    cfg.outdir.mkdir(parents=True, exist_ok=True)
    cfg.scratch.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(cfg.inp)
    df["sentence_id"] = df["sentence_id"].astype(str)
    df["sentence"] = df["sentence"].fillna("").astype(str)

    done = _load_ckpt(cfg.scratch) if cfg.resume else {}
    todo = [(sid, s) for sid, s in zip(df["sentence_id"], df["sentence"]) if sid not in done and s.strip()]
    res: Dict[str, Tuple[str, int, int]] = dict(done)

    client = TransformersClient(TransformersConfig(cfg.model_path, dtype=cfg.dtype))

    prompts = [todo[i:i + cfg.items_per_prompt] for i in range(0, len(todo), cfg.items_per_prompt)]
    t0 = time.time()

    for gi in range(0, len(prompts), cfg.prompts_per_batch):
        grp = prompts[gi:gi + cfg.prompts_per_batch]
        ups = [_user_prompt(p) for p in grp]

        ok = False
        for att in range(3):
            try:
                outs = client.chat_many(
                    SYSTEM_MESSAGE,
                    ups,
                    temperature=cfg.temperature,
                    max_tokens=cfg.max_tokens,
                )

                for j, (items, up, raw) in enumerate(zip(grp, ups, outs)):
                    got = _parse(raw)
                    for sid, _ in items:
                        res[sid] = got.get(sid, ("Other", 0, 0))

                ok = True
                break

            except Exception as e:
                # On first failure, store one concrete example for debugging
                try:
                    tag = f"g{gi}_att{att}"
                    _write_failure(cfg.scratch, tag, ups[0] if ups else "", outs[0] if 'outs' in locals() and outs else f"EXC: {e}")
                except Exception:
                    pass
                time.sleep(2 * (att + 1))

        if not ok:
            # Fallback only if *everything* failed after retries
            for items in grp:
                for sid, _ in items:
                    res[sid] = ("Other", 0, 0)

        if (gi // cfg.prompts_per_batch) % 5 == 0:
            _save_ckpt(cfg.scratch, res)
            done_n = min((gi * cfg.items_per_prompt), len(todo))
            print(f"[TSP] groups={gi//cfg.prompts_per_batch} done≈{done_n}/{len(todo)} elapsed_s={time.time()-t0:.1f}")

    df["t"] = df["sentence_id"].map(lambda x: res.get(str(x), ("Other", 0, 0))[0])
    df["sp"] = df["sentence_id"].map(lambda x: int(res.get(str(x), ("Other", 0, 0))[1]))
    df["p"] = df["sentence_id"].map(lambda x: int(res.get(str(x), ("Other", 0, 0))[2]))

    out_parq = cfg.outdir / "swissdox_sentences_with_t_sp_p.parquet"
    out_csv = cfg.outdir / "swissdox_sentences_with_t_sp_p.csv"
    df.to_parquet(out_parq, index=False)
    df.to_csv(out_csv, index=False)

    _save_ckpt(cfg.scratch, dict(zip(df["sentence_id"].astype(str), zip(df["t"].astype(str), df["sp"].astype(int), df["p"].astype(int)))))
    print(f"[DONE] {out_parq} | {out_csv} | ckpt={_ckpt(cfg.scratch)}")
    print(f"[DEBUG] failures (if any) in: {_faildir(cfg.scratch)}")