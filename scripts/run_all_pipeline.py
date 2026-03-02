# scripts/run_all_pipeline.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import json
import re
import pandas as pd

from src.client_src import TransformersClient, LLMConfig
from src.runner_src import run_llm_dataframe, RunConfig

import run_all_prompts as run_all_prompts


_ALLOWED_TOPICS = set(run_all_prompts.TOPICS)
_ALLOWED_STANCES = set(run_all_prompts.STANCES)
# dept can also be NONE in your corrected prompt
_ALLOWED_DEPTS = set(run_all_prompts.DEPARTMENTS) | {"NONE"}


def _extract_json_object(raw: str) -> dict | None:
    """Extract first JSON object from raw text (tolerant)."""
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    m = re.search(r"\{.*\}", s, flags=re.DOTALL)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except Exception:
        return None


def _norm_yes_no(x) -> str | pd._libs.missing.NAType:
    if x is None:
        return pd.NA
    s = str(x).strip().upper()
    m = re.findall(r"\b(YES|NO)\b", s)
    if not m:
        return pd.NA
    return m[-1]


def _norm_topic(x) -> str | pd._libs.missing.NAType:
    if x is None:
        return pd.NA
    s = str(x).strip().strip('"').strip("'").strip()
    if s in _ALLOWED_TOPICS:
        return s
    # try to recover from noisy outputs by matching any known topic token
    for t in sorted(_ALLOWED_TOPICS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", s):
            return t
    return pd.NA


def _norm_stance(x) -> str | pd._libs.missing.NAType:
    if x is None:
        return pd.NA
    s = str(x).strip().upper()

    # accept small variants
    replacements = {
        "CRITIQUE": "CRITICISM",
        "CRITIC": "CRITICISM",
        "NEGATIVE": "CRITICISM",
        "POSITIVE": "PRAISE",
        "NEUTRE": "NEUTRAL",
    }
    s = replacements.get(s, s)

    for st in ("CRITICISM", "PRAISE", "NEUTRAL"):
        if re.search(rf"\b{st}\b", s):
            return st
    return pd.NA


def _norm_dept_or_none(x) -> str | pd._libs.missing.NAType:
    """
    Normalize dept to one of DFAE/DFI/DFJP/DDPS/DFF/DEFR/DETEC/NONE.
    Returns NA if nothing usable.
    """
    if x is None:
        return pd.NA
    s = str(x).strip().upper().replace(".", "").replace(" ", "")
    # treat JSON null-like strings
    if s in {"NULL", "NONE", "N/A", "NA"}:
        s = "NONE"
    if s in _ALLOWED_DEPTS:
        return s
    # try to find any allowed token in a noisy string
    for d in sorted(_ALLOWED_DEPTS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(d)}\b", s):
            return d
    return pd.NA


def _norm_int_101(x) -> int | pd._libs.missing.NAType:
    if x is None:
        return pd.NA
    s = str(x).strip()
    m = re.findall(r"(?<!\d)(-1|0|1)(?!\d)", s)
    if not m:
        return pd.NA
    return int(m[-1])


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")

    # output columns (schema)
    ap.add_argument("--non_swiss_col", default="NON_SWISS")
    ap.add_argument("--stance_col", default="STANCE")
    ap.add_argument("--dept_col", default="DEPARTMENT")
    ap.add_argument("--topic_col", default="TOPIC")
    ap.add_argument("--populism_col", default="POPULISM")

    # behavior knobs
    ap.add_argument(
        "--only_fill_if_non_swiss_no",
        action="store_true",
        help="If set: when non_swiss=YES, set stance/dept/topic/populism to NA (short-circuit non-swiss rows).",
    )

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=128)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if args.text_col not in df.columns:
        raise KeyError(f"Missing text column: {args.text_col}")

    # send all non-empty sentences
    send_mask = df[args.text_col].notna() & (df[args.text_col].astype(str).str.strip() != "")

    # Ensure output cols exist
    for c in (args.non_swiss_col, args.stance_col, args.dept_col, args.topic_col):
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            df[c] = df[c].astype("string")

    if args.populism_col not in df.columns:
        df[args.populism_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    else:
        try:
            df[args.populism_col] = df[args.populism_col].astype("Int64")
        except Exception:
            df[args.populism_col] = pd.Series(df[args.populism_col], index=df.index, dtype="Int64")

    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
        )
    )

    run_cfg = RunConfig(
        id_col="row_uid" if "row_uid" in df.columns else "__index__",
        text_col=args.text_col,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    def _select_mask(df_: pd.DataFrame) -> pd.Series:
        return send_mask

    def _build_prompt(row: pd.Series, text_col: str) -> str:
        return run_all_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        obj = _extract_json_object(raw)
        if not isinstance(obj, dict):
            # conservative fallbacks
            return {
                args.non_swiss_col: pd.NA,
                args.stance_col: "NEUTRAL",
                args.dept_col: pd.NA,
                args.topic_col: "Other",
                args.populism_col: pd.NA,
            }

        non_swiss = _norm_yes_no(obj.get("non_swiss"))
        stance = _norm_stance(obj.get("stance"))
        dept = _norm_dept_or_none(obj.get("dept"))
        topic = _norm_topic(obj.get("topic"))
        populism = _norm_int_101(obj.get("populism"))

        # default stance if missing/unparseable
        if pd.isna(stance):
            stance = "NEUTRAL"

        # Optional short-circuit
        if args.only_fill_if_non_swiss_no and non_swiss == "YES":
            return {
                args.non_swiss_col: non_swiss,
                args.stance_col: pd.NA,
                args.dept_col: pd.NA,
                args.topic_col: pd.NA,
                args.populism_col: pd.NA,
            }

        # Apply corrected schema logic (aligned with prompt):
        # - NEUTRAL => topic required, dept null, populism null
        # - PRAISE  => dept required OR NONE, topic null, populism null
        # - CRITICISM => dept required OR NONE, populism required, topic null
        if stance == "NEUTRAL":
            if pd.isna(topic):
                topic = "Other"
            dept = pd.NA
            populism = pd.NA

        elif stance == "PRAISE":
            topic = pd.NA
            populism = pd.NA
            if pd.isna(dept):
                dept = "NONE"

        elif stance == "CRITICISM":
            topic = pd.NA
            if pd.isna(dept):
                dept = "NONE"
            if pd.isna(populism):
                # If model misses populism, default to -1 (not populist) rather than NA
                populism = -1

        else:
            # safety fallback
            stance = "NEUTRAL"
            if pd.isna(topic):
                topic = "Other"
            dept = pd.NA
            populism = pd.NA

        return {
            args.non_swiss_col: non_swiss,
            args.stance_col: stance,
            args.dept_col: dept,
            args.topic_col: topic,
            args.populism_col: populism,
        }

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run_all_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.non_swiss_col, args.stance_col, args.dept_col, args.topic_col, args.populism_col],
        # resume-safe: stance is filled for processed rows under normal parsing
        skip_if_already_filled=args.stance_col,
    )

    job_id = os.environ.get("SLURM_JOB_ID") or args.job_id or "nojobid"
    base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} and {csv_path} | Selected: {int(send_mask.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())