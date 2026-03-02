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

def _extract_json_object(raw: str) -> dict | None:
    """
    Tries to extract the first JSON object from raw text.
    Tolerant to small model mistakes (prefix/suffix text), but expects an object.
    """
    if raw is None:
        return None
    s = str(raw).strip()
    if not s:
        return None

    # Fast path
    if s.startswith("{") and s.endswith("}"):
        try:
            return json.loads(s)
        except Exception:
            pass

    # Tolerant path: find first {...} block
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
    # tolerant: search any allowed topic phrase
    for t in sorted(_ALLOWED_TOPICS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", s):
            return t
    return pd.NA


def _norm_int_101(x) -> int | pd._libs.missing.NAType:
    if x is None:
        return pd.NA
    s = str(x).strip()
    # accept int-like, or noisy text containing -1/0/1
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

    # output columns
    ap.add_argument("--swiss_col", default="SWISS_RELATED")
    ap.add_argument("--topic_col", default="TOPIC")
    ap.add_argument("--sentiment_col", default="SENTIMENT")
    ap.add_argument("--populism_col", default="POPULISM")

    # behavior knobs
    ap.add_argument(
        "--only_fill_if_swiss_yes",
        action="store_true",
        help="If set: when swiss=NO, set topic/sentiment/populism to NA (matches your 4-run logic).",
    )

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=96)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if args.text_col not in df.columns:
        raise KeyError(f"Missing text column: {args.text_col}")

    # send all non-empty sentences
    send_mask = df[args.text_col].notna() & (df[args.text_col].astype(str).str.strip() != "")

    # Ensure output cols exist
    for c in (args.swiss_col, args.topic_col):
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            df[c] = df[c].astype("string")

    for c in (args.sentiment_col, args.populism_col):
        if c not in df.columns:
            df[c] = pd.Series(pd.NA, index=df.index, dtype="Int64")
        else:
            try:
                df[c] = df[c].astype("Int64")
            except Exception:
                df[c] = pd.Series(df[c], index=df.index, dtype="Int64")

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
                args.swiss_col: pd.NA,
                args.topic_col: "Other",
                args.sentiment_col: 0,
                args.populism_col: 0,
            }

        swiss = _norm_yes_no(obj.get("swiss"))
        topic = _norm_topic(obj.get("topic"))
        sp = _norm_int_101(obj.get("sp"))
        p = _norm_int_101(obj.get("p"))

        # defaults on parse failure
        if pd.isna(topic):
            topic = "Other"
        if pd.isna(sp):
            sp = 0
        if pd.isna(p):
            p = 0

        # optional: match your 4-run behavior (2-4 only if swiss YES)
        if args.only_fill_if_swiss_yes and swiss == "NO":
            topic = pd.NA
            sp = pd.NA
            p = pd.NA

        return {
            args.swiss_col: swiss,
            args.topic_col: topic,
            args.sentiment_col: sp,
            args.populism_col: p,
        }

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run_all_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.swiss_col, args.topic_col, args.sentiment_col, args.populism_col],
        # resume-safe: skip only if ALL 4 already filled (simple approach: pick one “last” col)
        skip_if_already_filled=args.populism_col,
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