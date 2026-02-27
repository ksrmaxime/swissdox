# run4_pipeline.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import os
import argparse
import re
import pandas as pd

from src.client_src import TransformersClient, LLMConfig
from src.runner_src import run_llm_dataframe, RunConfig

import run4_prompts as run4_prompts
from run4_config import build_sentences_to_send_mask


_ALLOWED = set(run4_prompts.TOPICS)

def parse_topic(raw: str) -> str | pd._libs.missing.NAType:
    """
    Expected: one of the allowed topics, as a single string.
    Tolerant parsing:
      - trims quotes/spaces
      - if extra text: tries to extract any valid topic token/phrase
    """
    if raw is None:
        return pd.NA

    s = str(raw).strip()

    # common: quoted output
    s2 = s.strip().strip('"').strip("'").strip()

    if s2 in _ALLOWED:
        return s2

    # If noisy: find any allowed topic as a whole phrase
    # (longest first to avoid partial matches)
    for t in sorted(_ALLOWED, key=len, reverse=True):
        if re.search(rf"\b{re.escape(t)}\b", s2):
            return t

    return pd.NA


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--swiss_related_col", default="SWISS_RELATED")
    ap.add_argument("--topic_col", default="SENTIMENT")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=12)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    send_mask = build_sentences_to_send_mask(
        df,
        text_col=args.text_col,
        swiss_related_col=args.swiss_related_col,
    )

    # Ensure topic column exists
    if args.topic_col not in df.columns:
        df[args.topic_col] = pd.Series(pd.NA, index=df.index, dtype="string")
    else:
        df[args.topic_col] = df[args.topic_col].astype("string")

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
        return run4_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        topic = parse_topic(raw)
        # hard fallback to "Other" if parse fails (optional but usually desirable)
        if pd.isna(topic):
            topic = "Other"
        return {args.topic_col: topic}

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run4_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.topic_col],
        skip_if_already_filled=args.topic_col,  # resume
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