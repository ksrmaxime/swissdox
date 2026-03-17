# run3_pipeline.py
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

import run3_prompts as run3_prompts
from run3_config import build_sentences_to_send_mask


def parse_sp(raw: str) -> int | pd._libs.missing.NAType:
    """
    Expected output: exactly one token in {-1, 0, 1}.
    Tolerant parsing: if there is noise, extract the last valid token.
    """
    if raw is None:
        return pd.NA

    s = str(raw).strip()

    # Extract tokens -1, 0, 1 (as standalone)
    matches = re.findall(r"(?<!\d)(-1|0|1)(?!\d)", s)
    if not matches:
        return pd.NA

    tok = matches[-1]
    if tok not in ("-1", "0", "1"):
        return pd.NA
    return int(tok)


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--swiss_related_col", default="SWISS_RELATED")
    ap.add_argument("--sentiment_col", default="SENTIMENT")  # sentiment toward public administration

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=4)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    # Select only SWISS_RELATED == YES (+ non-empty sentence) via run3_config
    send_mask = build_sentences_to_send_mask(
        df,
        text_col=args.text_col,
        swiss_related_col=args.swiss_related_col,
    )

    # Ensure sentiment column exists (nullable int)
    if args.sentiment_col not in df.columns:
        df[args.sentiment_col] = pd.Series(pd.NA, index=df.index, dtype="Int64")
    else:
        # keep nullable ints if possible
        try:
            df[args.sentiment_col] = df[args.sentiment_col].astype("Int64")
        except Exception:
            df[args.sentiment_col] = pd.Series(df[args.sentiment_col], index=df.index, dtype="Int64")

    client = TransformersClient(
        LLMConfig(
            model_path=args.model_path,
            dtype=args.dtype,
            trust_remote_code=args.trust_remote_code,
            backend=args.backend,
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
        return run3_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        sp = parse_sp(raw)
        # if parsing fails, conservative fallback is 0 (as your guideline says)
        if pd.isna(sp):
            sp = 0
        return {args.sentiment_col: sp}

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run3_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.sentiment_col],
        skip_if_already_filled=args.sentiment_col,  # resume-safe
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