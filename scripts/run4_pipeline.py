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


def parse_p(raw: str) -> int | pd._libs.missing.NAType:
    """
    Expected output: exactly one token in {-1, 0, 1}.
    Tolerant parsing: if there is noise, extract the last valid token.
    """
    if raw is None:
        return pd.NA

    s = str(raw).strip()

    # Extract standalone tokens -1, 0, 1 (avoid capturing digits inside other numbers)
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
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--swiss_related_col", default="SWISS_RELATED")
    ap.add_argument("--populism_col", default="POPULISM")  # p (populism)

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=4)

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    # Selection: usually only SWISS_RELATED == YES (+ non-empty sentence)
    send_mask = build_sentences_to_send_mask(
        df,
        text_col=args.text_col,
        swiss_related_col=args.swiss_related_col,
    )

    # Ensure output column exists (nullable int)
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
        return run4_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        p = parse_p(raw)
        # If parsing fails, safest fallback is 0 ("somehow populist") rather than -1
        if pd.isna(p):
            p = 0
        return {args.populism_col: p}

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run4_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=[args.populism_col],
        skip_if_already_filled=args.populism_col,  # resume-safe
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