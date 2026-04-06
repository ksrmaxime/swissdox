# run_article_pipeline.py
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

import run_article_prompts as run_article_prompts
from run_article_config import build_sentences_to_send_mask


def parse_article_json(raw: str) -> dict:
    if raw is None:
        return {"swiss": pd.NA, "justification": pd.NA}

    s = str(raw).strip()

    try:
        obj = json.loads(s)
        swiss = str(obj.get("swiss", "")).strip().upper()
        justification = str(obj.get("justification", "")).strip().lower()

        if swiss not in {"YES", "NO"}:
            swiss = pd.NA
        if justification == "":
            justification = pd.NA

        return {"swiss": swiss, "justification": justification}
    except Exception:
        pass

    swiss_match = re.search(r'"?swiss"?\s*:\s*"?(YES|NO)"?', s, flags=re.I)
    justification_match = re.search(r'"?justification"?\s*:\s*"([^"\n\r,}]+)"?', s, flags=re.I)

    swiss = swiss_match.group(1).upper() if swiss_match else pd.NA
    justification = justification_match.group(1).strip().lower() if justification_match else pd.NA

    if isinstance(justification, str) and justification == "":
        justification = pd.NA

    return {"swiss": swiss, "justification": justification}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="lead")
    ap.add_argument("--decision_col", default="swiss")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=80)

    args = ap.parse_args()

    checkpoint_path = args.output_base + "_checkpoint.parquet"

    if Path(checkpoint_path).exists():
        print(f"[pipeline] Checkpoint trouvé, reprise depuis {checkpoint_path}", flush=True)
        df = pd.read_parquet(checkpoint_path)
    else:
        df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    send_mask = build_sentences_to_send_mask(df, title_col="title", lead_col="lead")

    for col in ["swiss", "justification"]:
        if col not in df.columns:
            df[col] = pd.Series(pd.NA, index=df.index, dtype="string")
        else:
            df[col] = df[col].astype("string")

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
        return run_article_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        return parse_article_json(raw)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run_article_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=["swiss", "justification"],
        skip_if_already_filled="justification",
        checkpoint_path=checkpoint_path,
        checkpoint_every=5,
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