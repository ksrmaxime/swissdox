# run_populism_pipeline.py
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

import run_populism_prompts as run_populism_prompts
from run_populism_config import build_sentences_to_send_mask

VALID_POPULISM = {"Not Populist", "Somehow Populist", "Clearly Populist"}


def parse_populism_json(raw: str) -> dict:
    empty = {"target_entity": pd.NA, "POPULISM": pd.NA, "justification": pd.NA}
    if raw is None:
        return empty

    s = str(raw).strip()

    try:
        obj = json.loads(s)
        target_entity = obj.get("target_entity")
        if target_entity is not None:
            target_entity = str(target_entity).strip()
            if target_entity.lower() in ("null", "none", ""):
                target_entity = pd.NA
        else:
            target_entity = pd.NA

        raw_pop = str(obj.get("populism", "")).strip()
        # normalise capitalisation to match VALID_POPULISM
        populism = next(
            (v for v in VALID_POPULISM if v.lower() == raw_pop.lower()),
            pd.NA,
        )

        justification = str(obj.get("justification", "")).strip() or pd.NA

        return {"target_entity": target_entity, "POPULISM": populism, "justification": justification}
    except Exception:
        pass

    # fallback regex
    entity_match = re.search(
        r'"?target_entity"?\s*:\s*"([^"\n\r,}]+)"?', s, flags=re.I
    )
    pop_match = re.search(
        r'"?populism"?\s*:\s*"(Not Populist|Somehow Populist|Clearly Populist)"?', s, flags=re.I
    )
    justification_match = re.search(
        r'"?justification"?\s*:\s*"([^"\n\r,}]+)"?', s, flags=re.I
    )

    target_entity = entity_match.group(1).strip() if entity_match else pd.NA
    raw_pop = pop_match.group(1).strip() if pop_match else ""
    populism = next(
        (v for v in VALID_POPULISM if v.lower() == raw_pop.lower()),
        pd.NA,
    )
    justification = (
        justification_match.group(1).strip() if justification_match else pd.NA
    )

    return {"target_entity": target_entity, "POPULISM": populism, "justification": justification}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True,
                    help="Output CSV/Parquet from run_critic_pipeline (must contain STANCE column).")
    ap.add_argument("--output_base", required=True)
    ap.add_argument("--job_id", default=None)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--stance_col", default="STANCE",
                    help="Column produced by run_critic_pipeline containing the stance label.")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=150)
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Limit the number of CRITIC rows processed by the LLM. "
                         "If omitted, all CRITIC rows are processed.")
    ap.add_argument("--task_id", type=int, default=None,
                    help="Array task index (0-indexed). Falls back to SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Total number of array tasks. Falls back to SLURM_ARRAY_TASK_COUNT.")

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    # ── Keep only CRITIC rows for processing (mask applied inside run_llm_dataframe) ──
    # max_rows applies to the CRITIC subset so the user can cap quickly
    if args.max_rows is not None:
        critic_idx = df.index[df[args.stance_col].astype(str).str.strip().str.upper() == "CRITIC"]
        keep_idx = critic_idx[: args.max_rows]
        non_critic_idx = df.index.difference(critic_idx)
        df = df.loc[sorted(non_critic_idx.tolist() + keep_idx.tolist())].copy()

    # ── Slicing for array jobs ────────────────────────────────────────────────
    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])

    num_tasks = args.num_tasks
    if num_tasks is None and os.environ.get("SLURM_ARRAY_TASK_COUNT"):
        num_tasks = int(os.environ["SLURM_ARRAY_TASK_COUNT"])

    if task_id is not None and num_tasks is not None:
        import math
        chunk_size = math.ceil(len(df) / num_tasks)
        start = task_id * chunk_size
        end = min(start + chunk_size, len(df))
        print(
            f"[pipeline] Array task {task_id}/{num_tasks} — rows {start}:{end} ({end - start} rows)",
            flush=True,
        )
        df = df.iloc[start:end].copy()

    send_mask = build_sentences_to_send_mask(
        df, sentence_col=args.text_col, stance_col=args.stance_col
    )
    print(f"[pipeline] CRITIC rows to process: {int(send_mask.sum()):,} / {len(df):,}", flush=True)

    for col in ["target_entity", "POPULISM", "justification"]:
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
        return run_populism_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        result = parse_populism_json(raw)
        result["raw_response"] = raw
        return result

    task_suffix = f"_task{task_id}" if task_id is not None else ""
    checkpoint_path = args.output_base + f"{task_suffix}_checkpoint.parquet"

    if Path(checkpoint_path).exists():
        print(f"[pipeline] Checkpoint found, resuming from {checkpoint_path}", flush=True)
        df = pd.read_parquet(checkpoint_path)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=run_populism_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=["target_entity", "POPULISM", "justification", "raw_response"],
        skip_if_already_filled="POPULISM",
        checkpoint_path=checkpoint_path,
        checkpoint_every=5,
    )

    job_id = (
        os.environ.get("SLURM_ARRAY_JOB_ID")
        or os.environ.get("SLURM_JOB_ID")
        or args.job_id
        or "nojobid"
    )

    if task_id is not None:
        base = f"{args.output_base}_task{task_id}_job{job_id}"
    else:
        base = f"{args.output_base}_job{job_id}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} and {csv_path} | CRITIC rows sent: {int(send_mask.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
