# 04_classify_stance.py
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

import stance_prompts as stance_prompts
from stance_config import build_sentences_to_send_mask

VALID_STANCES = {"CRITIC", "PRAISE", "NEUTRAL_STATEMENT"}


def parse_critic_json(raw: str) -> dict:
    if raw is None:
        return {"STANCE": pd.NA, "justification": pd.NA}

    s = str(raw).strip()

    try:
        obj = json.loads(s)
        stance = str(obj.get("stance", "")).strip().upper()
        justification = str(obj.get("justification", "")).strip()

        if stance not in VALID_STANCES:
            stance = pd.NA
        if justification == "":
            justification = pd.NA

        return {"STANCE": stance, "justification": justification}
    except Exception:
        pass

    stance_match = re.search(
        r'"?stance"?\s*:\s*"?(CRITIC|PRAISE|NEUTRAL_STATEMENT)"?', s, flags=re.I
    )
    justification_match = re.search(
        r'"?justification"?\s*:\s*"([^"\n\r,}]+)"?', s, flags=re.I
    )

    stance = stance_match.group(1).upper() if stance_match else pd.NA
    justification = (
        justification_match.group(1).strip() if justification_match else pd.NA
    )

    if isinstance(stance, str) and stance not in VALID_STANCES:
        stance = pd.NA
    if isinstance(justification, str) and justification == "":
        justification = pd.NA

    return {"STANCE": stance, "justification": justification}


def main() -> int:
    ap = argparse.ArgumentParser()

    ap.add_argument("--input", required=True)
    ap.add_argument("--output_base", required=True)

    ap.add_argument("--model_path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers"])
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--decision_col", default="STANCE")

    ap.add_argument("--batch_size", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max_new_tokens", type=int, default=80)
    ap.add_argument("--max_rows", type=int, default=None,
                    help="Limite le nombre de lignes traitées par le LLM (ex: 300). "
                         "Si omis, toutes les lignes sont traitées.")
    ap.add_argument("--task_id", type=int, default=None,
                    help="Index de la tâche array (0-indexed). Si omis, utilise SLURM_ARRAY_TASK_ID.")
    ap.add_argument("--num_tasks", type=int, default=None,
                    help="Nombre total de tâches array. Si omis, traite tout le dataset.")

    args = ap.parse_args()

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input)

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    # ── Slicing pour job array ────────────────────────────────────────────────
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
        print(f"[pipeline] Array task {task_id}/{num_tasks} — lignes {start}:{end} ({end - start} lignes)", flush=True)
        df = df.iloc[start:end].copy()

    send_mask = build_sentences_to_send_mask(df, sentence_col=args.text_col)

    for col in ["STANCE", "justification"]:
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
        return stance_prompts.build_user_prompt(row, text_col=text_col)

    def _parse(raw: str) -> dict:
        result = parse_critic_json(raw)
        result["raw_response"] = raw
        return result

    task_suffix = f"_task{task_id}" if task_id is not None else ""
    checkpoint_path = args.output_base + f"{task_suffix}_checkpoint.parquet"

    # reprise depuis checkpoint si disponible
    if Path(checkpoint_path).exists():
        print(f"[pipeline] Checkpoint trouvé, reprise depuis {checkpoint_path}", flush=True)
        df = pd.read_parquet(checkpoint_path)

    out = run_llm_dataframe(
        df=df,
        cfg=run_cfg,
        client=client,
        system_prompt=stance_prompts.SYSTEM_PROMPT,
        select_mask_fn=_select_mask,
        build_prompt_fn=_build_prompt,
        parse_fn=_parse,
        output_cols=["STANCE", "justification", "raw_response"],
        skip_if_already_filled="STANCE",
        checkpoint_path=checkpoint_path,
        checkpoint_every=5,
    )

    base = f"{args.output_base}{task_suffix}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"

    Path(parquet_path).parent.mkdir(parents=True, exist_ok=True)

    out.to_parquet(parquet_path, index=False)
    out.to_csv(csv_path, index=False)

    print(f"Saved: {parquet_path} and {csv_path} | Selected: {int(send_mask.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
