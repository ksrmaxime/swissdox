# step3a_stance_array.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))
sys.path.append(str(ROOT / "src"))   # stance_prompts / stance_config (sans dépendre de pip install -e)

import os
import argparse
import json
import re
import pandas as pd

from src.client_src import TransformersClient, LLMConfig
from src.runner_src import run_llm_dataframe, RunConfig, save_atomic

import stance_prompts as stance_prompts
from stance_config import build_sentences_to_send_mask

VALID_STANCES = {"CRITICISM", "PRAISE", "NEUTRAL_STATEMENT"}


def parse_criticism_json(raw: str) -> dict:
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
        r'"?stance"?\s*:\s*"?(CRITICISM|PRAISE|NEUTRAL_STATEMENT)"?', s, flags=re.I
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
    ap.add_argument("--backend", default="transformers", choices=["vllm", "transformers", "dummy"],
                    help="dummy = tests sans GPU (réponses factices)")
    ap.add_argument("--trust_remote_code", action="store_true")

    ap.add_argument("--text_col", default="sentence")
    ap.add_argument("--keyword_col", default="matched_keywords",
                    help="Colonne contenant l'entité/keyword ciblé par la ligne (voir step2_sentence_cutting.py).")
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
                    help="Nombre total de tâches array (doit rester le même pour une reprise). "
                         "Si omis, traite tout le dataset.")
    ap.add_argument("--checkpoint_minutes", type=float, default=10.0,
                    help="Sauvegarde intermédiaire toutes les N minutes (défaut 10)")

    args = ap.parse_args()

    # ── Tâche array et fichiers de cette tâche ────────────────────────────────
    task_id = args.task_id
    if task_id is None and os.environ.get("SLURM_ARRAY_TASK_ID"):
        task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    num_tasks = args.num_tasks   # volontairement PAS SLURM_ARRAY_TASK_COUNT : une tâche
                                 # relancée seule (--array=3) doit garder le même découpage
    if task_id is not None and num_tasks is not None and not (0 <= task_id < num_tasks):
        raise ValueError(f"task_id={task_id} hors de 0..{num_tasks - 1} (NUM_TASKS={num_tasks})")

    task_suffix = f"_task{task_id}" if task_id is not None else ""
    base = f"{args.output_base}{task_suffix}"
    parquet_path = base + ".parquet"
    csv_path = base + ".csv"
    checkpoint_path = base + "_checkpoint.parquet"

    if Path(parquet_path).exists():
        print(f"[pipeline] Tâche déjà terminée ({parquet_path} existe) : rien à faire.", flush=True)
        return 0

    df = pd.read_parquet(args.input) if args.input.endswith(".parquet") else pd.read_csv(args.input, low_memory=False)

    if args.keyword_col not in df.columns:
        raise KeyError(f"Missing keyword column: {args.keyword_col}")

    if args.max_rows is not None:
        df = df.iloc[: args.max_rows].copy()

    # ── Slicing pour job array ────────────────────────────────────────────────
    if task_id is not None and num_tasks is not None:
        import math
        chunk_size = math.ceil(len(df) / num_tasks)
        start = task_id * chunk_size
        end = min(start + chunk_size, len(df))
        print(f"[pipeline] Array task {task_id}/{num_tasks} — lignes {start}:{end} ({end - start} lignes)", flush=True)
        df = df.iloc[start:end].copy()
    # Index 0..n-1, identique à celui d'un checkpoint relu (sauvé sans index).
    df = df.reset_index(drop=True)

    # ── Reprise depuis le checkpoint de CETTE tâche ───────────────────────────
    if Path(checkpoint_path).exists():
        ckpt = pd.read_parquet(checkpoint_path)
        same = (len(ckpt) == len(df) and "sentence_id" in ckpt.columns
                and ckpt["sentence_id"].astype(str).tolist() == df["sentence_id"].astype(str).tolist())
        if not same:
            raise RuntimeError(
                f"Le checkpoint {checkpoint_path} ne correspond pas aux lignes de cette tâche "
                f"({len(ckpt)} lignes vs {len(df)}). Mêmes INPUT / NUM_TASKS / MAX_ROWS que le run "
                f"d'origine ? Sinon, supprimer ce checkpoint."
            )
        n_done = int(ckpt["STANCE"].notna().sum()) if "STANCE" in ckpt.columns else 0
        print(f"[pipeline] Checkpoint trouvé : reprise ({n_done:,} lignes déjà classées sur {len(ckpt):,})", flush=True)
        df = ckpt

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
        return stance_prompts.build_user_prompt(row, text_col=text_col, keyword_col=args.keyword_col)

    def _parse(raw: str) -> dict:
        result = parse_criticism_json(raw)
        result["raw_response"] = raw
        return result

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
        checkpoint_every_s=args.checkpoint_minutes * 60,
    )

    # Écriture atomique : le .parquet final n'existe que s'il est complet
    # (c'est lui qui marque la tâche comme terminée pour une relance).
    out.to_csv(csv_path + ".partial", index=False)
    os.replace(csv_path + ".partial", csv_path)
    save_atomic(out, parquet_path)

    # Le run est terminé et sauvegardé dans parquet_path/csv_path : le checkpoint
    # est supprimé pour qu'il ne soit plus compté par le glob "_task*.parquet"
    # du script de merge (sinon chaque ligne y est comptée deux fois).
    if Path(checkpoint_path).exists():
        Path(checkpoint_path).unlink()

    print(f"Saved: {parquet_path} and {csv_path} | Selected: {int(send_mask.sum()):,}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
