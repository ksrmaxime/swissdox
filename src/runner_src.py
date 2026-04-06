# src/runner_src.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional
import pandas as pd


@dataclass(frozen=True)
class RunConfig:
    id_col: str
    text_col: str
    batch_size: int = 32
    temperature: float = 0.0
    max_new_tokens: int = 200


def ensure_columns(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c not in df.columns:
            df[c] = pd.NA
    return df


def run_llm_dataframe(
    df: pd.DataFrame,
    cfg: RunConfig,
    client,
    system_prompt: str,
    select_mask_fn: Callable[[pd.DataFrame], pd.Series],
    build_prompt_fn: Callable[[pd.Series, str], str],
    parse_fn: Callable[[str], Dict[str, object]],
    output_cols: List[str],
    skip_if_already_filled: Optional[str] = None,
    checkpoint_path: Optional[str] = None,
    checkpoint_every: int = 5,
) -> pd.DataFrame:
    """
    - select_mask_fn(df) -> bool mask des lignes à traiter
    - build_prompt_fn(row, text_col) -> string prompt user
    - parse_fn(raw_completion) -> dict {col: value, ...}
    - output_cols = colonnes qu'on garantit dans df
    - skip_if_already_filled: si défini, on saute les lignes où df[col] n'est pas NA
    - checkpoint_path: si défini, sauvegarde le df toutes les checkpoint_every batchs
    - checkpoint_every: nombre de batchs entre chaque sauvegarde (défaut 5)
    """
    df = df.copy()
    df = ensure_columns(df, output_cols)

    mask = select_mask_fn(df)
    if mask is None:
        mask = pd.Series([True] * len(df), index=df.index)

    todo = mask.copy()
    if skip_if_already_filled:
        todo = todo & df[skip_if_already_filled].isna()

    idx = df.index[todo].tolist()
    if not idx:
        return df

    total = len(idx)
    print(f"[runner] {total} lignes à traiter, batch_size={cfg.batch_size}, checkpoint_every={checkpoint_every} batchs", flush=True)

    # batching
    for batch_num, start in enumerate(range(0, total, int(cfg.batch_size))):
        batch_idx = idx[start:start + int(cfg.batch_size)]
        batch_rows = df.loc[batch_idx]

        user_prompts = [build_prompt_fn(row, cfg.text_col) for _, row in batch_rows.iterrows()]
        raw = client.chat_many(
            system_prompt=system_prompt,
            user_prompts=user_prompts,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
        )

        # write back
        for i, row_id in enumerate(batch_idx):
            parsed = parse_fn(raw[i])
            for k, v in parsed.items():
                if k in df.columns:
                    df.at[row_id, k] = v

        done = min(start + int(cfg.batch_size), total)
        print(f"[runner] batch {batch_num + 1} — {done}/{total} lignes traitées", flush=True)

        # checkpoint
        if checkpoint_path and (batch_num + 1) % checkpoint_every == 0:
            Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
            df.to_parquet(checkpoint_path, index=False)
            print(f"[runner] checkpoint sauvegardé → {checkpoint_path}", flush=True)

    # checkpoint final
    if checkpoint_path:
        Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(checkpoint_path, index=False)
        print(f"[runner] checkpoint final sauvegardé → {checkpoint_path}", flush=True)

    return df
