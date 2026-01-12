from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple, Optional

import numpy as np
import pandas as pd

# sentence-transformers importe torch en interne; on ne dépend pas directement de torch ici
from sentence_transformers import SentenceTransformer


Theme = Tuple[str, str]  # (label, description)


@dataclass(frozen=True)
class EmbeddingThemeConfig:
    model_name: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    batch_size: int = 64
    threshold: float = 0.25
    other_label: str = "Others"
    max_content_chars_fallback: int = 1200


def build_text_for_theme(
    df: pd.DataFrame,
    *,
    head_col: str = "head",
    subhead_col: str = "subhead",
    content_col: str = "content",
    out_col: str = "text_for_theme",
    max_content_chars_fallback: int = 1200,
) -> pd.DataFrame:
    """Create a text field used for theme classification.

    Strategy:
    - head + subhead
    - if empty, fallback to first N chars of content
    """
    df = df.copy()

    head = df[head_col].fillna("").astype(str).str.strip() if head_col in df.columns else ""
    sub = df[subhead_col].fillna("").astype(str).str.strip() if subhead_col in df.columns else ""

    df[out_col] = (head + " — " + sub).astype(str).str.strip().str.strip(" —")

    if content_col in df.columns:
        mask_empty = df[out_col].eq("")
        df.loc[mask_empty, out_col] = (
            df.loc[mask_empty, content_col]
            .fillna("")
            .astype(str)
            .str.slice(0, max_content_chars_fallback)
        )

    return df


def _auto_device() -> str:
    """Best-effort device selection (macOS MPS if available, else CPU)."""
    try:
        import torch

        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def assign_themes_by_embeddings(
    df_articles: pd.DataFrame,
    themes: Sequence[Theme],
    *,
    text_col: str = "text_for_theme",
    out_theme_col: str = "main_theme",
    out_score_col: str = "theme_score",
    cfg: EmbeddingThemeConfig = EmbeddingThemeConfig(),
    device: Optional[str] = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Assign a main theme using multilingual sentence embeddings + cosine similarity.

    - Embeddings are normalized, so dot product == cosine similarity.
    - If best score < threshold -> other_label.
    """
    if not themes:
        raise ValueError("themes must be a non-empty list of (label, description).")
    if text_col not in df_articles.columns:
        raise ValueError(f"Missing column '{text_col}' in df_articles. Run build_text_for_theme first.")

    labels: List[str] = [t[0] for t in themes]
    theme_texts: List[str] = [f"{t[0]}: {t[1]}" for t in themes]

    device = device or _auto_device()
    model = SentenceTransformer(cfg.model_name, device=device)

    theme_emb = model.encode(theme_texts, normalize_embeddings=True, show_progress_bar=False)
    texts = df_articles[text_col].fillna("").astype(str).tolist()

    emb_articles = model.encode(
        texts,
        batch_size=cfg.batch_size,
        normalize_embeddings=True,
        show_progress_bar=show_progress,
    )

    scores = emb_articles @ theme_emb.T  # (n_articles, n_themes)
    best_idx = scores.argmax(axis=1)
    best_score = scores.max(axis=1).astype(float)

    df_out = df_articles.copy()
    df_out[out_theme_col] = [labels[i] for i in best_idx]
    df_out[out_score_col] = best_score

    df_out.loc[df_out[out_score_col] < cfg.threshold, out_theme_col] = cfg.other_label
    return df_out
