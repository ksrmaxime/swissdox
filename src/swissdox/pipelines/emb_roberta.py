from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Optional
import pandas as pd

from swissdox.embeddings import EmbeddingThemeConfig, build_text_for_theme, assign_themes_by_embeddings
from swissdox.sentences import SentenceSplitConfig, split_and_filter_sentences
from swissdox.sentiment import SentimentConfig, add_sentiment_columns

Theme = Tuple[str, str]


@dataclass(frozen=True)
class EmbRobertaPipelineConfig:
    themes_cfg: EmbeddingThemeConfig = EmbeddingThemeConfig()
    split_cfg: SentenceSplitConfig = SentenceSplitConfig()
    sentiment_cfg: SentimentConfig = SentimentConfig()


def run_emb_roberta_pipeline(
    df_articles: pd.DataFrame,
    *,
    themes: Sequence[Theme],
    keywords: Sequence[str],
    cfg: EmbRobertaPipelineConfig = EmbRobertaPipelineConfig(),
    hf_token: Optional[str] = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Runs the full Embedded + RoBERTa pipeline in-memory.

    Returns:
      df_articles_themed,
      df_sentences_filtered,
      df_sentences_with_sentiment
    """
    # 1) themes
    df1 = build_text_for_theme(df_articles, max_content_chars_fallback=cfg.themes_cfg.max_content_chars_fallback)
    df_articles_themed = assign_themes_by_embeddings(df1, themes, cfg=cfg.themes_cfg, show_progress=True)

    # 2) sentence split + keyword filter
    keep_cols = [
        "id",
        "pubtime",
        "medium_name",
        "language",
        "main_theme",
        "theme_score",
    ]
    df_sentences = split_and_filter_sentences(
        df_articles_themed,
        keywords,
        keep_cols=keep_cols,
        cfg=cfg.split_cfg,
    )

    # 3) sentence sentiment
    df_final = add_sentiment_columns(df_sentences, cfg=cfg.sentiment_cfg, hf_token=hf_token)

    return df_articles_themed, df_sentences, df_final
