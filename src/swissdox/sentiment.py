from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pandas as pd


@dataclass(frozen=True)
class SentimentConfig:
    model_name: str = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
    batch_size: int = 64
    max_length: int = 256
    text_col: str = "sentence"
    out_label_col: str = "sentiment_label"
    out_score_col: str = "sentiment_score"
    # common mapping for this model family
    label_map: dict = None


def _auto_device_for_pipeline():
    """Return device value compatible with transformers pipeline: 0 (cuda), 'mps', or -1 (cpu)."""
    try:
        import torch

        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return -1


def sentiment_predict(
    texts: List[str],
    *,
    model_name: str,
    hf_token: Optional[str] = None,
    batch_size: int = 64,
    max_length: int = 256,
) -> Tuple[List[str], List[float]]:
    """Run HF sentiment pipeline on a list of texts."""
    import torch  # local import keeps module import light
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline

    tokenizer_kwargs = {"token": hf_token} if hf_token else {}
    model_kwargs = {"token": hf_token} if hf_token else {}

    tokenizer = AutoTokenizer.from_pretrained(model_name, **tokenizer_kwargs)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, **model_kwargs)

    device = _auto_device_for_pipeline()
    sent_pipe = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, device=device)

    labels: List[str] = []
    scores: List[float] = []

    for i in range(0, len(texts), batch_size):
        preds = sent_pipe(texts[i : i + batch_size], truncation=True, max_length=max_length)
        for p in preds:
            labels.append(str(p["label"]))
            scores.append(float(p["score"]))

    return labels, scores


def add_sentiment_columns(
    df: pd.DataFrame,
    *,
    cfg: SentimentConfig = SentimentConfig(),
    hf_token: Optional[str] = None,
) -> pd.DataFrame:
    """Return df with sentiment_label + sentiment_score appended."""
    if cfg.text_col not in df.columns:
        raise ValueError(f"Missing column '{cfg.text_col}' in df")

    texts = df[cfg.text_col].fillna("").astype(str).tolist()
    labels, scores = sentiment_predict(
        texts,
        model_name=cfg.model_name,
        hf_token=hf_token,
        batch_size=cfg.batch_size,
        max_length=cfg.max_length,
    )

    df_out = df.copy()
    df_out[cfg.out_label_col] = labels
    df_out[cfg.out_score_col] = scores

    label_map = cfg.label_map or {"LABEL_0": "negative", "LABEL_1": "neutral", "LABEL_2": "positive"}
    df_out[cfg.out_label_col] = df_out[cfg.out_label_col].map(lambda x: label_map.get(x, x))

    return df_out
