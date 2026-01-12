from __future__ import annotations

from pathlib import Path
import argparse
import pandas as pd

from swissdox.llm_client import LlamaServerClient
from swissdox.pipelines.llm_apertus import LlmRunConfig, classify_dataframe_checkpointed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/raw/swissdox_articles_raw.parquet")
    ap.add_argument("--output", default="data/processed/swissdox_articles_labeled_llm.parquet")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--max-chars", type=int, default=2000)
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run scripts/download_articles.py first.")

    df = pd.read_parquet(in_path)

    client = LlamaServerClient(base_url=args.base_url)
    model_id = client.get_model_id()
    print(f"✅ llama-server reachable, using model='{model_id}'")

    cfg = LlmRunConfig(
        n_workers=args.workers,
        max_chars_input=args.max_chars,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    df_out = classify_dataframe_checkpointed(
        df,
        client=client,
        model_id=model_id,
        output_parquet=out_path,
        cfg=cfg,
        limit=args.limit,
    )

    print("✅ Done:", df_out.shape)
    print(df_out[["pubtime","language","medium_name","head","topic_llm","sentiment_public_admin_llm","populism_llm"]].head(10))


if __name__ == "__main__":
    main()
