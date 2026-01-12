# scripts/download_articles.py
from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from swissdox.api import SwissdoxClient
from swissdox.config import DEFAULT_API_BASE_URL, DEFAULT_COLUMNS, build_query_payload, default_query_name
from swissdox.text import clean_text, clean_xml_swissdox


def clean_articles_df(df_raw: pd.DataFrame) -> pd.DataFrame:
    df = df_raw.copy()

    if "pubtime" in df.columns:
        df["pubtime"] = pd.to_datetime(df["pubtime"].astype(str), errors="coerce", utc=True).dt.date

    for c in ["medium_name", "rubric", "dateline", "head", "subhead"]:
        if c in df.columns:
            df[c] = df[c].apply(clean_text)

    if "content" in df.columns:
        df["content"] = df["content"].apply(clean_xml_swissdox).apply(clean_text)

    return df


def main() -> None:
    load_dotenv()
    api_key = os.getenv("SWISSDOX_API_KEY")
    api_secret = os.getenv("SWISSDOX_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("Missing SWISSDOX_API_KEY / SWISSDOX_API_SECRET in .env")

    # ---- Your defaults (can later be CLI args) ----
    start_date = "2025-01-01"
    end_date = "2025-12-31"
    languages = ["de", "fr"]  # website language (keep single if you want avoid duplicates)
    sources = ["NZZO","NNTA","NNHEU","ZWSO","TPS","NZZ","TA","ZWAO","TPSO","HEU","ZWAS","NZZS","ZWAI"]
    max_results = 20000

    query_name = default_query_name("BuerokratieVerwaltung_2025")
    comment = "Query generated from scripts/download_articles.py"
    expiration_date = "2026-01-30"

    payload = build_query_payload(
        start_date=start_date,
        end_date=end_date,
        languages=languages,
        sources=sources,
        max_results=max_results,
        columns=DEFAULT_COLUMNS,
        query_name=query_name,
        comment=comment,
        expiration_date=expiration_date,
    )

    client = SwissdoxClient(api_key=api_key, api_secret=api_secret, base_url=DEFAULT_API_BASE_URL)

    query_id = client.submit_query(
        payload["yaml_payload"],
        name=payload["meta"]["name"],
        comment=payload["meta"]["comment"],
        expiration_date=payload["meta"]["expirationDate"],
        test=False,
    )
    print(f"✅ queryId={query_id}")

    download_url = client.wait_for_download_url(query_id)
    print(f"⬇️ Download: {download_url}")

    df_raw = client.download_tsv_xz(download_url)
    print("✅ df_raw:", df_raw.shape)

    df_articles = clean_articles_df(df_raw)
    print("✅ df_articles:", df_articles.shape)

    out_dir = Path("data/raw")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer parquet for speed/size
    out_path = out_dir / "swissdox_articles_raw.parquet"
    df_articles.to_parquet(out_path, index=False)
    print(f"💾 Saved: {out_path}")


if __name__ == "__main__":
    main()
