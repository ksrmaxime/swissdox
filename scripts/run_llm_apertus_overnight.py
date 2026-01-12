from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

import pandas as pd


def pending_count(parquet_path: Path) -> int:
    """Count remaining rows to classify (topic_llm is NA)."""
    if not parquet_path.exists():
        return -1
    df = pd.read_parquet(parquet_path, columns=["topic_llm"])
    return int(df["topic_llm"].isna().sum())


def run_once(args, *, limit: int | None) -> None:
    """Run one worker process (a fresh Python process)."""
    cmd = [
        sys.executable,
        "scripts/run_llm_apertus.py",
        "--input", args.input,
        "--output", args.output,
        "--base-url", args.base_url,
        "--workers", str(args.workers),
        "--max-chars", str(args.max_chars),
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]

    print("\n▶️", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Overnight runner: repeatedly launches run_llm_apertus.py to reset RAM.")
    ap.add_argument("--input", default="data/raw/swissdox_articles_raw.parquet")
    ap.add_argument("--output", default="data/processed/swissdox_articles_labeled_llm.parquet")
    ap.add_argument("--base-url", default="http://127.0.0.1:8080")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-chars", type=int, default=2000)

    # Overnight behavior
    ap.add_argument("--chunk", type=int, default=200, help="How many rows to process per process run (best-effort).")
    ap.add_argument("--sleep", type=int, default=20, help="Seconds to wait between runs.")
    ap.add_argument("--retries", type=int, default=3, help="How many times to retry a failed run.")
    ap.add_argument("--backoff", type=int, default=60, help="Seconds to wait before retry after a failure.")
    args = ap.parse_args()

    out_path = Path(args.output)
    in_path = Path(args.input)

    if not in_path.exists():
        raise FileNotFoundError(f"Input not found: {in_path}. Run scripts/download_articles.py first.")

    print("🌙 Overnight runner started.")
    print(f"   input : {in_path}")
    print(f"   output: {out_path}")
    print(f"   chunk : {args.chunk} | workers: {args.workers} | max_chars: {args.max_chars}")

    # Loop until no pending rows
    while True:
        pending = pending_count(out_path)
        if pending == 0:
            print("🎉 All done! No pending rows left.")
            break

        # If checkpoint not created yet, first run must be unlimited (it will create output and process some)
        first_run = (pending == -1)

        if first_run:
            print("🧩 No checkpoint yet. Running once to create it and start classification…")
            run_limit = None  # allow run_llm_apertus.py to create + start
        else:
            print(f"📌 Pending rows: {pending}")
            # Best effort: run on the whole df, but it will resume and keep going.
            # 'limit' limits the dataframe size, so we keep it None to allow resume on full,
            # BUT we still want process reset effect: we rely on checkpoint_every inside the classifier.
            # If you truly want strict chunking, set --chunk and we use it as limit only on FIRST creation.
            run_limit = None

        # Try running, retry if crash
        attempt = 0
        while True:
            attempt += 1
            try:
                # Note: limit is only useful for "test runs".
                # For real overnight stability, we keep limit=None and rely on internal checkpointing + resume.
                run_once(args, limit=run_limit if first_run else None)
                break
            except subprocess.CalledProcessError as e:
                print(f"⚠️ Run crashed (attempt {attempt}/{args.retries}): {e}")
                if attempt >= args.retries:
                    raise
                print(f"⏳ Backing off {args.backoff}s then retry…")
                time.sleep(args.backoff)

        # Re-check pending
        pending_after = pending_count(out_path)
        if pending_after == 0:
            print("🎉 Completed during last run.")
            break

        print(f"😴 Sleeping {args.sleep}s before next run… (pending now: {pending_after})")
        time.sleep(args.sleep)

    print("✅ Overnight runner finished.")


if __name__ == "__main__":
    main()
