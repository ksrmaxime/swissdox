from __future__ import annotations

import argparse
from pathlib import Path

from pr_classifier import PRConfig, build_apertus_client, classify_pr


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--workdir", required=True)
    ap.add_argument("--scratchdir", required=True)

    ap.add_argument("--input", required=True)
    ap.add_argument("--outdir", required=True)

    ap.add_argument("--model-path", required=True)
    ap.add_argument("--dtype", default="bf16", choices=["bf16", "fp16", "auto"])

    ap.add_argument("--batch-size", type=int, default=60)
    ap.add_argument("--max-tokens", type=int, default=250)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--resume", action="store_true")

    args = ap.parse_args()

    workdir = Path(args.workdir)
    scratchdir = Path(args.scratchdir) / "swissdox/pr"
    outdir = Path(args.outdir)

    llm = build_apertus_client(model_path=args.model_path, dtype=args.dtype)

    cfg = PRConfig(
        in_parquet=workdir / args.input,
        out_dir=outdir,
        scratch_dir=scratchdir,
        batch_size=int(args.batch_size),
        max_tokens=int(args.max_tokens),
        temperature=float(args.temperature),
        resume=bool(args.resume),
    )

    out = classify_pr(cfg, llm)

    print("[DONE] Outputs:")
    for k, p in out.items():
        print(f"  - {k}: {p}")


if __name__ == "__main__":
    main()
