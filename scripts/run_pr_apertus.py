from __future__ import annotations
import argparse
from pathlib import Path
from pr_run import Cfg, run

ap = argparse.ArgumentParser()
ap.add_argument("--workdir", required=True)
ap.add_argument("--scratchdir", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--outdir", required=True)
ap.add_argument("--model-path", required=True)
ap.add_argument("--dtype", default="bf16", choices=["bf16","fp16"])
ap.add_argument("--items-per-prompt", type=int, default=60)
ap.add_argument("--prompts-per-batch", type=int, default=8)
ap.add_argument("--max-tokens", type=int, default=250)
ap.add_argument("--temperature", type=float, default=0.0)
ap.add_argument("--resume", action="store_true")
args = ap.parse_args()

cfg = Cfg(
    inp=Path(args.workdir) / args.input,
    outdir=Path(args.outdir),
    scratch=Path(args.scratchdir) / "swissdox/pr",
    model_path=args.model_path,
    dtype=args.dtype,
    items_per_prompt=args.items_per_prompt,
    prompts_per_batch=args.prompts_per_batch,
    max_tokens=args.max_tokens,
    temperature=args.temperature,
    resume=args.resume,
)
run(cfg)
