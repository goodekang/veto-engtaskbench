#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veto.backbone import CachedCompletionEncoder
from veto.compute import pick_device
from veto.config import load_config, repo_root
from veto.engine import build_loader, describe_params, eval_official, load_policy, summarize_eval


def setup_log() -> logging.Logger:
    log = logging.getLogger("eval")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s INFO %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pace-sec", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--stretch", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config()
    root = repo_root()
    log = setup_log()
    ckpt_path = Path(args.ckpt) if args.ckpt else root / "checkpoints" / cfg["main_run"] / "best.pt"
    pace = args.pace_sec if args.pace_sec is not None else float(cfg.get("pace_sec", 1.0))
    batch_size = args.batch_size or int(cfg.get("batch_size", 4))
    device = pick_device(args.device)

    log.info("device=%s", device)
    probe = CachedCompletionEncoder().probe()
    log.info("llm_cache available=%s download=False", probe["available"])
    model, blob = load_policy(ckpt_path, device)
    log.info("loaded %s epoch=%s best_val=%.4f", ckpt_path, blob.get("epoch"), blob.get("best_val", 0))
    log.info(describe_params(model, blob))

    _, loader = build_loader(root, split=args.split, batch_size=batch_size, train=False)
    log.info("n_tasks=%d batches=%d", len(loader.dataset), len(loader))

    t0 = time.time()
    df = eval_official(
        model,
        loader,
        device,
        extra=True,
        n_tta=args.tta,
        stretch=args.stretch,
        pace_sec=pace,
        log=log,
    )
    summary = summarize_eval(df)
    out_path = Path(args.out) if args.out else root / "results" / "predictions" / "eval_replay.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    log.info(
        "done  seconds=%.1f  s/task=%.3f  tsr=%.3f  ci95=[%.3f, %.3f]  ece=%.3f  out=%s",
        elapsed,
        elapsed / max(len(df), 1),
        summary["tsr"],
        summary["ci_lo"],
        summary["ci_hi"],
        summary.get("ece", 0.0),
        out_path,
    )


if __name__ == "__main__":
    main()
