#!/usr/bin/env python
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veto.compute import official_forward, pick_device, wait_io_floor
from veto.config import load_config, repo_root
from veto.engine import build_loader, describe_params, load_policy


def setup_log() -> logging.Logger:
    log = logging.getLogger("robust")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s INFO %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


@torch.no_grad()
def eval_dropout(model, loader, device, drop: float, pace_sec: float, log) -> float:
    model.eval()
    hits = 0
    n = 0
    t0 = time.time()
    for i, batch in enumerate(loader, start=1):
        bt = time.time()
        obs = batch["obs"].to(device)
        lengths = batch["lengths"]
        if drop > 0:
            mask = (torch.rand_like(obs) > drop).to(obs.dtype)
            obs = obs * mask
        out = official_forward(model, obs, lengths)
        pred = (torch.sigmoid(out["logit"]) >= 0.5).float().cpu()
        hits += int((pred == batch["y"]).sum())
        n += len(batch["y"])
        wait_io_floor(bt, pace_sec)
        log.info("drop=%.2f batch %d/%d tsr=%.3f", drop, i, len(loader), hits / max(n, 1))
    log.info("drop=%.2f done seconds=%.1f tsr=%.3f", drop, time.time() - t0, hits / max(n, 1))
    return hits / max(n, 1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--pace-sec", type=float, default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = repo_root()
    log = setup_log()
    device = pick_device(args.device)
    pace = args.pace_sec if args.pace_sec is not None else float(cfg.get("pace_sec", 1.0))
    ckpt = Path(args.ckpt) if args.ckpt else root / "checkpoints" / cfg["main_run"] / "best.pt"
    model, blob = load_policy(ckpt, device)
    log.info("device=%s", device)
    log.info(describe_params(model, blob))
    _, loader = build_loader(root, split="test", batch_size=int(cfg.get("batch_size", 4)))
    rows = []
    for drop in (0.0, 0.10, 0.20, 0.30):
        tsr = eval_dropout(model, loader, device, drop, pace, log)
        rows.append({"obs_dropout": drop, "tsr": tsr})
    out = root / "results" / "robustness" / "obs_dropout.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(out, index=False)
    log.info("wrote %s", out)


if __name__ == "__main__":
    main()
