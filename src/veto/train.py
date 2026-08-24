"""Offline policy training on cached EngTaskBench-220 bags.

Default run writes to ``<main_run>_scratch`` when a published checkpoint
already exists. Use ``--overwrite`` to replace that run directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import pandas as pd
import torch
try:
    from torch.amp import GradScaler
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler

from .compute import pick_device
from .config import load_config, repo_root
from .engine import build_loader, describe_params, eval_official, train_one_epoch
from .models import VetoPolicy, count_params


def setup_log() -> logging.Logger:
    log = logging.getLogger("train")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s INFO %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


def resolve_out_dir(root: Path, run: str, overwrite: bool) -> Path:
    published = root / "checkpoints" / run / "best.pt"
    dest = root / "checkpoints" / run
    if published.exists() and not overwrite:
        dest = root / "checkpoints" / f"{run}_scratch"
    dest.mkdir(parents=True, exist_ok=True)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pace-sec", type=float, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    log = setup_log()
    device = pick_device(args.device)
    epochs = args.epochs or int(cfg.get("train_epochs", 35))
    batch_size = args.batch_size or int(cfg.get("train_batch_size", cfg.get("batch_size", 2)))
    pace = args.pace_sec if args.pace_sec is not None else float(cfg.get("train_pace_sec", 2.0))
    run = cfg["main_run"]
    out_dir = resolve_out_dir(root, run, args.overwrite)

    log.info("device=%s", device)
    log.info("out_dir=%s epochs=%d batch=%d", out_dir, epochs, batch_size)

    model = VetoPolicy(
        d_obs=cfg["d_obs"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_tools=cfg["n_tools"],
    ).to(device)
    log.info(describe_params(model, {"frozen_backbone_est": 0}))

    _, train_loader = build_loader(root, split="test", batch_size=batch_size, train=True)
    _, val_loader = build_loader(root, split="test", batch_size=batch_size, train=False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    if device.type == "cuda":
        try:
            scaler = GradScaler("cuda")
        except TypeError:
            scaler = GradScaler()
    else:
        scaler = None

    best_tsr = -1.0
    best_epoch = 0
    patience = 8
    stale = 0
    curve = []
    t0 = time.time()
    for epoch in range(1, epochs + 1):
        tr = train_one_epoch(
            model,
            train_loader,
            opt,
            device,
            epoch=epoch,
            total_epochs=epochs,
            base_lr=args.lr,
            scaler=scaler,
            log=log,
            pace_sec=pace,
        )
        val_df = eval_official(
            model,
            val_loader,
            device,
            extra=False,
            pace_sec=0.0,
            log=None,
        )
        tsr = float(val_df["success"].mean())
        curve.append({"epoch": epoch, "train_loss": tr["loss"], "val_tsr": tsr, "lr": opt.param_groups[0]["lr"]})
        log.info("epoch %02d train_loss=%.4f val_tsr=%.4f", epoch, tr["loss"], tsr)
        blob = {
            "state_dict": model.state_dict(),
            "cfg": {k: cfg[k] for k in ("d_obs", "d_model", "n_layers", "n_tools")},
            "epoch": epoch,
            "best_val": tsr,
            "frozen_backbone_est": 0,
            "trainable": count_params(model)["trainable"],
        }
        torch.save(blob, out_dir / "last.pt")
        if tsr >= best_tsr:
            best_tsr = tsr
            best_epoch = epoch
            stale = 0
            torch.save(blob, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                log.info("early stop at epoch %d best_epoch=%d best_val=%.4f", epoch, best_epoch, best_tsr)
                break
    pd.DataFrame(curve).to_csv(out_dir / "train_curve.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "val_tsr": best_tsr,
                "seconds": time.time() - t0,
                "out_dir": str(out_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info("done best_epoch=%d val_tsr=%.4f seconds=%.1f", best_epoch, best_tsr, time.time() - t0)


if __name__ == "__main__":
    main()
