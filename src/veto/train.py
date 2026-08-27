"""Offline policy training on cached EngTaskBench-220 bags.

Default run writes to ``<main_run>_scratch`` when a published checkpoint
already exists. Use ``--overwrite`` to replace that run directory.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path

import numpy as np
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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--pace-sec", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    root = repo_root()
    log = setup_log()
    seed = int(cfg.get("seed", 42))
    seed_everything(seed)
    device = pick_device(args.device or str(cfg.get("device", "auto")))
    epochs = args.epochs if args.epochs is not None else int(cfg.get("train_epochs", 35))
    batch_size = (
        args.batch_size
        if args.batch_size is not None
        else int(cfg.get("train_batch_size", cfg.get("batch_size", 2)))
    )
    lr = args.lr if args.lr is not None else float(cfg.get("lr", 3e-4))
    num_workers = (
        args.num_workers
        if args.num_workers is not None
        else int(cfg.get("num_workers", 0))
    )
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

    train_split = str(cfg.get("train_split", "policy_train"))
    val_split = str(cfg.get("val_split", "policy_val"))
    _, train_loader = build_loader(
        root,
        split=train_split,
        batch_size=batch_size,
        train=True,
        hard_boost=float(cfg.get("hard_tier_boost", 2.4)),
        failed_boost=float(cfg.get("sampling", {}).get("failed_task_boost", 1.3)),
        num_workers=num_workers,
        seed=seed,
    )
    _, val_loader = build_loader(
        root,
        split=val_split,
        batch_size=batch_size,
        train=False,
        num_workers=num_workers,
        seed=seed,
    )
    opt = torch.optim.AdamW(
        model.parameters(),
        lr=lr,
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
    )
    if device.type == "cuda" and bool(cfg.get("amp", True)):
        try:
            scaler = GradScaler("cuda")
        except TypeError:
            scaler = GradScaler()
    else:
        scaler = None

    best_match = -1.0
    best_epoch = 0
    patience = int(cfg.get("early_stop_patience", 8))
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
            base_lr=lr,
            scaler=scaler,
            log=log,
            pace_sec=pace,
            min_lr=float(cfg.get("scheduler", {}).get("min_lr", 1e-6)),
            loss_weights=cfg.get("loss", {}),
        )
        val_df = eval_official(
            model,
            val_loader,
            device,
            extra=False,
            pace_sec=0.0,
            log=None,
        )
        match_accuracy = float(
            (val_df["y_hat"].astype(int) == val_df["y_true"].astype(int)).mean()
        )
        curve.append(
            {
                "epoch": epoch,
                "train_loss": tr["loss"],
                "val_match_accuracy": match_accuracy,
                "lr": opt.param_groups[0]["lr"],
            }
        )
        log.info(
            "epoch %02d train_loss=%.4f val_match_accuracy=%.4f",
            epoch,
            tr["loss"],
            match_accuracy,
        )
        blob = {
            "state_dict": model.state_dict(),
            "cfg": {k: cfg[k] for k in ("d_obs", "d_model", "n_layers", "n_tools")},
            "epoch": epoch,
            "best_val": match_accuracy,
            "frozen_backbone_est": 0,
            "trainable": count_params(model)["trainable"],
            "optimizer": opt.state_dict(),
            "scaler": scaler.state_dict() if scaler is not None else None,
            "seed": seed,
            "train_split": train_split,
            "val_split": val_split,
            "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "artifact_role": "policy_surrogate",
            "selection_metric": "val_match_accuracy",
        }
        torch.save(blob, out_dir / "last.pt")
        if match_accuracy >= best_match:
            best_match = match_accuracy
            best_epoch = epoch
            stale = 0
            torch.save(blob, out_dir / "best.pt")
        else:
            stale += 1
            if stale >= patience:
                log.info(
                    "early stop at epoch %d best_epoch=%d best_match=%.4f",
                    epoch,
                    best_epoch,
                    best_match,
                )
                break
    pd.DataFrame(curve).to_csv(out_dir / "train_curve.csv", index=False)
    (out_dir / "metrics.json").write_text(
        json.dumps(
            {
                "best_epoch": best_epoch,
                "val_match_accuracy": best_match,
                "seconds": time.time() - t0,
                "out_dir": str(out_dir),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    log.info(
        "done best_epoch=%d val_match_accuracy=%.4f seconds=%.1f",
        best_epoch,
        best_match,
        time.time() - t0,
    )


if __name__ == "__main__":
    main()
