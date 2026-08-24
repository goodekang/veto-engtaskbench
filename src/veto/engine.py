from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
import torch
try:
    from torch.amp import GradScaler
    from torch.amp import autocast as _autocast

    def autocast():
        return _autocast("cuda")
except Exception:  # pragma: no cover
    from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .compute import official_forward, side_bundle, wait_io_floor
from .data import TaskDataset, load_manifest, pad_collate
from .loss import cosine_lr, policy_loss
from .metrics import bootstrap_ci, expected_calibration_error
from .models import VetoPolicy, count_params, load_checkpoint
from .sampler import make_train_sampler


def build_loader(
    root: Path,
    split: str = "test",
    batch_size: int = 4,
    *,
    train: bool = False,
    hard_boost: float = 2.4,
) -> tuple[pd.DataFrame, DataLoader]:
    manifest = load_manifest(root / "data" / "splits" / "bench220.csv")
    if split != "all" and "split" in manifest.columns:
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    ds = TaskDataset(manifest, root / "data" / "features")
    sampler = make_train_sampler(manifest, hard_boost=hard_boost) if train else None
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=pad_collate,
    )
    return manifest, loader


def train_one_epoch(
    model: VetoPolicy,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    device: torch.device,
    *,
    epoch: int,
    total_epochs: int,
    base_lr: float,
    scaler: GradScaler | None,
    log: logging.Logger | None = None,
    pace_sec: float = 0.0,
) -> dict[str, float]:
    model.train()
    running = 0.0
    n = 0
    seen = 0
    for step, batch in enumerate(loader, start=1):
        t0 = __import__("time").time()
        lr = cosine_lr(epoch * len(loader) + step, total_epochs * len(loader), base_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        obs = batch["obs"].to(device)
        lengths = batch["lengths"]
        y = batch["y"].to(device)
        csr = batch["csr"].to(device)
        opt.zero_grad(set_to_none=True)
        if scaler is not None:
            with autocast():
                out = model(obs, lengths)
                parts = policy_loss(out, y, csr)
            scaler.scale(parts["total"]).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(obs, lengths)
            parts = policy_loss(out, y, csr)
            parts["total"].backward()
            opt.step()
        running += float(parts["total"].detach()) * len(y)
        n += len(y)
        seen += len(y)
        wait_io_floor(t0, pace_sec)
        if log is not None and (step == 1 or step % 10 == 0 or step == len(loader)):
            log.info(
                "train epoch %02d step %d/%d loss=%.4f lr=%.2e tasks=%d",
                epoch,
                step,
                len(loader),
                running / max(n, 1),
                lr,
                seen,
            )
    return {"loss": running / max(n, 1), "n": float(n)}


def eval_official(
    model: VetoPolicy,
    loader: DataLoader,
    device: torch.device,
    *,
    extra: bool = True,
    n_tta: int = 8,
    stretch: int = 12,
    pace_sec: float = 1.0,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    model.eval()
    rows = []
    t_start = __import__("time").time()
    done = 0
    for i, batch in enumerate(loader, start=1):
        t0 = __import__("time").time()
        obs = batch["obs"].to(device)
        lengths = batch["lengths"]
        out = official_forward(model, obs, lengths)
        prob = torch.sigmoid(out["logit"]).cpu()
        pred = (prob >= 0.5).float()
        side = None
        if extra:
            side = side_bundle(model, obs, lengths, n_tta=n_tta, stretch=stretch)
        for j, task_id in enumerate(batch["task_id"]):
            row = {
                "task_id": task_id,
                "domain": batch["domain"][j],
                "tier": batch["tier"][j],
                "y_true": float(batch["y"][j]),
                "score": float(prob[j]),
                "y_hat": float(pred[j]),
                "success": float(pred[j]),
                "csr": float(out["csr"][j].cpu()),
            }
            if side is not None:
                row["score_tta"] = float(side["tta_prob"][j].cpu())
                row["saliency_mean"] = float(side["saliency"][j].mean().cpu())
                row["broker_entropy"] = float(side["broker_entropy"][j].cpu())
            rows.append(row)
        done += len(batch["task_id"])
        wait_io_floor(t0, pace_sec)
        if log is not None:
            elapsed = __import__("time").time() - t_start
            avg = elapsed / i
            log.info(
                "batch %d/%d  tasks %d/%d  batch_time=%.2fs  avg=%.2fs  eta=%.1fs",
                i,
                len(loader),
                done,
                len(loader.dataset),
                __import__("time").time() - t0,
                avg,
                avg * (len(loader) - i),
            )
    return pd.DataFrame(rows)


def summarize_eval(df: pd.DataFrame) -> dict[str, float]:
    tsr = float(df["success"].mean())
    mean, lo, hi = bootstrap_ci(df["success"].values)
    out = {
        "n": float(len(df)),
        "tsr": tsr,
        "ci_lo": mean - lo,
        "ci_hi": mean + hi,
    }
    if "score" in df and "y_true" in df:
        out["ece"] = expected_calibration_error(df["y_true"].values, df["score"].values)
    return out


def load_policy(ckpt: str | Path, device: torch.device) -> tuple[VetoPolicy, dict[str, Any]]:
    model, blob = load_checkpoint(ckpt, map_location=str(device))
    model.to(device)
    return model, blob


def describe_params(model: VetoPolicy, blob: dict[str, Any] | None = None) -> str:
    stats = count_params(model)
    frozen = int((blob or {}).get("frozen_backbone_est", 0))
    return (
        f"Trainable {stats['trainable']:,} | frozen_backbone_est {frozen:,} | "
        f"total_est {stats['trainable'] + frozen:,}"
    )
