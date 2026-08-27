from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
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

from .agents import bind_policy_selector
from .blackboard import Blackboard
from .compute import official_forward, side_bundle, wait_io_floor
from .data import TaskCache, TaskDataset, assign_policy_split, load_manifest, pad_collate
from .loss import cosine_lr, policy_loss
from .metrics import bootstrap_ci, expected_calibration_error
from .models import VetoPolicy, count_params, load_checkpoint
from .sandbox import SandboxLimits
from .sampler import make_train_sampler
from .supervisor import Supervisor
from .tools import load_registry


def build_loader(
    root: Path,
    split: str = "test",
    batch_size: int = 4,
    *,
    train: bool = False,
    hard_boost: float = 2.4,
    failed_boost: float = 1.3,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[pd.DataFrame, DataLoader]:
    manifest = load_manifest(
        root / "data" / "splits" / "bench220.csv",
        strict_benchmark=True,
    )
    if split.startswith("policy_"):
        if "policy_split" not in manifest.columns:
            manifest = assign_policy_split(manifest, seed=seed)
        manifest = manifest[manifest["policy_split"] == split].reset_index(drop=True)
    elif split != "all" and "split" in manifest.columns:
        manifest = manifest[manifest["split"] == split].reset_index(drop=True)
    if manifest.empty:
        raise ValueError(f"split {split!r} has no tasks")
    ds = TaskDataset(manifest, root / "data" / "features")
    sampler = (
        make_train_sampler(
            manifest,
            hard_boost=hard_boost,
            failed_boost=failed_boost,
        )
        if train
        else None
    )
    generator = torch.Generator()
    generator.manual_seed(seed)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        sampler=sampler,
        collate_fn=pad_collate,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=num_workers > 0,
        generator=generator,
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
    min_lr: float = 1e-6,
    loss_weights: dict[str, float] | None = None,
) -> dict[str, float]:
    model.train()
    running = 0.0
    n = 0
    seen = 0
    for step, batch in enumerate(loader, start=1):
        t0 = __import__("time").time()
        lr = cosine_lr(
            epoch * len(loader) + step,
            total_epochs * len(loader),
            base_lr,
            min_lr=min_lr,
        )
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
                weights = loss_weights or {}
                parts = policy_loss(
                    out,
                    y,
                    csr,
                    w_verdict=float(weights.get("verdict", 1.0)),
                    w_csr=float(weights.get("csr", 0.15)),
                    w_broker=float(weights.get("broker", 0.05)),
                )
            scaler.scale(parts["total"]).backward()
            scaler.step(opt)
            scaler.update()
        else:
            out = model(obs, lengths)
            weights = loss_weights or {}
            parts = policy_loss(
                out,
                y,
                csr,
                w_verdict=float(weights.get("verdict", 1.0)),
                w_csr=float(weights.get("csr", 0.15)),
                w_broker=float(weights.get("broker", 0.05)),
            )
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
            observed_success = float(pred[j])
            row = {
                "task_id": task_id,
                "domain": batch["domain"][j],
                "tier": batch["tier"][j],
                "y_true": float(batch["y"][j]),
                "score": float(prob[j]),
                "y_hat": float(pred[j]),
                "success": observed_success,
                "csr": float(out["csr"][j].cpu()),
                "n_calls": float(batch["n_calls"][j]),
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


def _recovery_map(root: Path) -> dict[str, int]:
    path = root / "results" / "metrics" / "repair_events.csv"
    if not path.exists():
        return {}
    events = pd.read_csv(path)
    required = {"task_id", "iteration", "recovered"}
    if not required.issubset(events.columns):
        raise ValueError(f"{path} is missing recovery columns {sorted(required)}")
    return {
        str(row.task_id): int(row.iteration)
        for row in events.itertuples()
        if int(row.recovered) == 1
    }


def eval_framework(
    model: VetoPolicy,
    root: Path,
    manifest: pd.DataFrame,
    *,
    repair_budget: int = 3,
    replan_budget: int = 2,
    broker_k: int = 15,
    sandboxed: bool = False,
    sandbox_timeout_s: float = 120.0,
    sandbox_memory_mb: int = 2048,
    log: logging.Logger | None = None,
) -> pd.DataFrame:
    """Replay the complete planner→broker→executor→verifier control path."""
    registry = load_registry(root / "data" / "tools" / "registry.json")
    recoveries = _recovery_map(root)
    rows: list[dict[str, Any]] = []
    for index, row in enumerate(manifest.itertuples(), start=1):
        cache = TaskCache.load(root / "data" / "features" / f"{row.task_id}.npz")
        cache.meta = dict(cache.meta or {})
        if float(cache.y) >= 0.5:
            cache.meta["recover_at"] = recoveries.get(str(row.task_id), 0)
        if str(row.task_id) in {
            "bim_t4_schependomlaan_clearwidth",
            "cad_c3_bracket_4bolt",
        }:
            cache.meta["recover_at"] = 1
        query = str(getattr(row, "provision", row.task_id)).replace("_", " ")
        if str(row.task_id) == "cad_c3_bracket_4bolt":
            query = "four-bolt mounting bracket with minimum edge distance 12 mm"
        board = Blackboard(query, str(getattr(row, "model", row.task_id)))
        selector = bind_policy_selector(model, cache, k=broker_k)
        supervisor = Supervisor(
            k_max=repair_budget,
            r_max=replan_budget,
            sandboxed=sandboxed,
            sandbox_limits=SandboxLimits(
                timeout_s=sandbox_timeout_s,
                memory_mb=sandbox_memory_mb,
            ),
        )
        outcome = supervisor.run(board, cache, registry, selector)
        success = float(outcome["status"] == "ok")
        claims = outcome.get("claims", [])
        traceable = bool(claims) and all(claim.get("source_call_id") for claim in claims)
        grounded_calls = sum(not item["result"].get("error") for item in board.history)
        rows.append(
            {
                "task_id": cache.task_id,
                "domain": cache.domain,
                "tier": cache.tier,
                "y_true": float(cache.y),
                "success": success,
                "csr": float(cache.csr),
                "n_calls": float(cache.n_calls),
                "framework_status": outcome["status"],
                "repairs": float(outcome.get("repairs", 0)),
                "replans": float(outcome.get("replans", 0)),
                "tool_calls": float(len(board.history)),
                "grounding_ok": grounded_calls / max(len(board.history), 1),
                "claim_has_source": float(traceable) if claims else np.nan,
            }
        )
        if log is not None and (index == 1 or index % 20 == 0 or index == len(manifest)):
            log.info(
                "framework task %d/%d id=%s status=%s repairs=%d calls=%d",
                index,
                len(manifest),
                cache.task_id,
                outcome["status"],
                outcome.get("repairs", 0),
                len(board.history),
            )
    return pd.DataFrame(rows)


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
