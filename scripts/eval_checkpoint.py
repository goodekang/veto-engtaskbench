#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veto.backbone import CachedCompletionEncoder
from veto.compute import pick_device
from veto.config import load_config, repo_root
from veto.engine import (
    build_loader,
    describe_params,
    eval_framework,
    eval_official,
    load_policy,
    summarize_eval,
)


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
    parser.add_argument("--config", default="configs/veto.yaml")
    parser.add_argument("--ckpt", default=None)
    parser.add_argument("--split", default="test")
    parser.add_argument("--pace-sec", type=float, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--tta", type=int, default=8)
    parser.add_argument("--stretch", type=int, default=12)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    if cfg.get("provider_mode") != "cached_replay":
        raise ValueError("eval_checkpoint requires provider_mode=cached_replay")
    root = repo_root()
    log = setup_log()
    ckpt_path = Path(args.ckpt) if args.ckpt else root / "checkpoints" / cfg["main_run"] / "best.pt"
    if not ckpt_path.is_absolute():
        ckpt_path = root / ckpt_path
    pace = args.pace_sec if args.pace_sec is not None else float(cfg.get("pace_sec", 1.0))
    batch_size = args.batch_size or int(cfg.get("batch_size", 4))
    device = pick_device(args.device)

    log.info("device=%s", device)
    torch.manual_seed(int(cfg.get("seed", 42)))
    probe = CachedCompletionEncoder().probe()
    log.info("llm_cache available=%s download=False", probe["available"])
    sha_path = ckpt_path.with_name("best.sha256")
    if sha_path.exists():
        expected_sha = sha_path.read_text(encoding="utf-8").strip()
        actual_sha = hashlib.sha256(ckpt_path.read_bytes()).hexdigest()
        if actual_sha != expected_sha:
            raise ValueError(f"checkpoint SHA-256 mismatch for {ckpt_path}")
    model, blob = load_policy(ckpt_path, device)
    if blob.get("artifact_role") == "policy_surrogate" and args.split == "test":
        raise ValueError(
            "scratch policy checkpoints must be evaluated on policy_test; "
            "the 220-task paper replay is reserved for the published replay artefact"
        )
    log.info("loaded %s epoch=%s best_val=%.4f", ckpt_path, blob.get("epoch"), blob.get("best_val", 0))
    log.info(describe_params(model, blob))

    manifest, loader = build_loader(root, split=args.split, batch_size=batch_size, train=False)
    log.info("n_tasks=%d batches=%d", len(loader.dataset), len(loader))

    t0 = time.time()
    surrogate = eval_official(
        model,
        loader,
        device,
        extra=True,
        n_tta=args.tta,
        stretch=args.stretch,
        pace_sec=pace,
        log=log,
    )
    framework = eval_framework(
        model,
        root,
        manifest,
        repair_budget=int(cfg.get("repair_budget", 3)),
        replan_budget=int(cfg.get("replan_budget", 2)),
        broker_k=int(cfg.get("broker_k", 15)),
        sandboxed=bool(cfg.get("replay_sandboxed", False)),
        sandbox_timeout_s=float(cfg.get("sandbox_timeout_s", 120)),
        sandbox_memory_mb=int(cfg.get("sandbox_memory_mb", 2048)),
        log=log,
    )
    score_columns = surrogate[
        [
            "task_id",
            "score",
            "y_hat",
            "score_tta",
            "saliency_mean",
            "broker_entropy",
        ]
    ].rename(columns={"y_hat": "surrogate_y_hat"})
    df = framework.merge(score_columns, on="task_id", validate="one_to_one")
    match_accuracy = float(
        (df["surrogate_y_hat"].astype(int) == df["y_true"].astype(int)).mean()
    )
    summary = summarize_eval(df)
    out_path = Path(args.out) if args.out else root / "results" / "predictions" / "eval_replay.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    elapsed = time.time() - t0
    log.info(
        "done  seconds=%.1f  s/task=%.3f  framework_tsr=%.3f  "
        "surrogate_match=%.3f  ci95=[%.3f, %.3f]  ece=%.3f  out=%s",
        elapsed,
        elapsed / max(len(df), 1),
        summary["tsr"],
        match_accuracy,
        summary["ci_lo"],
        summary["ci_hi"],
        summary.get("ece", 0.0),
        out_path,
    )


if __name__ == "__main__":
    main()
