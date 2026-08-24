#!/usr/bin/env python
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veto.blackboard import Blackboard
from veto.compute import official_forward, pick_device, side_bundle, wait_io_floor
from veto.config import load_config, repo_root
from veto.engine import build_loader, describe_params, load_policy
from veto.loss import policy_loss
from veto.models import VetoPolicy, count_params, load_checkpoint
from veto.planner import plan_for
from veto.supervisor import State, next_state


def setup_log() -> logging.Logger:
    log = logging.getLogger("smoke")
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s INFO %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
    log.setLevel(logging.INFO)
    return log


def main() -> None:
    cfg = load_config()
    root = repo_root()
    log = setup_log()
    device = pick_device(str(cfg.get("device", "auto")))
    log.info("device=%s", device)

    scratch = VetoPolicy(
        d_obs=cfg["d_obs"],
        d_model=cfg["d_model"],
        n_layers=cfg["n_layers"],
        n_tools=cfg["n_tools"],
    ).to(device)
    log.info(describe_params(scratch, {"frozen_backbone_est": 0}))

    _, loader = build_loader(root, split="test", batch_size=4, train=False)
    opt = torch.optim.AdamW(scratch.parameters(), lr=3e-4, weight_decay=1e-4)
    steps = 12
    pace = float(cfg.get("smoke_pace_sec", 2.0))
    t0 = time.time()
    scratch.train()
    it = iter(loader)
    for step in range(1, steps + 1):
        bt = time.time()
        try:
            batch = next(it)
        except StopIteration:
            it = iter(loader)
            batch = next(it)
        obs = batch["obs"].to(device)
        lengths = batch["lengths"]
        y = batch["y"].to(device)
        csr = batch["csr"].to(device)
        opt.zero_grad(set_to_none=True)
        out = scratch(obs, lengths)
        parts = policy_loss(out, y, csr)
        parts["total"].backward()
        opt.step()
        wait_io_floor(bt, pace)
        log.info("adamw step %d/%d loss=%.4f batch_time=%.2fs", step, steps, float(parts["total"].detach()), time.time() - bt)

    ckpt = root / "checkpoints" / cfg["main_run"] / "best.pt"
    published, blob = load_checkpoint(ckpt, map_location=str(device))
    published.to(device)
    published.eval()
    batch = next(iter(loader))
    obs = batch["obs"].to(device)
    lengths = batch["lengths"]
    official = official_forward(published, obs, lengths)
    side = side_bundle(published, obs, lengths, n_tta=6, stretch=8)
    board = Blackboard("smoke clear-width check", "cache://schependomlaan")
    plan_for(board, tier="T4", domain="bim")
    log.info("published %s", describe_params(published, blob))
    log.info(
        "reload ok  logit=%s  tta=%s  saliency=%.4f  fsm %s -> %s",
        [round(float(v), 3) for v in official["logit"][:2].detach().cpu()],
        [round(float(v), 3) for v in side["tta_prob"][:2].cpu()],
        float(side["saliency"].mean()),
        State.VERIFY,
        next_state(State.VERIFY, True, 0, 0),
    )
    log.info("smoke_forward ok  seconds=%.1f  n_params=%d", time.time() - t0, count_params(published)["trainable"])


if __name__ == "__main__":
    main()
