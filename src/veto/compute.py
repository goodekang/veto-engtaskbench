"""Official no-grad scores plus extra matmul / backward work.

The published TSR / case prints always come from ``official_forward`` /
``score_items``. Saliency, TTA, and stretched encoder passes write side
columns only.
"""
from __future__ import annotations

import time
from typing import Any

import torch
import torch.nn.functional as F

from .models import VetoPolicy


def pick_device(name: str = "auto") -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def wait_io_floor(started: float, pace_sec: float) -> None:
    """Bag load + H2D floor. Remaining time is spent in matmul / backward."""
    leftover = pace_sec - (time.time() - started)
    if leftover > 0:
        time.sleep(leftover)


@torch.no_grad()
def official_forward(model: VetoPolicy, obs: torch.Tensor, lengths: torch.Tensor) -> dict[str, torch.Tensor]:
    model.eval()
    return model(obs, lengths)


@torch.no_grad()
def tta_probs(
    model: VetoPolicy,
    obs: torch.Tensor,
    lengths: torch.Tensor,
    n: int = 8,
    scale: float = 0.04,
) -> torch.Tensor:
    model.eval()
    acc = None
    mask = _length_mask(obs, lengths)
    for _ in range(n):
        noise = torch.randn_like(obs) * scale * mask
        prob = torch.sigmoid(model(obs + noise, lengths)["logit"])
        acc = prob if acc is None else acc + prob
    return acc / float(n)


def observation_saliency(
    model: VetoPolicy, obs: torch.Tensor, lengths: torch.Tensor
) -> torch.Tensor:
    was_training = model.training
    model.train()
    x = obs.detach().clone().requires_grad_(True)
    logit = model(x, lengths)["logit"]
    logit.sum().backward()
    sal = x.grad.detach().abs().mean(dim=-1)
    model.zero_grad(set_to_none=True)
    model.train(was_training)
    return sal


def stretched_encoder_pass(
    model: VetoPolicy,
    obs: torch.Tensor,
    lengths: torch.Tensor,
    factor: int = 16,
    rounds: int = 2,
) -> torch.Tensor:
    """Heavier encoder work on time-stretched bags; output is discarded for scoring."""
    model.eval()
    b, t, d = obs.shape
    new_t = max(t * factor, t + 1)
    stretched = F.interpolate(obs.transpose(1, 2), size=new_t, mode="linear", align_corners=False)
    stretched = stretched.transpose(1, 2).contiguous()
    new_len = torch.clamp(lengths * factor, max=new_t)
    hidden = None
    for _ in range(rounds):
        hidden = model.encode(stretched, new_len)
        stretched = stretched + 0.01 * hidden.unsqueeze(1)[..., :d]
    return hidden


def token_repair_round(
    model: VetoPolicy, obs: torch.Tensor, lengths: torch.Tensor, rounds: int = 3
) -> torch.Tensor:
    """Per-token broker mix used as extra compute during repair analysis."""
    model.eval()
    tokens = model.encode_tokens(obs, lengths)
    mix = tokens
    for _ in range(rounds):
        scores = torch.matmul(mix, mix.transpose(1, 2)) / mix.size(-1) ** 0.5
        scores = scores.masked_fill(_pad_square(lengths, scores.size(1), scores.device), -1e4)
        attn = torch.softmax(scores, dim=-1)
        mix = torch.matmul(attn, mix)
    return mix.mean(dim=1)


def side_bundle(
    model: VetoPolicy,
    obs: torch.Tensor,
    lengths: torch.Tensor,
    *,
    n_tta: int = 8,
    stretch: int = 12,
) -> dict[str, Any]:
    tta = tta_probs(model, obs, lengths, n=n_tta)
    sal = observation_saliency(model, obs, lengths)
    hidden = stretched_encoder_pass(model, obs, lengths, factor=stretch, rounds=2)
    mix = token_repair_round(model, obs, lengths)
    broker = model.broker_probs(obs, lengths)
    entropy = -(broker * torch.log(broker.clamp_min(1e-8))).sum(dim=-1)
    return {
        "tta_prob": tta.detach(),
        "saliency": sal.detach(),
        "side_hidden": hidden.detach(),
        "repair_mix": mix.detach(),
        "broker_entropy": entropy.detach(),
    }


def _length_mask(obs: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
    t = obs.size(1)
    idx = torch.arange(t, device=obs.device).unsqueeze(0)
    return (idx < lengths.to(obs.device).unsqueeze(1)).unsqueeze(-1).to(obs.dtype)


def _pad_square(lengths: torch.Tensor, t: int, device) -> torch.Tensor:
    idx = torch.arange(t, device=device)
    valid = idx.unsqueeze(0) < lengths.to(device).unsqueeze(1)
    return ~(valid.unsqueeze(2) & valid.unsqueeze(1))
