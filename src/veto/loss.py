from __future__ import annotations

import torch
import torch.nn.functional as F


def verdict_logit_target(y: torch.Tensor, pos: float = 4.0, neg: float = -4.0) -> torch.Tensor:
    return y * (pos - neg) + neg


def policy_loss(
    out: dict[str, torch.Tensor],
    y: torch.Tensor,
    csr: torch.Tensor | None = None,
    *,
    w_verdict: float = 1.0,
    w_csr: float = 0.15,
    w_broker: float = 0.05,
) -> dict[str, torch.Tensor]:
    target = verdict_logit_target(y)
    l_verdict = F.mse_loss(out["logit"], target)
    loss = w_verdict * l_verdict
    parts = {"verdict": l_verdict}
    if csr is not None:
        l_csr = F.binary_cross_entropy(out["csr"].clamp(1e-4, 1 - 1e-4), csr.clamp(0, 1))
        loss = loss + w_csr * l_csr
        parts["csr"] = l_csr
    # keep broker mass off a single tool
    probs = torch.softmax(out["broker"], dim=-1)
    l_broker = (probs.max(dim=-1).values.mean() - 1.0 / probs.size(-1)).abs()
    loss = loss + w_broker * l_broker
    parts["broker"] = l_broker
    parts["total"] = loss
    return parts


def cosine_lr(step: int, total: int, base: float, min_lr: float = 1e-6) -> float:
    if total <= 1:
        return base
    import math

    frac = min(max(step / float(total), 0.0), 1.0)
    return min_lr + 0.5 * (base - min_lr) * (1.0 + math.cos(math.pi * frac))
