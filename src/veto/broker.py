from __future__ import annotations

from typing import Any

import numpy as np
import torch

from .blackboard import Blackboard, SubGoal
from .models import VetoPolicy
from .tools import load_registry, shortlist, token_overlap


def retrieve(
    registry: list[dict[str, Any]],
    goal: SubGoal,
    domain: str,
    k: int = 15,
) -> list[dict[str, Any]]:
    pool = [t for t in registry if t.get("domain") in {domain, "shared"}]
    ranked = sorted(pool, key=lambda t: token_overlap(goal.acceptance + " " + goal.name, t.get("doc", "")), reverse=True)
    return ranked[:k]


def ground_args(goal: SubGoal, board: Blackboard) -> dict[str, Any]:
    args = {
        "artefact": board.artefact,
        "goal": goal.name,
        "evidence_type": goal.evidence_type,
    }
    if "width" in goal.name or "clear" in goal.acceptance:
        args["unit"] = "mm"
        args["threshold_mm"] = 850.0
    if "hole" in goal.name or "pattern" in goal.acceptance:
        args["count"] = 4
        args["min_edge_mm"] = 12.0
    return args


def rerank_with_policy(
    model: VetoPolicy,
    obs: torch.Tensor,
    lengths: torch.Tensor,
    candidates: list[dict[str, Any]],
    registry: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    probs = model.broker_probs(obs, lengths)[0].detach().cpu().numpy()
    name_to_idx = {t["name"]: i for i, t in enumerate(registry)}
    scored = []
    for tool in candidates:
        idx = name_to_idx.get(tool["name"])
        score = float(probs[idx]) if idx is not None else 0.0
        scored.append((score, tool))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [t for _, t in scored]


def select_tool(
    board: Blackboard,
    goal: SubGoal,
    registry: list[dict[str, Any]],
    *,
    domain: str,
    k: int = 15,
    model: VetoPolicy | None = None,
    obs: torch.Tensor | None = None,
    lengths: torch.Tensor | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidates = retrieve(registry, goal, domain, k=k)
    if model is not None and obs is not None and lengths is not None:
        candidates = rerank_with_policy(model, obs, lengths, candidates, registry)
    if not candidates:
        candidates = shortlist(registry, k, domain=domain)
    tool = candidates[0]
    args = ground_args(goal, board)
    return tool, args, candidates
