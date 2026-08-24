from __future__ import annotations

from typing import Any

from .blackboard import Blackboard, SubGoal
from .broker import select_tool
from .data import TaskCache
from .models import VetoPolicy
from .planner import plan_for
from .tools import load_registry


def plan_clearwidth(board: Blackboard) -> list[SubGoal]:
    return plan_for(board, tier="T4", domain="bim")


def plan_bracket(board: Blackboard) -> list[SubGoal]:
    return plan_for(board, tier="C3", domain="cad")


def broker_for(board: Blackboard, registry, k: int = 15, goal: SubGoal | None = None):
    goal = goal or (board.plan[0] if board.plan else SubGoal("probe", "Verdict", "any"))
    domain = "bim" if "ifc" in board.artefact.lower() or "schepen" in board.artefact.lower() else "cad"
    tool, args, short = select_tool(board, goal, registry, domain=domain, k=k)
    return short


def bind_policy_selector(model: VetoPolicy, cache: TaskCache, k: int = 15):
    import torch

    obs = torch.from_numpy(cache.obs).unsqueeze(0)
    lengths = torch.tensor([cache.obs.shape[0]])

    def _select(board, goal, registry):
        return select_tool(
            board,
            goal,
            registry,
            domain=cache.domain,
            k=k,
            model=model,
            obs=obs,
            lengths=lengths,
        )

    return _select


def load_domain_registry(path) -> list[dict[str, Any]]:
    return load_registry(path)
