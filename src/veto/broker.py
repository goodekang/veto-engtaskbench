from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from .blackboard import Blackboard, SubGoal
from .models import VetoPolicy
from .tools import shortlist, token_overlap, validate_registry


_WORD = re.compile(r"[a-z0-9_]+")


def _hash_embedding(text: str, dim: int = 384) -> np.ndarray:
    """Deterministic signed hashing embedding for offline dense retrieval."""
    vec = np.zeros(dim, dtype=np.float32)
    tokens = _WORD.findall(text.lower())
    features = tokens + [f"{a}:{b}" for a, b in zip(tokens, tokens[1:])]
    for feature in features:
        digest = hashlib.blake2b(feature.encode("utf-8"), digest_size=8).digest()
        raw = int.from_bytes(digest, "little")
        idx = raw % dim
        sign = -1.0 if (raw >> 8) & 1 else 1.0
        vec[idx] += sign * (1.0 + math.log1p(len(feature)))
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm else vec


def _tool_text(tool: dict[str, Any]) -> str:
    return " ".join(
        [
            str(tool.get("name", "")),
            str(tool.get("category", "")),
            str(tool.get("doc", "")),
            json.dumps(tool.get("args", {}), sort_keys=True),
            json.dumps(tool.get("postconditions", []), sort_keys=True),
        ]
    )


@dataclass
class DenseToolRetriever:
    registry: list[dict[str, Any]]
    dim: int = 384

    def __post_init__(self) -> None:
        validate_registry(self.registry)
        self._matrix = np.stack([_hash_embedding(_tool_text(t), self.dim) for t in self.registry])

    def search(self, query: str, domain: str, k: int) -> list[dict[str, Any]]:
        allowed = np.array(
            [tool.get("domain") in {domain, "shared"} for tool in self.registry],
            dtype=bool,
        )
        scores = self._matrix @ _hash_embedding(query, self.dim)
        scores[~allowed] = -np.inf
        order = np.argsort(-scores, kind="stable")
        return [
            {**self.registry[i], "retrieval_score": float(scores[i])}
            for i in order[: min(k, int(allowed.sum()))]
            if np.isfinite(scores[i])
        ]


def retrieve(
    registry: list[dict[str, Any]],
    goal: SubGoal,
    domain: str,
    k: int = 15,
) -> list[dict[str, Any]]:
    retriever = DenseToolRetriever(registry)
    query = f"{goal.name} {goal.evidence_type} {goal.acceptance}"
    candidates = retriever.search(query, domain, max(k, 15))
    hints = {
        "retrieve": ("entities_by_type",),
        "filter": ("egress_paths", "rel_"),
        "measure_clear_width": ("clear_width",),
        "compare": ("predicate_eval", "validate_"),
        "sketch": ("sketch_profile",),
        "build_body": ("extrude",),
        "holes": ("cq_hole", "cq_pattern"),
        "feature": ("cq_",),
        "validate": ("kernel_validate", "bbox_mass_validate"),
        "export": ("step_export", "evidence_export"),
    }

    def score(tool: dict[str, Any]) -> float:
        post = set(tool.get("postconditions", []))
        compatible = 1.0 if goal.evidence_type in post else 0.0
        lexical = token_overlap(query, _tool_text(tool))
        hint = 0.0
        for key, names in hints.items():
            if key in goal.name and any(name in tool["name"] for name in names):
                hint = 1.0
                break
        return (
            0.35 * float(tool.get("retrieval_score", 0.0))
            + 0.30 * lexical
            + 0.20 * compatible
            + 0.15 * hint
        )

    ranked = sorted(candidates, key=score, reverse=True)
    return [{**tool, "broker_score": score(tool)} for tool in ranked[:k]]


def _constraint_value(board: Blackboard, name: str, default: Any = None) -> Any:
    for constraint in board.constraints:
        if hasattr(constraint, "name") and constraint.name == name:
            return constraint.value
        if isinstance(constraint, dict) and constraint.get("name") == name:
            return constraint.get("value", default)
    return default


def _constraint_unit(board: Blackboard, name: str, default: str | None = None) -> str | None:
    for constraint in board.constraints:
        if hasattr(constraint, "name") and constraint.name == name:
            return constraint.unit or default
        if isinstance(constraint, dict) and constraint.get("name") == name:
            return constraint.get("unit") or default
    return default


def ground_args(
    goal: SubGoal,
    board: Blackboard,
    tool: dict[str, Any] | None = None,
) -> dict[str, Any]:
    args = {
        "artefact": board.artefact,
        "goal": goal.name,
        "evidence_type": goal.evidence_type,
    }
    if "width" in goal.name or "clear" in goal.acceptance:
        args["unit"] = _constraint_unit(board, "clear_width", "mm")
        args["threshold"] = float(_constraint_value(board, "clear_width", 850.0))
    if "hole" in goal.name or "pattern" in goal.acceptance:
        args["count"] = int(_constraint_value(board, "hole_count", 4))
        args["min_edge_mm"] = float(_constraint_value(board, "minimum_edge_distance", 12.0))
    if goal.evidence_type == "Verdict" and board.constraints:
        first = board.constraints[0]
        if hasattr(first, "name"):
            args["constraint_name"] = first.name
            args["operator"] = first.operator
            args["threshold"] = first.value
            if first.unit:
                args["unit"] = first.unit
    if board.counter_evidence:
        latest = board.counter_evidence[-1]
        args["repair_revision"] = len(board.counter_evidence)
        args["repair_family"] = latest.get("family")
        if latest.get("family") == "unit":
            args["unit"] = "mm"
        if latest.get("family") == "geometric" and "min_edge_mm" in args:
            args["min_edge_mm"] = max(float(args["min_edge_mm"]), 12.0) + 2.0
    if tool:
        schema = tool.get("args", {})
        for key, spec in schema.items():
            if key not in args and isinstance(spec, dict) and "default" in spec:
                args[key] = spec["default"]
        args["tool_name"] = tool.get("name")
    frame = board.context.get("coordinate_frame")
    if frame:
        args["coordinate_frame"] = frame
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
        policy_score = float(probs[idx]) if idx is not None else 0.0
        broker_score = float(tool.get("broker_score", tool.get("retrieval_score", 0.0)))
        score = 0.20 * policy_score + 0.80 * max(broker_score, 0.0)
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
    exclude: set[str] | None = None,
) -> tuple[dict[str, Any], dict[str, Any], list[dict[str, Any]]]:
    candidates = retrieve(registry, goal, domain, k=k)
    if exclude:
        candidates = [tool for tool in candidates if tool.get("name") not in exclude]
    if model is not None and obs is not None and lengths is not None:
        candidates = rerank_with_policy(model, obs, lengths, candidates, registry)
    if not candidates:
        candidates = shortlist(registry, k, domain=domain)
        if exclude:
            candidates = [tool for tool in candidates if tool.get("name") not in exclude]
    if not candidates:
        raise LookupError(f"no eligible tool for goal {goal.name!r} in domain {domain!r}")
    tool = candidates[0]
    args = ground_args(goal, board, tool)
    return tool, args, candidates
