from __future__ import annotations

from typing import Any

import numpy as np

from .blackboard import Blackboard, SubGoal
from .data import TaskCache
from .verifier import canonicalize_width, geometric_oracle, schema_ok, unit_range_ok


def run_tool(
    tool: dict[str, Any],
    args: dict[str, Any],
    cache: TaskCache,
    goal: SubGoal,
) -> dict[str, Any]:
    name = tool.get("name", "")
    if not schema_ok(args, ["artefact", "goal"]):
        return {"ok": False, "error": "schema", "tool": name}
    if goal.evidence_type == "QuantityTable" and cache.items is not None:
        widths = cache.items[:, 0] * 1000.0
        return {
            "ok": True,
            "tool": name,
            "kind": "widths",
            "values": widths.astype(float),
            "unit": args.get("unit", "mm"),
        }
    if goal.evidence_type == "Verdict" and cache.items is not None:
        if cache.domain == "cad":
            scores = cache.items[:, 0]
            return {"ok": True, "tool": name, "kind": "constraints", "values": scores.astype(float)}
        widths = cache.items[:, 0] * 1000.0
        return {"ok": True, "tool": name, "kind": "verdict", "values": widths.astype(float)}
    if goal.evidence_type == "SolidModel":
        geom = geometric_oracle(cache)
        return {"ok": geom["ok"], "tool": name, "kind": "solid", **geom}
    pooled = cache.obs.mean(axis=0)
    return {
        "ok": True,
        "tool": name,
        "kind": "bag",
        "values": pooled.astype(float),
        "n_obs": int(len(cache.obs)),
    }


def parse_result(result: dict[str, Any], board: Blackboard) -> None:
    board.history.append({"tool": result.get("tool"), "ok": result.get("ok")})
    if result.get("ok"):
        board.evidence.append(result)
