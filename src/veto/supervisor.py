from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .blackboard import Blackboard
from .data import TaskCache
from .executor import parse_result, run_tool
from .planner import plan_for, topo_order
from .verifier import verify_candidate


class State(str, Enum):
    PLAN = "PLAN"
    SELECT = "SELECT"
    EXECUTE = "EXECUTE"
    VERIFY = "VERIFY"
    COMMIT = "COMMIT"
    REPAIR = "REPAIR"
    REPLAN = "REPLAN"
    ESCALATE = "ESCALATE"


def next_state(current: State, passed: bool, k: int, r: int, k_max: int = 3, r_max: int = 2) -> State:
    if current == State.VERIFY:
        if passed:
            return State.COMMIT
        if k < k_max:
            return State.REPAIR
        if r < r_max:
            return State.REPLAN
        return State.ESCALATE
    order = {
        State.PLAN: State.SELECT,
        State.SELECT: State.EXECUTE,
        State.EXECUTE: State.VERIFY,
        State.REPAIR: State.SELECT,
        State.REPLAN: State.PLAN,
        State.COMMIT: State.SELECT,
    }
    return order.get(current, State.ESCALATE)


@dataclass
class Supervisor:
    k_max: int = 3
    r_max: int = 2
    trace: list[dict[str, Any]] = field(default_factory=list)

    def run(
        self,
        board: Blackboard,
        cache: TaskCache,
        registry: list[dict[str, Any]],
        select_fn,
    ) -> dict[str, Any]:
        r = 0
        while r <= self.r_max:
            plan_for(board, tier=cache.tier, domain=cache.domain)
            committed = 0
            for goal in topo_order(board.plan):
                k = 0
                done = False
                while k <= self.k_max:
                    tool, args, short = select_fn(board, goal, registry)
                    result = run_tool(tool, args, cache, goal)
                    verdict = verify_candidate(goal.evidence_type, result, cache)
                    self.trace.append(
                        {
                            "goal": goal.name,
                            "tool": tool.get("name"),
                            "k": k,
                            "r": r,
                            "passed": verdict["passed"],
                            "family": next(iter(verdict["checks"])),
                        }
                    )
                    if verdict["passed"]:
                        parse_result(verdict["result"], board)
                        goal.status = "committed"
                        committed += 1
                        done = True
                        break
                    k += 1
                if not done:
                    break
            if committed == len(board.plan):
                return {"status": "ok", "repairs": r, "evidence": len(board.evidence)}
            r += 1
        return {"status": "escalate", "repairs": r, "evidence": len(board.evidence)}
