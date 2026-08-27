from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .blackboard import Blackboard
from .data import TaskCache
from .executor import parse_result, precheck_tool, run_tool
from .planner import plan_for, topo_order
from .sandbox import SandboxLimits, execute_sandboxed
from .verifier import defect_family, verify_candidate


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
    sandboxed: bool = True
    sandbox_limits: SandboxLimits = field(default_factory=SandboxLimits)
    trace: list[dict[str, Any]] = field(default_factory=list)

    def _select(self, select_fn, board, goal, registry, excluded):
        try:
            return select_fn(board, goal, registry, exclude=excluded)
        except TypeError:
            return select_fn(board, goal, registry)

    @staticmethod
    def _report(board: Blackboard, cache: TaskCache) -> dict[str, Any]:
        return {
            "task_id": cache.task_id,
            "domain": cache.domain,
            "status": "ok",
            "claims": [
                {
                    "goal": item.get("goal"),
                    "source_call_id": item.get("source_call_id"),
                    "tool": item.get("tool"),
                    "kind": item.get("kind"),
                }
                for item in board.evidence
            ],
            "evidence_count": len(board.evidence),
            "verifier_passes": sum(bool(v.get("passed")) for v in board.verdicts),
        }

    def run(
        self,
        board: Blackboard,
        cache: TaskCache,
        registry: list[dict[str, Any]],
        select_fn,
    ) -> dict[str, Any]:
        self.trace.clear()
        repairs_total = 0
        committed_names: set[str] = set()
        attempts_by_goal: dict[str, int] = {}
        for r in range(self.r_max + 1):
            board.emit(State.PLAN.value, replan=r)
            plan_for(board, tier=cache.tier, domain=cache.domain)
            for planned in board.plan:
                if planned.name in committed_names:
                    planned.status = "committed"
            for goal in topo_order(board.plan):
                if goal.name in committed_names:
                    continue
                done = False
                excluded: set[str] = set()
                for k in range(self.k_max + 1):
                    attempts_by_goal[goal.name] = attempts_by_goal.get(goal.name, 0) + 1
                    goal.attempts = attempts_by_goal[goal.name]
                    board.emit(State.SELECT.value, goal=goal.name, repair=k, replan=r)
                    tool, args, shortlist = self._select(
                        select_fn, board, goal, registry, excluded
                    )
                    board.emit(
                        State.EXECUTE.value,
                        goal=goal.name,
                        tool=tool.get("name"),
                        shortlist=[item.get("name") for item in shortlist],
                    )
                    precheck = precheck_tool(tool, args, board)
                    if not precheck["passed"]:
                        result = {
                            "ok": False,
                            "tool": tool.get("name"),
                            "error": "precondition",
                            "details": precheck["errors"],
                        }
                    elif self.sandboxed:
                        result = execute_sandboxed(
                            tool,
                            args,
                            cache,
                            goal,
                            self.sandbox_limits,
                        )
                    else:
                        result = run_tool(tool, args, cache, goal, board)
                    call_id = parse_result(
                        result,
                        board,
                        goal=goal,
                        args=args,
                        attempt=k,
                    )
                    board.emit(State.VERIFY.value, goal=goal.name, call_id=call_id)
                    verdict = verify_candidate(
                        goal.evidence_type,
                        result,
                        cache,
                        board=board,
                    )
                    board.add_verdict(verdict, call_id=call_id)
                    family = defect_family(verdict["checks"])
                    event = {
                        "goal": goal.name,
                        "tool": tool.get("name"),
                        "call_id": call_id,
                        "repair": k,
                        "replan": r,
                        "passed": verdict["passed"],
                        "family": family,
                    }
                    self.trace.append(event)
                    if verdict["passed"]:
                        board.commit(
                            verdict["result"],
                            goal=goal,
                            call_id=call_id,
                        )
                        board.emit(State.COMMIT.value, **event)
                        committed_names.add(goal.name)
                        done = True
                        break
                    if family in {"schema", "evidence"}:
                        excluded.add(str(tool.get("name")))
                    if k < self.k_max:
                        repairs_total += 1
                        goal.status = "repair"
                        board.emit(
                            State.REPAIR.value,
                            **event,
                            counter_evidence=verdict.get("counter_evidence"),
                        )
                if not done:
                    board.emit(State.REPLAN.value, goal=goal.name, replan=r)
                    break
            if all(goal.name in committed_names for goal in board.plan):
                report = self._report(board, cache)
                return {
                    **report,
                    "repairs": repairs_total,
                    "replans": r,
                    "trace": list(self.trace),
                }
        board.emit(State.ESCALATE.value, reason="recovery_budget_exhausted")
        return {
            "task_id": cache.task_id,
            "domain": cache.domain,
            "status": "escalate",
            "human_referral": True,
            "repairs": repairs_total,
            "replans": self.r_max,
            "evidence": len(board.evidence),
            "counter_evidence": list(board.counter_evidence),
            "trace": list(self.trace),
        }
