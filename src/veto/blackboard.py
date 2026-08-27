from __future__ import annotations

import copy
import time
from dataclasses import dataclass, field
from typing import Any


EVIDENCE_TYPES = ("ElementSet", "QuantityTable", "SolidModel", "Verdict")


@dataclass
class SubGoal:
    name: str
    evidence_type: str
    acceptance: str
    depends_on: tuple[str, ...] = ()
    status: str = "pending"
    attempts: int = 0
    slots: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unknown evidence type {self.evidence_type}")
        if self.name in self.depends_on:
            raise ValueError(f"sub-goal {self.name!r} cannot depend on itself")

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "evidence_type": self.evidence_type,
            "acceptance": self.acceptance,
            "depends_on": list(self.depends_on),
            "status": self.status,
            "attempts": self.attempts,
            "slots": copy.deepcopy(self.slots),
        }


@dataclass(frozen=True)
class Constraint:
    name: str
    operator: str
    value: Any
    unit: str | None = None
    applicability: str | None = None
    source: str = "planner"

    def as_record(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "operator": self.operator,
            "value": self.value,
            "unit": self.unit,
            "applicability": self.applicability,
            "source": self.source,
        }


@dataclass
class Blackboard:
    query: str
    artefact: str
    constraints: list[Constraint | dict[str, Any] | str] = field(default_factory=list)
    plan: list[SubGoal] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)
    counter_evidence: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    context: dict[str, Any] = field(default_factory=dict)

    def _constraint_records(self) -> list[Any]:
        return [
            item.as_record() if isinstance(item, Constraint) else copy.deepcopy(item)
            for item in self.constraints
        ]

    def record(self) -> dict[str, Any]:
        return {
            "q": self.query,
            "A": self.artefact,
            "C": self._constraint_records(),
            "P": [g.as_record() for g in self.plan],
            "H": copy.deepcopy(self.history),
            "E": copy.deepcopy(self.evidence),
            "V": copy.deepcopy(self.verdicts),
            "counter_evidence": copy.deepcopy(self.counter_evidence),
            "events": copy.deepcopy(self.events),
        }

    def projection(self, role: str) -> dict[str, Any]:
        if role == "planner":
            return {
                "query": self.query,
                "artefact": self.artefact,
                "constraints": self._constraint_records(),
                "counter_evidence": copy.deepcopy(self.counter_evidence[-3:]),
            }
        if role == "broker":
            return {
                "artefact": self.artefact,
                "plan": [g.as_record() for g in self.plan],
                "evidence": copy.deepcopy(self.evidence),
                "counter_evidence": copy.deepcopy(self.counter_evidence[-3:]),
            }
        if role == "executor":
            return {
                "artefact": self.artefact,
                "history": copy.deepcopy(self.history),
                "plan": [g.as_record() for g in self.plan],
                "counter_evidence": copy.deepcopy(self.counter_evidence[-3:]),
            }
        if role == "verifier":
            return {
                "history": copy.deepcopy(self.history),
                "evidence": copy.deepcopy(self.evidence),
                "constraints": self._constraint_records(),
            }
        return {"query": self.query}

    def append_history(
        self,
        *,
        goal: str,
        tool: str,
        args: dict[str, Any],
        result: dict[str, Any],
        attempt: int,
    ) -> int:
        call_id = len(self.history) + 1
        self.history.append(
            {
                "call_id": call_id,
                "goal": goal,
                "tool": tool,
                "args": copy.deepcopy(args),
                "result": copy.deepcopy(result),
                "attempt": attempt,
                "timestamp": time.time(),
            }
        )
        return call_id

    def add_verdict(self, verdict: dict[str, Any], *, call_id: int | None = None) -> None:
        item = copy.deepcopy(verdict)
        if call_id is not None:
            item["call_id"] = call_id
        self.verdicts.append(item)
        if not item.get("passed", False):
            counter = item.get("counter_evidence")
            if counter:
                self.counter_evidence.append(copy.deepcopy(counter))

    def commit(
        self,
        record: dict[str, Any],
        *,
        goal: SubGoal | None = None,
        call_id: int | None = None,
    ) -> None:
        item = copy.deepcopy(record)
        if call_id is not None:
            item["source_call_id"] = call_id
        if goal is not None:
            item["goal"] = goal.name
            item["evidence_type"] = goal.evidence_type
            goal.status = "committed"
            goal.slots["evidence_index"] = len(self.evidence)
        self.evidence.append(item)

    def pending(self) -> list[SubGoal]:
        return [g for g in self.plan if g.status != "committed"]

    def ready(self) -> list[SubGoal]:
        committed = {g.name for g in self.plan if g.status == "committed"}
        return [
            g
            for g in self.plan
            if g.status != "committed" and set(g.depends_on).issubset(committed)
        ]

    def emit(self, state: str, **payload: Any) -> None:
        self.events.append({"state": state, "timestamp": time.time(), **copy.deepcopy(payload)})
