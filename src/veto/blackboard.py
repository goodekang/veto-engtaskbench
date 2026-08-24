from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


EVIDENCE_TYPES = ("ElementSet", "QuantityTable", "SolidModel", "Verdict")


@dataclass
class SubGoal:
    name: str
    evidence_type: str
    acceptance: str
    status: str = "pending"
    slots: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.evidence_type not in EVIDENCE_TYPES:
            raise ValueError(f"unknown evidence type {self.evidence_type}")


@dataclass
class Blackboard:
    query: str
    artefact: str
    constraints: list[str] = field(default_factory=list)
    plan: list[SubGoal] = field(default_factory=list)
    history: list[dict[str, Any]] = field(default_factory=list)
    evidence: list[dict[str, Any]] = field(default_factory=list)
    verdicts: list[dict[str, Any]] = field(default_factory=list)

    def record(self) -> dict[str, Any]:
        return {
            "q": self.query,
            "C": list(self.constraints),
            "P": [g.name for g in self.plan],
            "H": list(self.history),
            "E": list(self.evidence),
            "V": list(self.verdicts),
        }

    def projection(self, role: str) -> dict[str, Any]:
        if role == "planner":
            return {"query": self.query, "artefact": self.artefact, "constraints": self.constraints}
        if role == "broker":
            return {"plan": [g.__dict__ for g in self.plan], "evidence": self.evidence}
        if role == "executor":
            return {"history": self.history, "plan": [g.__dict__ for g in self.plan]}
        if role == "verifier":
            return {"evidence": self.evidence, "constraints": self.constraints}
        return {"query": self.query}

    def commit(self, record: dict[str, Any]) -> None:
        self.evidence.append(record)
        self.verdicts.append({"ok": True, "record": record.get("id") or record.get("tool")})

    def pending(self) -> list[SubGoal]:
        return [g for g in self.plan if g.status != "committed"]
