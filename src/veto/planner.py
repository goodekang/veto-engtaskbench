from __future__ import annotations

from .blackboard import Blackboard, SubGoal


BIM_CLEARWIDTH = [
    ("retrieve_doors", "ElementSet", "IfcDoor candidates on egress routes"),
    ("filter_egress", "ElementSet", "doors on required exits"),
    ("measure_clear_width", "QuantityTable", "clear width with declared unit"),
    ("compare_threshold", "Verdict", "pass if width >= declared threshold"),
]

CAD_BRACKET = [
    ("sketch_plate", "SolidModel", "base plate outline"),
    ("holes", "SolidModel", "four-bolt circular pattern"),
    ("validate", "Verdict", "nine gold constraints"),
    ("export_step", "SolidModel", "STEP with audit hash"),
]

TIER_PLANS = {
    "T1": [("fetch_attr", "QuantityTable", "single-entity property"), ("compare", "Verdict", "threshold with unit")],
    "T2": [("join", "ElementSet", "entity-property join"), ("compare", "Verdict", "relational predicate")],
    "T3": [("topo", "ElementSet", "containment or path"), ("compare", "Verdict", "path exists")],
    "T4": BIM_CLEARWIDTH,
    "C1": [("sketch", "SolidModel", "short feature sequence"), ("validate", "Verdict", "solid + bbox")],
    "C2": [("features", "SolidModel", "6-10 operations"), ("validate", "Verdict", "feature counts")],
    "C3": CAD_BRACKET,
    "C4": [("long_seq", "SolidModel", ">15 operations"), ("validate", "Verdict", "all gold constraints")],
}


def infer_schema(query: str, domain: str) -> list[str]:
    q = query.lower()
    if domain == "bim":
        schema = ["unit=mm"]
        if "width" in q or "clear" in q:
            schema.append("clear_width>=850mm")
        if "egress" in q or "door" in q:
            schema.append("applicability=egress_doors")
        return schema or ["clause_predicate"]
    schema = ["solid_valid", "bbox_tol=0.02", "mass_tol=0.02"]
    if "bolt" in q or "hole" in q:
        schema.append("edge_distance>=12mm")
    return schema


def plan_for(board: Blackboard, tier: str | None = None, domain: str = "bim") -> list[SubGoal]:
    board.constraints = infer_schema(board.query, domain)
    key = tier or ("T4" if domain == "bim" else "C3")
    spec = TIER_PLANS.get(key, TIER_PLANS["T1"])
    goals = [SubGoal(name, ev, acc) for name, ev, acc in spec]
    board.plan = goals
    return goals


def topo_order(plan: list[SubGoal]) -> list[SubGoal]:
    return list(plan)
