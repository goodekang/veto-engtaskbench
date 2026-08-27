from __future__ import annotations

import re
from collections import deque

from .blackboard import Blackboard, Constraint, SubGoal


BIM_CLEARWIDTH = [
    ("retrieve_doors", "ElementSet", "IfcDoor candidates with stable GlobalIds", ()),
    ("filter_egress", "ElementSet", "doors applicable to required egress routes", ("retrieve_doors",)),
    ("measure_clear_width", "QuantityTable", "clear widths with source, value, and declared unit", ("filter_egress",)),
    ("compare_threshold", "Verdict", "per-door decision under the inferred threshold", ("measure_clear_width",)),
]

CAD_BRACKET = [
    ("sketch_plate", "SolidModel", "fully constrained base-plate sketch", ()),
    ("build_body", "SolidModel", "watertight base solid with requested thickness", ("sketch_plate",)),
    ("holes", "SolidModel", "four-hole parametric pattern with minimum edge distance", ("build_body",)),
    ("validate", "Verdict", "all inferred geometric and parametric constraints", ("holes",)),
    ("export_step", "SolidModel", "STEP export with geometry and audit hashes", ("validate",)),
]

TIER_PLANS = {
    "T1": [
        ("fetch_attr", "QuantityTable", "single-entity property with provenance and unit", ()),
        ("compare", "Verdict", "typed threshold comparison", ("fetch_attr",)),
    ],
    "T2": [
        ("select_entities", "ElementSet", "entities matching applicability conditions", ()),
        ("join", "QuantityTable", "entity-property or entity-relation join", ("select_entities",)),
        ("compare", "Verdict", "relational predicate for every applicable entity", ("join",)),
    ],
    "T3": [
        ("build_relation_graph", "ElementSet", "containment, adjacency, and connectivity graph", ()),
        ("topology_query", "ElementSet", "path or topology result with traversed identifiers", ("build_relation_graph",)),
        ("compare", "Verdict", "topological constraint decision", ("topology_query",)),
    ],
    "T4": BIM_CLEARWIDTH,
    "C1": [
        ("sketch", "SolidModel", "fully constrained primitive sketch", ()),
        ("feature", "SolidModel", "short parametric feature sequence", ("sketch",)),
        ("validate", "Verdict", "solid validity, bounding box, and required dimensions", ("feature",)),
        ("export_step", "SolidModel", "STEP export with audit hash", ("validate",)),
    ],
    "C2": [
        ("sketch", "SolidModel", "constrained profile set", ()),
        ("features", "SolidModel", "six-to-ten ordered parametric operations", ("sketch",)),
        ("validate", "Verdict", "solid, tolerance, and feature-count checks", ("features",)),
        ("export_step", "SolidModel", "STEP export with audit hash", ("validate",)),
    ],
    "C3": CAD_BRACKET,
    "C4": [
        ("decompose", "SolidModel", "feature dependency plan for more than fifteen operations", ()),
        ("long_seq", "SolidModel", "ordered parametric construction with intermediate validity checks", ("decompose",)),
        ("validate", "Verdict", "all inferred geometry and relation constraints", ("long_seq",)),
        ("export_step", "SolidModel", "STEP export with geometry and audit hashes", ("validate",)),
    ],
}


_NUMBER_UNIT = re.compile(
    r"(?P<op>>=|<=|>|<|at\s+least|at\s+most|minimum|maximum)?\s*"
    r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>mm|cm|m|degrees?|deg|%)?",
    re.IGNORECASE,
)

BIM_PROVISION_RULES: dict[str, list[tuple[str, str, object, str | None, str]]] = {
    "egress_clear_width": [("clear_width", ">=", 850.0, "mm", "egress_doors")],
    "egress_travel": [("travel_distance", "<=", 30.0, "m", "egress_paths")],
    "door_swing": [("door_swing_direction", "==", "egress", None, "exit_doors")],
    "corridor_width": [("corridor_width", ">=", 1200.0, "mm", "egress_corridors")],
    "stair_riser": [("riser_height", "<=", 190.0, "mm", "stairs")],
    "ramp_slope": [("ramp_slope", "<=", 8.33, "%", "accessible_ramps")],
    "accessible_route": [("continuous_accessible_route", "==", True, None, "public_spaces")],
    "wc_clearance": [("wc_clearance_diameter", ">=", 1500.0, "mm", "accessible_wc")],
    "turning_circle": [("turning_diameter", ">=", 1500.0, "mm", "accessible_spaces")],
    "handrail_height": [("handrail_height", ">=", 900.0, "mm", "stairs_and_ramps")],
    "landing_depth": [("landing_depth", ">=", 1200.0, "mm", "stairs")],
    "headroom": [("clear_headroom", ">=", 2000.0, "mm", "circulation_paths")],
    "room_area": [("net_room_area", ">=", 9.0, "m2", "occupied_rooms")],
    "ceiling_height": [("ceiling_height", ">=", 2400.0, "mm", "occupied_rooms")],
    "window_sill": [("window_sill_height", "<=", 1100.0, "mm", "habitable_rooms")],
    "fire_compartment": [("fire_compartment_closed", "==", True, None, "fire_zones")],
    "fire_door_rating": [("fire_resistance", ">=", 60.0, "min", "fire_doors")],
    "exit_signage": [("exit_sign_visible", "==", True, None, "egress_routes")],
    "dead_end": [("dead_end_length", "<=", 15.0, "m", "corridors")],
    "occupancy_load": [("occupancy_within_capacity", "==", True, None, "occupied_spaces")],
    "opening_lintel": [("lintel_present", "==", True, None, "structural_openings")],
    "guard_height": [("guard_height", ">=", 1100.0, "mm", "fall_edges")],
    "stair_width": [("stair_clear_width", ">=", 1000.0, "mm", "egress_stairs")],
    "refuge_area": [("refuge_area", ">=", 1.5, "m2", "accessible_refuges")],
}


def _provision_key(query: str) -> str | None:
    normalised = re.sub(r"[^a-z0-9]+", "_", query.lower()).strip("_")
    return next((key for key in BIM_PROVISION_RULES if key in normalised), None)


def _cad_constraints(query: str) -> list[Constraint]:
    schema = [
        Constraint("solid_valid", "==", True),
        Constraint("watertight", "==", True),
        Constraint("bbox_relative_error", "<=", 0.02),
        Constraint("mass_relative_error", "<=", 0.02),
    ]
    q = query.lower()
    count_match = re.search(r"(\d+)[-\s]*(?:bolt|hole)", q)
    if count_match:
        schema.append(Constraint("hole_count", "==", int(count_match.group(1))))
    elif "bolt" in q or "hole" in q:
        schema.append(Constraint("hole_count", "==", 4))
    for pattern, name in (
        (r"(?:thickness|thick)\D{0,8}(\d+(?:\.\d+)?)\s*mm", "thickness"),
        (r"(?:radius|fillet)\D{0,8}(\d+(?:\.\d+)?)\s*mm", "radius"),
        (r"(?:edge distance|edge)\D{0,8}(\d+(?:\.\d+)?)\s*mm", "minimum_edge_distance"),
    ):
        match = re.search(pattern, q)
        if match:
            schema.append(
                Constraint(
                    name,
                    ">=" if name == "minimum_edge_distance" else "==",
                    float(match.group(1)),
                    "mm",
                )
            )
    return schema


def _threshold(query: str, default: float, default_unit: str) -> tuple[str, float, str]:
    matches = list(_NUMBER_UNIT.finditer(query))
    if not matches:
        return ">=", default, default_unit
    match = matches[-1]
    value = float(match.group("value"))
    unit = (match.group("unit") or default_unit).lower()
    raw_op = (match.group("op") or ">=").lower()
    if raw_op in {"at least", "minimum"}:
        raw_op = ">="
    elif raw_op in {"at most", "maximum"}:
        raw_op = "<="
    return raw_op, value, unit


def infer_schema(query: str, domain: str) -> list[Constraint]:
    q = query.lower()
    if domain == "bim":
        provision = _provision_key(query)
        if provision:
            return [
                Constraint(name, op, value, unit, applicability)
                for name, op, value, unit, applicability in BIM_PROVISION_RULES[provision]
            ]
        schema: list[Constraint] = []
        if "width" in q or "clear" in q:
            op, value, unit = _threshold(query, 850.0, "mm")
            schema.append(
                Constraint(
                    "clear_width",
                    op,
                    value,
                    unit,
                    applicability="egress_doors" if "egress" in q else "selected_doors",
                )
            )
        elif "travel" in q or "distance" in q:
            op, value, unit = _threshold(query, 30.0, "m")
            schema.append(Constraint("travel_distance", op, value, unit, "egress_paths"))
        elif "area" in q:
            op, value, unit = _threshold(query, 1.5, "m")
            schema.append(Constraint("area", op, value, f"{unit}2", "selected_spaces"))
        else:
            schema.append(Constraint("clause_predicate", "==", True, applicability="selected_entities"))
        if "egress" in q or "door" in q:
            schema.append(Constraint("applicability", "==", "egress_doors"))
        return schema
    schema = _cad_constraints(query)
    if ("bolt" in q or "hole" in q) and not any(
        constraint.name == "minimum_edge_distance" for constraint in schema
    ):
        op, value, unit = _threshold(query, 12.0, "mm")
        schema.append(Constraint("minimum_edge_distance", op, value, unit))
    return schema


def plan_for(board: Blackboard, tier: str | None = None, domain: str = "bim") -> list[SubGoal]:
    board.constraints = infer_schema(board.query, domain)
    key = tier or ("T4" if domain == "bim" else "C3")
    if key not in TIER_PLANS:
        raise ValueError(f"unsupported EngTaskBench tier: {key}")
    spec = TIER_PLANS[key]
    goals = [SubGoal(name, ev, acc, tuple(deps)) for name, ev, acc, deps in spec]
    if board.counter_evidence:
        latest = board.counter_evidence[-1]
        family = str(latest.get("family", "unknown"))
        for goal in goals:
            goal.slots["revision"] = len(board.counter_evidence)
            goal.slots["repair_focus"] = family
            if family in {"unit", "geometric", "constraint"}:
                goal.acceptance += f"; resolve prior {family} counter-evidence"
    board.plan = goals
    board.context.update({"domain": domain, "tier": key})
    board.emit("PLAN", tier=key, n_goals=len(goals))
    return goals


def topo_order(plan: list[SubGoal]) -> list[SubGoal]:
    by_name = {goal.name: goal for goal in plan}
    if len(by_name) != len(plan):
        raise ValueError("plan contains duplicate sub-goal names")
    unknown = {
        dep
        for goal in plan
        for dep in goal.depends_on
        if dep not in by_name
    }
    if unknown:
        raise ValueError(f"plan references unknown dependencies: {sorted(unknown)}")
    indegree = {goal.name: len(goal.depends_on) for goal in plan}
    children: dict[str, list[str]] = {goal.name: [] for goal in plan}
    for goal in plan:
        for dep in goal.depends_on:
            children[dep].append(goal.name)
    queue = deque(goal.name for goal in plan if indegree[goal.name] == 0)
    ordered: list[SubGoal] = []
    while queue:
        name = queue.popleft()
        ordered.append(by_name[name])
        for child in children[name]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if len(ordered) != len(plan):
        raise ValueError("plan contains a dependency cycle")
    return ordered
