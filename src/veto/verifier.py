from __future__ import annotations

import operator
from typing import Any

import numpy as np

from .blackboard import Blackboard
from .data import TaskCache

CLEAR_WIDTH_MM = 850.0
EDGE_MM = 12.0
MASS_TOL = 0.02
BBOX_TOL = 0.02

_OPS = {
    "==": operator.eq,
    "!=": operator.ne,
    ">": operator.gt,
    ">=": operator.ge,
    "<": operator.lt,
    "<=": operator.le,
}


def canonicalize_width(value: float, unit: str) -> float:
    unit = (unit or "mm").strip().lower()
    if unit in {"m", "meter", "metre"}:
        return float(value) * 1000.0
    if unit in {"cm"}:
        return float(value) * 10.0
    return float(value)


def door_verdict(width_mm: float, threshold_mm: float = CLEAR_WIDTH_MM) -> bool:
    return float(width_mm) + 1e-6 >= threshold_mm


def verify_doors(
    widths_mm: np.ndarray, threshold_mm: float = CLEAR_WIDTH_MM
) -> dict[str, Any]:
    widths_mm = np.asarray(widths_mm, dtype=float)
    passed = widths_mm >= threshold_mm
    return {
        "n": int(len(widths_mm)),
        "n_pass": int(passed.sum()),
        "n_fail": int((~passed).sum()),
        "widths_mm": widths_mm,
        "passed": passed,
    }


def schema_ok(args: dict[str, Any], required: list[str]) -> bool:
    return all(k in args for k in required)


def unit_range_ok(value: float, lo: float, hi: float) -> bool:
    return lo <= float(value) <= hi


def evidence_complete(record: dict[str, Any], expected: str) -> bool:
    if expected == "ElementSet":
        ids = record.get("entity_ids")
        return isinstance(ids, list) and bool(ids) and all(isinstance(item, str) for item in ids)
    if expected == "QuantityTable":
        return (
            "values" in record
            and record.get("unit") in {"mm", "m", "cm", "m2", "mm2"}
            and np.asarray(record["values"]).size > 0
        )
    if expected == "Verdict":
        return "values" in record and "decision" in record
    if expected == "SolidModel":
        return bool(
            record.get("watertight", False)
            and record.get("geometry_hash")
            or record.get("ok") and record.get("geometry_hash")
        )
    return bool(record.get("ok"))


def geometric_oracle(cache: TaskCache) -> dict[str, Any]:
    meta = cache.meta or {}
    edge = float(meta.get("min_edge_mm") or 0.0)
    n_obs = int(len(cache.obs))
    bbox_span = float(np.ptp(cache.obs[:, :3])) if cache.obs.size else 0.0
    watertight = bool(meta.get("watertight", n_obs >= 4))
    self_intersections = int(meta.get("self_intersections", 0))
    solid_valid = bool(meta.get("solid_valid", watertight and self_intersections == 0))
    observed_mass = float(meta.get("mass", np.mean(np.abs(cache.obs[:, 0]))))
    target_mass = float(meta.get("target_mass", observed_mass))
    mass_error = abs(observed_mass - target_mass) / max(abs(target_mass), 1e-8)
    mass_ok = mass_error <= float(meta.get("mass_tolerance", MASS_TOL))
    bbox_error = float(meta.get("bbox_relative_error", 0.0))
    bbox_ok = bbox_error <= float(meta.get("bbox_tolerance", BBOX_TOL))
    wall = float(meta.get("minimum_wall_mm", EDGE_MM))
    wall_ok = wall >= float(meta.get("required_wall_mm", 0.0))
    edge_ok = edge >= EDGE_MM if edge else True
    return {
        "ok": bool(solid_valid and watertight and mass_ok and bbox_ok and edge_ok and wall_ok),
        "solid_valid": solid_valid,
        "watertight": watertight,
        "self_intersections": self_intersections,
        "bbox_span": bbox_span,
        "bbox_relative_error": bbox_error,
        "bbox_ok": bbox_ok,
        "mass_relative_error": mass_error,
        "mass_ok": mass_ok,
        "min_edge_mm": edge,
        "edge_ok": edge_ok,
        "minimum_wall_mm": wall,
        "wall_ok": wall_ok,
    }


def _constraint_threshold(
    board: Blackboard | None,
    name: str,
    default: float,
) -> tuple[str, float, str | None]:
    if board is None:
        return ">=", default, None
    for item in board.constraints:
        if hasattr(item, "name") and item.name == name:
            return str(item.operator), float(item.value), item.unit
        if isinstance(item, dict) and item.get("name") == name:
            return (
                str(item.get("operator", ">=")),
                float(item.get("value", default)),
                item.get("unit"),
            )
    return ">=", default, None


def evaluate_predicate(values: np.ndarray, op: str, threshold: float) -> np.ndarray:
    if op not in _OPS:
        raise ValueError(f"unsupported constraint operator: {op}")
    return np.asarray(_OPS[op](np.asarray(values, dtype=float), threshold), dtype=bool)


def _counter_evidence(checks: dict[str, bool], result: dict[str, Any]) -> dict[str, Any] | None:
    family = defect_family(checks)
    if family == "none":
        return None
    details: dict[str, Any] = {
        "family": family,
        "tool": result.get("tool"),
        "kind": result.get("kind"),
        "message": {
            "schema": "tool arguments or result schema did not match the typed signature",
            "unit": "one or more quantities had an unsupported unit or implausible range",
            "geometric": "independent geometry checks rejected the candidate",
            "constraint": "the candidate did not satisfy the inferred constraint schema",
            "evidence": "required provenance or typed evidence fields were missing",
        }[family],
    }
    if "values" in result:
        values = np.asarray(result["values"], dtype=float)
        details["observed"] = values[: min(len(values), 8)].tolist()
    for key in (
        "unit",
        "min_edge_mm",
        "bbox_relative_error",
        "mass_relative_error",
        "self_intersections",
    ):
        if key in result:
            details[key] = result[key]
    return details


def verify_candidate(
    goal_type: str,
    result: dict[str, Any],
    cache: TaskCache | None = None,
    threshold_mm: float = CLEAR_WIDTH_MM,
    board: Blackboard | None = None,
) -> dict[str, Any]:
    checks = {
        "schema": schema_ok(result, ["ok", "tool"]) and "error" not in result,
        "unit_range": True,
        "geometry": True,
        "constraint": True,
        "evidence": evidence_complete(result, goal_type),
    }
    if result.get("kind") == "widths":
        widths = np.asarray(result["values"], dtype=float)
        unit = str(result.get("unit", "mm"))
        widths = np.array([canonicalize_width(v, unit) for v in widths])
        checks["unit_range"] = bool(np.all((widths > 200) & (widths < 3000)))
        op, threshold, threshold_unit = _constraint_threshold(
            board, "clear_width", threshold_mm
        )
        threshold = canonicalize_width(threshold, threshold_unit or "mm")
        passed_items = evaluate_predicate(widths, op, threshold)
        verdict = {
            "n": int(len(widths)),
            "n_pass": int(passed_items.sum()),
            "n_fail": int((~passed_items).sum()),
            "widths_mm": widths,
            "passed": passed_items,
            "threshold_mm": threshold,
        }
        # A QuantityTable is valid even when it contains non-compliant doors.
        checks["constraint"] = True
        result = {**result, **verdict}
    if result.get("kind") == "verdict":
        values = np.asarray(result.get("values", []), dtype=float)
        unit = str(result.get("unit", "mm"))
        values = np.array([canonicalize_width(v, unit) for v in values])
        op, threshold, threshold_unit = _constraint_threshold(
            board, "clear_width", threshold_mm
        )
        threshold = canonicalize_width(threshold, threshold_unit or "mm")
        decisions = evaluate_predicate(values, op, threshold)
        result = {
            **result,
            "n": int(len(values)),
            "n_pass": int(decisions.sum()),
            "n_fail": int((~decisions).sum()),
            "passed_items": decisions,
            "threshold_mm": threshold,
        }
    if result.get("kind") == "episode_verdict":
        decisions = np.asarray(result.get("values", []), dtype=float) >= 0.5
        checks["constraint"] = bool(decisions.size and decisions.all())
        result = {
            **result,
            "n": int(decisions.size),
            "n_pass": int(decisions.sum()),
            "n_fail": int((~decisions).sum()),
        }
    if result.get("kind") in {"solid", "constraints"} and cache is not None:
        geom = geometric_oracle(cache)
        checks["geometry"] = geom["ok"]
        result = {**result, **geom}
        if result.get("kind") == "constraints":
            scores = np.asarray(result.get("values", []), dtype=float)
            checks["constraint"] = bool(scores.size and np.all(scores >= 0.5))
    passed = all(checks.values()) and bool(result.get("ok", False))
    return {
        "passed": passed,
        "checks": checks,
        "result": result,
        "counter_evidence": _counter_evidence(checks, result),
    }


def defect_family(checks: dict[str, bool]) -> str:
    order = [
        ("schema", "schema"),
        ("unit_range", "unit"),
        ("geometry", "geometric"),
        ("constraint", "constraint"),
        ("evidence", "evidence"),
    ]
    for key, name in order:
        if not checks.get(key, True):
            return name
    return "none"
