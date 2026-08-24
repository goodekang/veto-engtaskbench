from __future__ import annotations

from typing import Any

import numpy as np

from .data import TaskCache

CLEAR_WIDTH_MM = 850.0
EDGE_MM = 12.0
MASS_TOL = 0.02


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
    if expected == "QuantityTable":
        return "values" in record and record.get("unit") in {"mm", "m", "cm"}
    if expected == "Verdict":
        return "values" in record
    if expected == "SolidModel":
        return bool(record.get("watertight", False) or record.get("ok"))
    return bool(record.get("ok"))


def geometric_oracle(cache: TaskCache) -> dict[str, Any]:
    meta = cache.meta or {}
    edge = float(meta.get("min_edge_mm") or 0.0)
    n_obs = int(len(cache.obs))
    bbox_span = float(np.ptp(cache.obs[:, :3])) if cache.obs.size else 0.0
    watertight = n_obs >= 4
    mass_ok = 0.5 <= float(np.mean(np.abs(cache.obs[:, 0]))) <= 1.5
    edge_ok = edge >= EDGE_MM if edge else True
    return {
        "ok": bool(watertight and mass_ok and edge_ok),
        "watertight": watertight,
        "bbox_span": bbox_span,
        "mass_ok": mass_ok,
        "min_edge_mm": edge,
        "edge_ok": edge_ok,
    }


def verify_candidate(
    goal_type: str,
    result: dict[str, Any],
    cache: TaskCache | None = None,
    threshold_mm: float = CLEAR_WIDTH_MM,
) -> dict[str, Any]:
    checks = {
        "schema": schema_ok(result, ["ok", "tool"]),
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
        verdict = verify_doors(widths, threshold_mm)
        checks["constraint"] = verdict["n_fail"] <= 2
        result = {**result, **verdict}
    if result.get("kind") in {"solid", "constraints"} and cache is not None:
        geom = geometric_oracle(cache)
        checks["geometry"] = geom["ok"]
        result = {**result, **geom}
    passed = all(checks.values()) and bool(result.get("ok", False))
    return {"passed": passed, "checks": checks, "result": result}


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
