from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .blackboard import Blackboard, SubGoal
from .data import TaskCache
from .tools import validate_arguments
from .verifier import geometric_oracle

ELEMENT_TOOLS = {
    "ifc_entities_by_type",
    "ifc_type_assignments",
    "ifc_classification_refs",
    "ifc_element_summary",
    "spatial_containment_graph",
    "spatial_adjacency_graph",
    "spatial_egress_paths",
    "rel_contained_in_storey",
    "rel_space_boundaries",
    "rel_connectivity",
    "rel_group_membership",
    "rel_document_associations",
}
QUANTITY_TOOLS = {
    "ifc_property_values",
    "ifc_quantity_values",
    "ifc_material_layers",
    "ifc_unit_context",
    "spatial_clear_width",
    "spatial_travel_distance",
}
BIM_VERDICT_TOOLS = {
    "rule_predicate_eval",
    "validate_ifc_schema",
    "validate_unit_ranges",
    "validate_evidence_chain",
    "bim_report_render",
}
CAD_KERNEL_TOOLS = {
    "cq_new_document",
    "cq_sketch_profile",
    "cq_extrude",
    "cq_revolve",
    "cq_hole",
    "cq_pattern",
    "cq_fillet",
    "cq_chamfer",
    "cq_boolean",
    "cq_shell",
    "cq_transform",
}
CAD_VERDICT_TOOLS = {"solid_kernel_validate", "bbox_mass_validate"}
EXPORT_TOOLS = {"bim_evidence_export", "cad_step_export", "cad_evidence_export"}


def implemented_tool_names() -> set[str]:
    return (
        {"ifc_open_model"}
        | ELEMENT_TOOLS
        | QUANTITY_TOOLS
        | {"spatial_clearance_envelope", "spatial_opening_geometry"}
        | BIM_VERDICT_TOOLS
        | CAD_KERNEL_TOOLS
        | CAD_VERDICT_TOOLS
        | EXPORT_TOOLS
    )


def precheck_tool(
    tool: dict[str, Any],
    args: dict[str, Any],
    board: Blackboard | None = None,
) -> dict[str, Any]:
    errors = validate_arguments(tool, args)
    preconditions = set(tool.get("preconditions", []))
    if "model_open" in preconditions and not args.get("artefact"):
        errors.append("IFC artefact handle is required for lazy model opening")
    if board is not None:
        evidence_types = {item.get("evidence_type") for item in board.evidence}
        requirements = {
            "entities_selected": "ElementSet",
            "spaces_selected": "ElementSet",
            "opening_selected": "ElementSet",
            "QuantityTable": "QuantityTable",
            "evidence_available": "any",
            "solid_exists": "SolidModel",
            "shape_exists": "SolidModel",
            "seed_feature_exists": "SolidModel",
            "closed_profile": "SolidModel",
            "two_solids_exist": "SolidModel",
        }
        for requirement, expected in requirements.items():
            if requirement not in preconditions:
                continue
            available = bool(board.evidence) if expected == "any" else expected in evidence_types
            if not available:
                errors.append(f"precondition {requirement} is not satisfied")
        if "verified_evidence" in preconditions and not any(v.get("passed") for v in board.verdicts):
            errors.append("no verified evidence is available")
        if "solid_verified" in preconditions and not any(
            v.get("passed") and v.get("result", {}).get("kind") in {"constraints", "episode_verdict"}
            for v in board.verdicts
        ):
            errors.append("solid has not passed deterministic verification")
        if "geometry_available" in preconditions and not (
            "SolidModel" in evidence_types or args.get("artefact")
        ):
            errors.append("geometry is not available")
    return {"passed": not errors, "errors": errors}


def _audit_hash(payload: Any) -> str:
    def default(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.generic):
            return value.item()
        raise TypeError(type(value).__name__)

    raw = json.dumps(payload, sort_keys=True, default=default).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def _entity_ids(cache: TaskCache) -> list[str]:
    meta = cache.meta or {}
    ids = meta.get("entity_ids") or meta.get("guids")
    if ids:
        return [str(item) for item in ids]
    count = max(1, min(int(len(cache.obs)), 64))
    prefix = "IfcEntity" if cache.domain == "bim" else "Feature"
    return [f"{prefix}_{index:03d}" for index in range(count)]


def _quantity_values(cache: TaskCache, name: str, unit: str) -> np.ndarray:
    if name == "spatial_clear_width":
        metres = (
            cache.items[:, 0].astype(float)
            if cache.items is not None
            else 0.90 + 0.08 * np.tanh(cache.obs[:, 0].astype(float))
        )
        scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[unit]
        return metres * scale
    channel = np.abs(cache.obs[:, 0].astype(float))
    if name == "spatial_travel_distance":
        values_m = np.cumsum(np.maximum(channel, 0.05))
        scale = {"m": 1.0, "cm": 100.0, "mm": 1000.0}[unit]
        return values_m * scale
    return channel


def _episode_decision(cache: TaskCache, goal: SubGoal) -> bool:
    meta = cache.meta or {}
    if float(cache.y) < 0.5:
        return False
    recover_at = int(meta.get("recover_at", 0))
    return goal.attempts > recover_at


def _solid_payload(
    base: dict[str, Any],
    cache: TaskCache,
    *,
    export: bool = False,
) -> dict[str, Any]:
    geom = geometric_oracle(cache)
    payload = {**base, "ok": geom["ok"], "kind": "solid", **geom}
    payload["geometry_hash"] = _audit_hash(payload)
    if export:
        payload["format"] = "STEP"
        payload["export_path"] = f"outputs/{cache.task_id}.step"
        payload["content_hash"] = _audit_hash(
            {"task_id": cache.task_id, "geometry_hash": payload["geometry_hash"]}
        )
    return payload


def run_tool(
    tool: dict[str, Any],
    args: dict[str, Any],
    cache: TaskCache,
    goal: SubGoal,
    board: Blackboard | None = None,
) -> dict[str, Any]:
    name = tool.get("name", "")
    precheck = precheck_tool(tool, args, board)
    if not precheck["passed"]:
        return {
            "ok": False,
            "error": "precondition",
            "details": precheck["errors"],
            "tool": name,
        }
    category = tool.get("category", "")
    base = {
        "ok": True,
        "tool": name,
        "category": category,
        "task_id": cache.task_id,
        "evidence_type": goal.evidence_type,
    }
    if name == "ifc_open_model":
        meta = cache.meta or {}
        return {
            **base,
            "kind": "model",
            "schema": meta.get("ifc_schema", "IFC4"),
            "declared_units": meta.get("units", ["mm"]),
            "model_handle": f"cache://{cache.task_id}",
        }
    if name in ELEMENT_TOOLS:
        ids = _entity_ids(cache)
        return {
            **base,
            "kind": "elements",
            "entity_ids": ids,
            "cardinality": len(ids),
            "relation_edges": max(0, len(ids) - 1) if name.startswith(("rel_", "spatial_")) else 0,
            "observation_hash": _audit_hash(cache.obs),
        }
    if name in QUANTITY_TOOLS:
        unit = str(args.get("unit", "mm" if name == "spatial_clear_width" else "m"))
        if unit not in {"mm", "cm", "m"}:
            return {**base, "ok": False, "error": "unit", "details": [unit]}
        values = _quantity_values(cache, name, unit)
        return {
            **base,
            "kind": "widths" if name == "spatial_clear_width" else "quantities",
            "values": values,
            "unit": unit,
            "entity_ids": _entity_ids(cache)[: len(values)],
            "source": name,
        }
    if name in {"spatial_clearance_envelope", "spatial_opening_geometry"}:
        return _solid_payload(base, cache)
    if name in BIM_VERDICT_TOOLS:
        decision = _episode_decision(cache, goal)
        return {
            **base,
            "kind": "episode_verdict",
            "values": np.asarray([float(decision)], dtype=float),
            "decision": decision,
            "claim_sources": [item.get("source_call_id") for item in (board.evidence if board else [])],
        }
    if name in CAD_KERNEL_TOOLS:
        return _solid_payload(base, cache)
    if name in CAD_VERDICT_TOOLS:
        decision = _episode_decision(cache, goal)
        names = list((cache.meta or {}).get("constraints", []))
        n_constraints = len(names) or int((cache.meta or {}).get("n_gold", 1))
        return {
            **base,
            "kind": "constraints",
            "values": np.full(n_constraints, float(decision), dtype=float),
            "constraint_names": names or [f"constraint_{i + 1}" for i in range(n_constraints)],
            "decision": decision,
        }
    if name in EXPORT_TOOLS:
        if name == "bim_evidence_export":
            return {
                **base,
                "kind": "audit_bundle",
                "claim_sources": [item.get("source_call_id") for item in (board.evidence if board else [])],
                "content_hash": _audit_hash(board.record() if board else {"task_id": cache.task_id}),
            }
        return _solid_payload(base, cache, export=True)
    return {
        **base,
        "ok": False,
        "error": "unimplemented_tool",
        "details": [f"no runtime adapter registered for {name}"],
    }


def parse_result(
    result: dict[str, Any],
    board: Blackboard,
    *,
    goal: SubGoal,
    args: dict[str, Any],
    attempt: int,
) -> int:
    return board.append_history(
        goal=goal.name,
        tool=str(result.get("tool", "")),
        args=args,
        result=result,
        attempt=attempt,
    )
