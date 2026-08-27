from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")
REQUIRED_TOOL_FIELDS = {
    "name",
    "domain",
    "category",
    "doc",
    "args",
    "preconditions",
    "postconditions",
}
EXPECTED_COUNTS = {"bim": 27, "cad": 15}


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        registry = json.load(f)
    validate_registry(registry)
    return registry


def validate_registry(registry: list[dict[str, Any]]) -> None:
    if not isinstance(registry, list):
        raise TypeError("tool registry must be a list")
    names: set[str] = set()
    counts = {"bim": 0, "cad": 0}
    for index, tool in enumerate(registry):
        missing = REQUIRED_TOOL_FIELDS - set(tool)
        if missing:
            raise ValueError(f"tool #{index} is missing fields: {sorted(missing)}")
        name = str(tool["name"])
        if name in names:
            raise ValueError(f"duplicate tool name: {name}")
        names.add(name)
        domain = str(tool["domain"])
        if domain not in counts:
            raise ValueError(f"tool {name} has unsupported domain {domain!r}")
        counts[domain] += 1
        if not isinstance(tool["args"], dict):
            raise TypeError(f"tool {name} args must be a mapping")
        if not isinstance(tool["preconditions"], list) or not isinstance(tool["postconditions"], list):
            raise TypeError(f"tool {name} preconditions/postconditions must be lists")
    if len(registry) != 42 or counts != EXPECTED_COUNTS:
        raise ValueError(
            f"expected 42 tools split BIM=27/CAD=15, got total={len(registry)} {counts}"
        )


def tokenize(text: str) -> set[str]:
    return set(_TOKEN.findall(text.lower()))


def token_overlap(query: str, doc: str) -> float:
    q = tokenize(query)
    d = tokenize(doc)
    if not q or not d:
        return 0.0
    return len(q & d) / len(q | d)


def shortlist(registry: list[dict[str, Any]], k: int, domain: str | None = None) -> list[dict[str, Any]]:
    items = registry
    if domain:
        items = [t for t in registry if t.get("domain") in {domain, "shared"}]
    return items[:k]


def by_name(registry: list[dict[str, Any]], name: str) -> dict[str, Any] | None:
    for tool in registry:
        if tool.get("name") == name:
            return tool
    return None


def category_of(name: str) -> str:
    if name.startswith(("ifc_", "bim_")):
        return "parse"
    if name.startswith(("spatial_", "geom_")):
        return "spatial"
    if name.startswith("rel_"):
        return "relational"
    if name.startswith("cq_"):
        return "cad_kernel"
    if name.startswith(("validate_", "rule_", "solid_", "bbox_", "mass_")):
        return "validation"
    return "io"


def validate_arguments(tool: dict[str, Any], args: dict[str, Any]) -> list[str]:
    errors: list[str] = []
    schema = tool.get("args", {})
    for key, spec in schema.items():
        spec = spec if isinstance(spec, dict) else {"type": str(spec)}
        if spec.get("required", False) and key not in args:
            errors.append(f"missing required argument {key}")
            continue
        if key not in args:
            continue
        value = args[key]
        expected = spec.get("type")
        python_types = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": (list, tuple),
            "object": dict,
        }
        if expected in python_types and not isinstance(value, python_types[expected]):
            errors.append(f"{key} must be {expected}")
        if "enum" in spec and value not in spec["enum"]:
            errors.append(f"{key} must be one of {spec['enum']}")
        if isinstance(value, (int, float)):
            if "minimum" in spec and value < spec["minimum"]:
                errors.append(f"{key} is below minimum {spec['minimum']}")
            if "maximum" in spec and value > spec["maximum"]:
                errors.append(f"{key} exceeds maximum {spec['maximum']}")
    return errors
