from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


_TOKEN = re.compile(r"[a-z0-9]+")


def load_registry(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as f:
        return json.load(f)


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
    if name.startswith(("ifc_", "bim_io")):
        return "parse"
    if name.startswith("spatial_"):
        return "spatial"
    if name.startswith("rel_"):
        return "relational"
    if name.startswith("cq_"):
        return "cad_kernel"
    if "val" in name:
        return "validation"
    return "io"
