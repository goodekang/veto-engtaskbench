from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import load_config


@dataclass(frozen=True)
class RunSpec:
    method: str
    backbone: str
    domain: str
    run: int
    seed: int
    repair_budget: int
    replan_budget: int
    broker_k: int
    sandbox_timeout_s: int
    sandbox_memory_mb: int

    @property
    def run_id(self) -> str:
        return f"{self.method}_{self.backbone}_{self.domain}_r{self.run}"


def load_protocol(
    baseline_path: str | Path = "configs/baselines.yaml",
    veto_path: str | Path = "configs/veto.yaml",
) -> dict[str, Any]:
    cfg = load_config([veto_path], validate=True)
    baselines = load_config([baseline_path], validate=True)
    cfg["methods"] = baselines["methods"]
    cfg["backbones"] = baselines["backbones"]
    cfg["repeated_runs"] = baselines["repeated_runs"]
    cfg["matched_budget"] = baselines["matched_budget"]
    return cfg


def build_run_matrix(
    cfg: dict[str, Any],
    *,
    methods: Iterable[str] | None = None,
    backbones: Iterable[str] | None = None,
) -> list[RunSpec]:
    selected_methods = list(methods or cfg["methods"].keys())
    selected_backbones = list(backbones or cfg["backbones"].keys())
    unknown_methods = set(selected_methods) - set(cfg["methods"])
    unknown_backbones = set(selected_backbones) - set(cfg["backbones"])
    if unknown_methods or unknown_backbones:
        raise ValueError(
            f"unknown methods={sorted(unknown_methods)}, backbones={sorted(unknown_backbones)}"
        )
    repeated = set(cfg.get("repeated_runs", {}).get("methods", []))
    seeds = list(cfg.get("repeated_runs", {}).get("seeds", [cfg.get("seed", 42)]))
    matrix: list[RunSpec] = []
    for method in selected_methods:
        method_cfg = cfg["methods"][method]
        for backbone in selected_backbones:
            backbone_runs = int(cfg["backbones"][backbone].get("runs", 1))
            n_runs = max(backbone_runs, len(seeds) if method in repeated and backbone == "gpt-4o" else 1)
            for domain in method_cfg.get("domains", ["bim", "cad"]):
                for run in range(1, n_runs + 1):
                    matrix.append(
                        RunSpec(
                            method=method,
                            backbone=backbone,
                            domain=domain,
                            run=run,
                            seed=int(seeds[(run - 1) % len(seeds)]),
                            repair_budget=int(method_cfg.get("repair_budget", cfg.get("repair_budget", 0))),
                            replan_budget=int(method_cfg.get("replan_budget", cfg.get("replan_budget", 0))),
                            broker_k=int(method_cfg.get("broker_k", cfg.get("broker_k", 42))),
                            sandbox_timeout_s=int(cfg.get("sandbox_timeout_s", 120)),
                            sandbox_memory_mb=int(cfg.get("sandbox_memory_mb", 2048)),
                        )
                    )
    return matrix


def validate_matched_budget(cfg: dict[str, Any]) -> None:
    budget = cfg.get("matched_budget", {})
    if budget.get("reference_method") != "veto":
        raise ValueError("matched-budget comparisons must use VETO as the reference")
    if float(budget.get("tool_calls", 0)) <= 0 or float(budget.get("cost_usd", 0)) <= 0:
        raise ValueError("matched-budget limits must be positive")
