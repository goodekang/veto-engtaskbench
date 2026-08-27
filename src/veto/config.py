from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable

import yaml


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "configs" / "default.yaml").exists():
            return p
    return Path.cwd()


def _read_yaml(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"configuration file does not exist: {path}")
    with path.open(encoding="utf-8") as f:
        value = yaml.safe_load(f) or {}
    if not isinstance(value, dict):
        raise TypeError(f"top-level configuration must be a mapping: {path}")
    return value


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge mappings while replacing scalar and list values."""
    out = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_merge(out[key], value)
        else:
            out[key] = deepcopy(value)
    return out


def _normalise_paths(paths: str | Path | Iterable[str | Path] | None) -> list[Path]:
    if paths is None:
        return []
    if isinstance(paths, (str, Path)):
        paths = [paths]
    root = repo_root()
    resolved = []
    for path in paths:
        candidate = Path(path)
        if not candidate.is_absolute():
            candidate = root / candidate
        resolved.append(candidate)
    return resolved


def validate_config(cfg: dict[str, Any]) -> None:
    required = ("seed", "d_obs", "d_model", "n_layers", "n_tools", "main_run")
    missing = [key for key in required if key not in cfg]
    if missing:
        raise ValueError(f"missing required configuration keys: {', '.join(missing)}")
    positive = ("d_obs", "d_model", "n_layers", "n_tools", "batch_size")
    for key in positive:
        if key in cfg and int(cfg[key]) <= 0:
            raise ValueError(f"{key} must be positive")
    for key in ("repair_budget", "replan_budget", "broker_k"):
        if key in cfg and int(cfg[key]) < 0:
            raise ValueError(f"{key} must be non-negative")
    if int(cfg["n_tools"]) != 42:
        raise ValueError("EngTaskBench-220 exposes exactly 42 agent-callable tools")
    if int(cfg.get("broker_k", 15)) > int(cfg["n_tools"]):
        raise ValueError("broker_k cannot exceed n_tools")


def load_config(
    path: str | Path | Iterable[str | Path] | None = None,
    *,
    validate: bool = True,
) -> dict[str, Any]:
    """Compose ``default.yaml`` with one or more experiment overrides.

    Passing ``configs/train.yaml`` no longer discards shared paths, budgets,
    or model dimensions. Later files take precedence over earlier files.
    """
    root = repo_root()
    default_path = root / "configs" / "default.yaml"
    cfg = _read_yaml(default_path)
    for override_path in _normalise_paths(path):
        if override_path.resolve() == default_path.resolve():
            continue
        cfg = deep_merge(cfg, _read_yaml(override_path))
    if validate:
        validate_config(cfg)
    return cfg


def resolve_path(cfg: dict[str, Any], key: str) -> Path:
    value = cfg.get("paths", {}).get(key)
    if value is None:
        raise KeyError(f"unknown configured path: {key}")
    path = Path(value)
    return path if path.is_absolute() else repo_root() / path
