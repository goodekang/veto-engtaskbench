from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def repo_root() -> Path:
    here = Path(__file__).resolve()
    for p in here.parents:
        if (p / "configs" / "default.yaml").exists():
            return p
    return Path.cwd()


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path else repo_root() / "configs" / "default.yaml"
    with cfg_path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)
