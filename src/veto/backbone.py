from __future__ import annotations

import os
from pathlib import Path

from .models import CachedObsEncoder, VetoPolicy


class CachedCompletionEncoder:
    """Offline stand-in for the LLM backbone.

    Eval and replay never download weights. If a local cache directory is
    present we record its path; otherwise scoring stays on ``VetoPolicy``.
    """

    def __init__(self, cache_dir: str | os.PathLike | None = None):
        self.cache_dir = Path(cache_dir) if cache_dir else _default_cache()
        self.available = self.cache_dir is not None and self.cache_dir.exists()

    def probe(self) -> dict[str, object]:
        return {
            "cache_dir": str(self.cache_dir) if self.cache_dir else None,
            "available": bool(self.available),
            "download": False,
        }


def _default_cache() -> Path | None:
    env = os.environ.get("VETO_LLM_CACHE") or os.environ.get("HF_HOME")
    return Path(env) if env else None


def attach_cached_encoder(model: VetoPolicy) -> CachedObsEncoder:
    return model.encoder
