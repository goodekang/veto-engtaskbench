from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


def load_manifest(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


@dataclass
class TaskCache:
    task_id: str
    domain: str
    tier: str
    obs: np.ndarray
    y: float
    csr: float
    n_calls: float
    items: np.ndarray | None = None
    item_y: np.ndarray | None = None
    meta: dict[str, Any] | None = None

    @classmethod
    def load(cls, path: str | Path) -> "TaskCache":
        blob = np.load(path, allow_pickle=True)
        meta = blob["meta"].item() if "meta" in blob.files else {}
        items = blob["items"] if "items" in blob.files else None
        item_y = blob["item_y"] if "item_y" in blob.files else None
        return cls(
            task_id=str(blob["task_id"]),
            domain=str(blob["domain"]),
            tier=str(blob["tier"]),
            obs=blob["obs"].astype(np.float32),
            y=float(blob["y"]),
            csr=float(blob["csr"]),
            n_calls=float(blob["n_calls"]),
            items=None if items is None else items.astype(np.float32),
            item_y=None if item_y is None else item_y.astype(np.float32),
            meta=meta,
        )


class TaskDataset(Dataset):
    def __init__(self, manifest: pd.DataFrame, feature_dir: str | Path):
        self.manifest = manifest.reset_index(drop=True)
        self.feature_dir = Path(feature_dir)

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.manifest.iloc[idx]
        cache = TaskCache.load(self.feature_dir / f"{row.task_id}.npz")
        return {
            "task_id": cache.task_id,
            "domain": cache.domain,
            "tier": cache.tier,
            "obs": torch.from_numpy(cache.obs),
            "y": torch.tensor(float(row.y_true) if "y_true" in row.index else cache.y),
            "csr": torch.tensor(cache.csr),
            "n_calls": torch.tensor(cache.n_calls),
        }


def pad_collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    lengths = torch.tensor([x["obs"].shape[0] for x in batch], dtype=torch.long)
    d = batch[0]["obs"].shape[-1]
    t = int(lengths.max())
    obs = torch.zeros(len(batch), t, d)
    for i, x in enumerate(batch):
        obs[i, : x["obs"].shape[0]] = x["obs"]
    return {
        "task_id": [x["task_id"] for x in batch],
        "domain": [x["domain"] for x in batch],
        "tier": [x["tier"] for x in batch],
        "obs": obs,
        "lengths": lengths,
        "y": torch.stack([x["y"].float() for x in batch]),
        "csr": torch.stack([x["csr"].float() for x in batch]),
    }


def list_feature_ids(feature_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(feature_dir).glob("*.npz"))


def load_split_caches(manifest: pd.DataFrame, feature_dir: str | Path) -> list[TaskCache]:
    root = Path(feature_dir)
    return [TaskCache.load(root / f"{tid}.npz") for tid in manifest["task_id"]]


def domain_mask(manifest: pd.DataFrame, domain: str) -> pd.DataFrame:
    return manifest[manifest["domain"] == domain].reset_index(drop=True)
