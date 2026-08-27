from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


EXPECTED_TIERS = {
    "T1": 30,
    "T2": 34,
    "T3": 30,
    "T4": 26,
    "C1": 28,
    "C2": 36,
    "C3": 22,
    "C4": 14,
}


def validate_manifest(manifest: pd.DataFrame, *, strict_benchmark: bool = False) -> None:
    required = {"task_id", "domain", "tier"}
    missing = required - set(manifest.columns)
    if missing:
        raise ValueError(f"manifest is missing columns: {sorted(missing)}")
    if manifest["task_id"].isna().any() or manifest["task_id"].duplicated().any():
        raise ValueError("task_id values must be non-empty and unique")
    if not set(manifest["domain"]).issubset({"bim", "cad"}):
        raise ValueError("domain must be 'bim' or 'cad'")
    unknown = set(manifest["tier"]) - set(EXPECTED_TIERS)
    if unknown:
        raise ValueError(f"unknown complexity tiers: {sorted(unknown)}")
    bad_domain = manifest[
        ((manifest["domain"] == "bim") & ~manifest["tier"].str.startswith("T"))
        | ((manifest["domain"] == "cad") & ~manifest["tier"].str.startswith("C"))
    ]
    if not bad_domain.empty:
        raise ValueError("tier prefixes do not agree with task domains")
    if strict_benchmark:
        counts = manifest["tier"].value_counts().to_dict()
        if len(manifest) != 220 or counts != EXPECTED_TIERS:
            raise ValueError(
                f"EngTaskBench-220 composition mismatch: n={len(manifest)}, tiers={counts}"
            )


def load_manifest(path: str | Path, *, strict_benchmark: bool = False) -> pd.DataFrame:
    manifest = pd.read_csv(path)
    validate_manifest(manifest, strict_benchmark=strict_benchmark)
    return manifest


def assign_policy_split(
    manifest: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    val_fraction: float = 0.15,
    seed: int = 42,
) -> pd.DataFrame:
    """Create deterministic tier-stratified splits for the replay surrogate."""
    if train_fraction <= 0 or val_fraction <= 0 or train_fraction + val_fraction >= 1:
        raise ValueError("policy split fractions must be positive and sum to less than one")
    out = manifest.copy()
    labels = pd.Series(index=out.index, dtype="object")
    for tier, group in out.groupby("tier", sort=True):
        ranked = sorted(
            group.index,
            key=lambda idx: hashlib.sha256(
                f"{seed}:{tier}:{out.at[idx, 'task_id']}".encode("utf-8")
            ).digest(),
        )
        n = len(ranked)
        n_train = max(1, int(round(n * train_fraction)))
        n_val = max(1, int(round(n * val_fraction)))
        if n_train + n_val >= n:
            n_train = max(1, n - n_val - 1)
        labels.loc[ranked[:n_train]] = "policy_train"
        labels.loc[ranked[n_train : n_train + n_val]] = "policy_val"
        labels.loc[ranked[n_train + n_val :]] = "policy_test"
    out["policy_split"] = labels
    return out


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
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"task cache is missing: {path}")
        with np.load(path, allow_pickle=True) as blob:
            required = {"task_id", "domain", "tier", "obs", "y", "csr", "n_calls"}
            missing = required - set(blob.files)
            if missing:
                raise ValueError(f"{path} is missing cache arrays: {sorted(missing)}")
            meta = blob["meta"].item() if "meta" in blob.files else {}
            items = blob["items"] if "items" in blob.files else None
            item_y = blob["item_y"] if "item_y" in blob.files else None
            obs = blob["obs"].astype(np.float32)
            if obs.ndim != 2 or not obs.shape[0] or not obs.shape[1]:
                raise ValueError(f"{path} obs must be a non-empty [tokens, features] array")
            if not np.isfinite(obs).all():
                raise ValueError(f"{path} obs contains non-finite values")
            return cls(
                task_id=str(blob["task_id"]),
                domain=str(blob["domain"]),
                tier=str(blob["tier"]),
                obs=obs,
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
        if cache.task_id != str(row.task_id):
            raise ValueError(f"cache/manifest task mismatch: {cache.task_id} != {row.task_id}")
        if cache.domain != str(row.domain) or cache.tier != str(row.tier):
            raise ValueError(f"cache metadata mismatch for {row.task_id}")
        return {
            "task_id": cache.task_id,
            "domain": cache.domain,
            "tier": cache.tier,
            "obs": torch.from_numpy(cache.obs),
            "y": torch.tensor(float(row.y_true) if "y_true" in row.index else cache.y),
            "csr": torch.tensor(cache.csr),
            "n_calls": torch.tensor(cache.n_calls),
            "items": None if cache.items is None else torch.from_numpy(cache.items),
            "item_y": None if cache.item_y is None else torch.from_numpy(cache.item_y),
            "meta": cache.meta or {},
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
        "n_calls": torch.stack([x["n_calls"].float() for x in batch]),
        "meta": [x["meta"] for x in batch],
    }


def list_feature_ids(feature_dir: str | Path) -> list[str]:
    return sorted(p.stem for p in Path(feature_dir).glob("*.npz"))


def load_split_caches(manifest: pd.DataFrame, feature_dir: str | Path) -> list[TaskCache]:
    root = Path(feature_dir)
    return [TaskCache.load(root / f"{tid}.npz") for tid in manifest["task_id"]]


def domain_mask(manifest: pd.DataFrame, domain: str) -> pd.DataFrame:
    return manifest[manifest["domain"] == domain].reset_index(drop=True)
