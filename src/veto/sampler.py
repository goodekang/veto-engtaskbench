from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch.utils.data import WeightedRandomSampler


HARD_TIERS = {"T3", "T4", "C3", "C4"}


def tier_weights(manifest: pd.DataFrame, hard_boost: float = 2.4) -> torch.Tensor:
    weights = []
    for row in manifest.itertuples():
        w = hard_boost if str(row.tier) in HARD_TIERS else 1.0
        if getattr(row, "y_true", 1) == 0:
            w *= 1.3
        weights.append(w)
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.mean()
    return torch.as_tensor(w, dtype=torch.double)


def make_train_sampler(manifest: pd.DataFrame, hard_boost: float = 2.4) -> WeightedRandomSampler:
    w = tier_weights(manifest, hard_boost=hard_boost)
    return WeightedRandomSampler(weights=w, num_samples=len(manifest), replacement=True)
