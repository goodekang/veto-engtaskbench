from __future__ import annotations

import numpy as np
import pandas as pd


def task_success_rate(y_true, y_hat) -> float:
    y_true = np.asarray(y_true).astype(int)
    y_hat = np.asarray(y_hat).astype(int)
    return float(np.mean(y_true == y_hat)) if y_hat.size else 0.0


def binary_success_rate(success) -> float:
    return float(np.mean(np.asarray(success).astype(float)))


def constraint_satisfaction(csr) -> float:
    return float(np.mean(np.asarray(csr).astype(float)))


def youden_threshold(y_true, scores) -> tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    s = np.asarray(scores).astype(float)
    order = np.argsort(-s)
    y = y[order]
    tp = np.cumsum(y)
    fp = np.cumsum(1 - y)
    p = max(int(y.sum()), 1)
    n = max(int((1 - y).sum()), 1)
    tpr = tp / p
    fpr = fp / n
    j = tpr - fpr
    idx = int(np.argmax(j))
    return float(s[order][idx]), float(j[idx])


def expected_calibration_error(y_true, scores, n_bins: int = 10) -> float:
    y = np.asarray(y_true).astype(float)
    s = np.clip(np.asarray(scores).astype(float), 0, 1)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (s >= bins[i]) & (s < bins[i + 1] if i < n_bins - 1 else s <= bins[i + 1])
        if not mask.any():
            continue
        ece += abs(y[mask].mean() - s[mask].mean()) * mask.mean()
    return float(ece)


def mcnemar_exact(a_ok, b_ok) -> float:
    a = np.asarray(a_ok).astype(int)
    b = np.asarray(b_ok).astype(int)
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    # two-sided binomial tail
    from math import comb

    p = sum(comb(n, i) for i in range(0, k + 1)) / (2 ** n)
    return float(min(1.0, 2 * p))


def bootstrap_ci(success, n_boot: int = 10000, alpha: float = 0.05, seed: int = 42):
    rng = np.random.default_rng(seed)
    x = np.asarray(success).astype(float)
    n = len(x)
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = rng.choice(x, size=n, replace=True).mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    mean = float(x.mean())
    return mean, float(mean - lo), float(hi - mean)


def stratified_bootstrap_ci(df: pd.DataFrame, col: str = "success", strata: str = "tier", n_boot: int = 10000, seed: int = 42):
    rng = np.random.default_rng(seed)
    groups = [g[col].to_numpy() for _, g in df.groupby(strata)]
    stats = np.empty(n_boot)
    for i in range(n_boot):
        draw = [rng.choice(g, size=len(g), replace=True) for g in groups if len(g)]
        stats[i] = np.concatenate(draw).mean()
    mean = float(df[col].mean())
    lo, hi = np.quantile(stats, [0.025, 0.975])
    return mean, float(mean - lo), float(hi - mean)


def summarize_predictions(df: pd.DataFrame) -> dict[str, float]:
    tsr = float(df["success"].mean()) if "success" in df else float(df["y_hat"].mean())
    out = {"tsr": tsr, "n": float(len(df))}
    if "executable" in df:
        out["er"] = float(df["executable"].mean())
    if "csr" in df:
        out["csr"] = float(df["csr"].mean())
    if "n_calls" in df:
        out["n_calls"] = float(df["n_calls"].mean())
    if "cost_usd" in df:
        out["cost_usd"] = float(df["cost_usd"].mean())
    if "score" in df and "y_true" in df:
        out["ece"] = expected_calibration_error(df["y_true"].values, df["score"].values)
    return out
