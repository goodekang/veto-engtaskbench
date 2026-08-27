from __future__ import annotations

import numpy as np
import pandas as pd


def task_success_rate(success, y_hat=None) -> float:
    """Mean episode success.

    ``y_hat`` is accepted for backward compatibility and then computes
    prediction agreement; paper TSR should pass the episode-success vector.
    """
    if y_hat is not None:
        truth = np.asarray(success).astype(int)
        pred = np.asarray(y_hat).astype(int)
        if truth.shape != pred.shape:
            raise ValueError("y_true and y_hat must have the same shape")
        return float(np.mean(truth == pred)) if pred.size else 0.0
    values = np.asarray(success, dtype=float)
    return float(values.mean()) if values.size else 0.0


def binary_success_rate(success) -> float:
    return float(np.mean(np.asarray(success).astype(float)))


def constraint_satisfaction(csr) -> float:
    return float(np.mean(np.asarray(csr).astype(float)))


def grounding_accuracy(argument_ok) -> float:
    values = np.asarray(argument_ok, dtype=float)
    return float(values.mean()) if values.size else 0.0


def report_faithfulness(claim_has_source) -> float:
    values = np.asarray(claim_has_source, dtype=float)
    return float(values.mean()) if values.size else 0.0


def stable_success(run_success: np.ndarray) -> float:
    values = np.asarray(run_success, dtype=bool)
    if values.ndim != 2:
        raise ValueError("run_success must have shape [runs, tasks]")
    return float(values.all(axis=0).mean()) if values.shape[1] else 0.0


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
    if n == 0:
        raise ValueError("bootstrap requires at least one observation")
    if n_boot <= 0:
        raise ValueError("n_boot must be positive")
    stats = np.empty(n_boot)
    for i in range(n_boot):
        stats[i] = rng.choice(x, size=n, replace=True).mean()
    lo, hi = np.quantile(stats, [alpha / 2, 1 - alpha / 2])
    mean = float(x.mean())
    return mean, float(mean - lo), float(hi - mean)


def stratified_bootstrap_ci(df: pd.DataFrame, col: str = "success", strata: str = "tier", n_boot: int = 10000, seed: int = 42):
    if df.empty:
        raise ValueError("stratified bootstrap requires at least one observation")
    if col not in df or strata not in df:
        raise KeyError(f"missing bootstrap columns: {col!r}, {strata!r}")
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
    if "grounding_ok" in df:
        out["grounding"] = grounding_accuracy(df["grounding_ok"])
    if "claim_has_source" in df:
        out["faithfulness"] = report_faithfulness(df["claim_has_source"])
    return out


def grouped_tsr(df: pd.DataFrame, by: str) -> pd.DataFrame:
    if by not in df or "success" not in df:
        raise KeyError(f"grouped TSR requires {by!r} and 'success'")
    return (
        df.groupby(by, sort=False)["success"]
        .agg(n="size", success="sum", tsr="mean")
        .reset_index()
    )
