from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .config import repo_root
from .metrics import (
    bootstrap_ci,
    expected_calibration_error,
    mcnemar_exact,
    stratified_bootstrap_ci,
    summarize_predictions,
    youden_threshold,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred", required=True)
    parser.add_argument("--against", default=None, help="optional paired dump for McNemar")
    args = parser.parse_args()
    root = repo_root()
    path = root / args.pred if not Path(args.pred).is_absolute() else Path(args.pred)
    df = pd.read_csv(path)
    summary = summarize_predictions(df)
    success = df["success"] if "success" in df.columns else df["y_hat"]
    mean, lo, hi = bootstrap_ci(success.values)
    smean, slo, shi = stratified_bootstrap_ci(df) if "tier" in df.columns else (mean, lo, hi)
    print(f"n={int(summary['n'])}  tsr={100 * summary['tsr']:.1f}%")
    if "er" in summary:
        print(f"er={100 * summary['er']:.1f}%")
    if "csr" in summary:
        print(f"csr={100 * summary['csr']:.1f}%")
    if "n_calls" in summary:
        print(f"calls={summary['n_calls']:.1f}")
    if "cost_usd" in summary:
        print(f"cost={summary['cost_usd']:.3f} USD")
    print(f"bootstrap95=[{100 * (mean - lo):.1f}, {100 * (mean + hi):.1f}]")
    print(f"stratified95=[{100 * (smean - slo):.1f}, {100 * (smean + shi):.1f}]")
    if "score" in df.columns and "y_true" in df.columns:
        thr, j = youden_threshold(df["y_true"].values, df["score"].values)
        print(f"youden_thr={thr:.3f}  J={j:.3f}  ece={summary.get('ece', 0):.3f}")
    if args.against:
        other = pd.read_csv(root / args.against if not Path(args.against).is_absolute() else args.against)
        merged = df.merge(other, on="task_id", suffixes=("", "_b"))
        p = mcnemar_exact(merged["success"], merged["success_b"])
        print(f"mcnemar_p={p:.4g}")


if __name__ == "__main__":
    main()
