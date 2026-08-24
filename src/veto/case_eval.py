from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import TaskCache
from .models import VetoPolicy
from .verifier import CLEAR_WIDTH_MM, canonicalize_width, verify_doors


def run_bim_clearwidth(
    cache: TaskCache, model: VetoPolicy, threshold_mm: float = CLEAR_WIDTH_MM
) -> dict[str, Any]:
    items = cache.items
    if items is None:
        raise ValueError(f"{cache.task_id} has no item bag")
    x = torch.from_numpy(items)
    with torch.no_grad():
        pred = model.score_items(x)
        widths = pred["width"].cpu().numpy()
    meta = cache.meta or {}
    units = meta.get("units")
    if units is not None:
        units = list(units)
        widths = np.array(
            [canonicalize_width(float(w), str(u)) for w, u in zip(widths, units)],
            dtype=float,
        )
    verdict = verify_doors(widths, threshold_mm=threshold_mm)
    calls = meta.get("trace") or []
    out = {
        "task_id": cache.task_id,
        "n_doors": verdict["n"],
        "n_pass": verdict["n_pass"],
        "n_fail": verdict["n_fail"],
        "widths_mm": verdict["widths_mm"],
        "n_calls": len(calls) if calls else int(round(cache.n_calls)),
        "threshold_mm": threshold_mm,
        "repaired_idx": list(meta.get("repaired_idx") or []),
    }
    x_grad = x.detach().clone().requires_grad_(True)
    width_g = model.score_items(x_grad)["width"]
    width_g.sum().backward()
    out["width_saliency"] = x_grad.grad.detach().abs().mean(dim=-1).cpu().numpy()
    model.zero_grad(set_to_none=True)
    return out


def run_cad_bracket(cache: TaskCache, model: VetoPolicy) -> dict[str, Any]:
    items = cache.items
    if items is None:
        raise ValueError(f"{cache.task_id} has no item bag")
    x = torch.from_numpy(items)
    with torch.no_grad():
        pred = model.score_items(x)
        scores = torch.sigmoid(pred["pass_logit"]).cpu().numpy()
    passed = scores >= 0.5
    meta = cache.meta or {}
    calls = meta.get("trace") or []
    out = {
        "task_id": cache.task_id,
        "n_constraints": int(len(scores)),
        "n_pass": int(passed.sum()),
        "n_fail": int((~passed).sum()),
        "scores": scores,
        "n_calls": len(calls) if calls else int(round(cache.n_calls)),
        "min_edge_mm": float((meta.get("min_edge_mm") or 0.0)),
    }
    x_grad = x.detach().clone().requires_grad_(True)
    logit = model.score_items(x_grad)["pass_logit"]
    logit.sum().backward()
    out["constraint_saliency"] = x_grad.grad.detach().abs().mean(dim=-1).cpu().numpy()
    model.zero_grad(set_to_none=True)
    return out


def load_case(feature_dir: str | Path, task_id: str) -> TaskCache:
    return TaskCache.load(Path(feature_dir) / f"{task_id}.npz")
