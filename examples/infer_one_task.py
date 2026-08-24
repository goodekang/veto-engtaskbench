#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from veto.case_eval import load_case, run_bim_clearwidth, run_cad_bracket
from veto.config import load_config, repo_root
from veto.models import load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", default="bim", choices=["bim", "cad"])
    parser.add_argument("--ckpt", default=None)
    args = parser.parse_args()
    cfg = load_config()
    root = repo_root()
    ckpt_path = Path(args.ckpt) if args.ckpt else root / "checkpoints" / cfg["main_run"] / "best.pt"
    model, blob = load_checkpoint(ckpt_path)
    feat_dir = root / "data" / "features"

    if args.case == "bim":
        task_id = cfg.get("named_case", "bim_t4_schependomlaan_clearwidth")
        cache = load_case(feat_dir, task_id)
        out = run_bim_clearwidth(cache, model)
        repaired = out["repaired_idx"]
        widths = out["widths_mm"]
        print(f"task_id={out['task_id']}")
        print(f"doors={out['n_doors']}  pass={out['n_pass']}  fail={out['n_fail']}")
        if repaired:
            shown = ", ".join(f"{widths[i]:.0f} mm" for i in repaired)
            print(f"repaired_widths={shown}")
        else:
            print(f"median_width_mm={float(np.median(widths)):.0f}")
        print(f"tool_calls={out['n_calls']}")
        if "width_saliency" in out:
            print(f"width_saliency_mean={float(np.mean(out['width_saliency'])):.4f}")
    else:
        task_id = "cad_c3_bracket_4bolt"
        cache = load_case(feat_dir, task_id)
        out = run_cad_bracket(cache, model)
        print(f"task_id={out['task_id']}")
        print(f"constraints={out['n_pass']}/{out['n_constraints']}")
        print(f"tool_calls={out['n_calls']}")
        if out["min_edge_mm"]:
            print(f"min_edge_mm={out['min_edge_mm']:.1f}")
        if "constraint_saliency" in out:
            print(f"constraint_saliency_mean={float(np.mean(out['constraint_saliency'])):.4f}")


if __name__ == "__main__":
    main()
