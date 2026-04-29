"""Aggregate Sim(3) metrics across N runs (compare_with_colmap output).

Inputs: list of metrics.json files written by tools/compare_with_colmap.py.
Each contains ate_rmse, rpe_rot_rmse_deg, sim3_scale, etc.

Output:
  - prints a markdown summary (mean ± σ for each metric)
  - optionally writes JSON

Usage:
  python tools/aggregate_sim3.py \
      compare/b1_5run_seed42/metrics.json \
      compare/b1_5run_seed43/metrics.json \
      ... \
      --out report/iter100/rig_unit_sim3_5run_summary.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", help="N JSON files")
    parser.add_argument("--out", default=None)
    parser.add_argument("--label", default="")
    args = parser.parse_args()

    keys = [
        ("ate_rmse", "ATE RMSE (m)"),
        ("rpe_trans_rmse", "RPE trans RMSE"),
        ("rpe_rot_rmse_deg", "RPE rot RMSE (°)"),
        ("sim3_scale", "Sim(3) scale"),
        ("sim3_R_angle_deg", "Sim(3) R angle (°)"),
        ("sim3_t_norm", "Sim(3) |t|"),
        ("scene_scale_colmap", "scene scale (COLMAP)"),
    ]
    rows = {k: [] for k, _ in keys}
    n = 0
    for p in args.paths:
        if not os.path.exists(p):
            print(f"[warn] missing {p}; skip"); continue
        with open(p) as fh:
            m = json.load(fh)
        n += 1
        for k, _ in keys:
            v = m.get(k)
            if v is not None:
                rows[k].append(float(v))

    if n == 0:
        sys.exit("[error] no inputs read")

    summary = {}
    for k, label in keys:
        vs = rows[k]
        if vs:
            summary[k] = {"label": label, "n": len(vs),
                          "mean": float(np.mean(vs)),
                          "std": float(np.std(vs, ddof=0)),
                          "min": float(np.min(vs)),
                          "max": float(np.max(vs))}

    L = []
    L.append(f"| metric | mean ± σ ({args.label or f'N={n}'}) |")
    L.append("|---|---|")
    for k, label in keys:
        s = summary.get(k)
        if s is None:
            continue
        L.append(f"| {label} | {s['mean']:.4g} ± {s['std']:.2g} |")
    print("\n".join(L))

    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                    exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump({"summary": summary, "n_runs": n,
                       "input_paths": args.paths}, fh, indent=2)
        print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
