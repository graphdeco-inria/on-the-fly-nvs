"""5-cell extension of fig_3_5_1: adds the rig-unit (B-1) measurement 5 row.

Reads:
  results/feedback_round1/ablation_lr_poses.json   (4 cells, view-unit)
  results/b1_v2/run_B1/render_eval/split_summary.json  (m5 PSNR)
  results/b1_v2/run_B1/metadata_view_drift.json   (m5 σ_center)
  compare/run_B1_v2/metrics.json                  (m5 ATE / RPE)

Writes:
  ../video_picture/260429/fig_3_5_2_lr_poses_5cell.png

Run from on-the-fly-nvs/ root.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict

import matplotlib.pyplot as plt
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ablation_json",
                   default="results/feedback_round1/ablation_lr_poses.json")
    p.add_argument("--m5_split",
                   default="results/b1_v2/run_B1/render_eval/split_summary.json")
    p.add_argument("--m5_drift",
                   default="results/b1_v2/run_B1/metadata_view_drift.json")
    p.add_argument("--m5_compare",
                   default="compare/run_B1_v2/metrics.json")
    p.add_argument("--out",
                   default="../video_picture/260429/fig_3_5_2_lr_poses_5cell.png")
    args = p.parse_args()

    rows = json.load(open(args.ablation_json))  # list of 4
    m5_split = json.load(open(args.m5_split))
    m5_drift = json.load(open(args.m5_drift))
    m5_cmp = json.load(open(args.m5_compare))

    # build 5 cells
    cells = []
    for r in sorted(rows, key=lambda r: r["id"]):
        rs = r.get("render_split") or {}
        cells.append({
            "id": r["id"],
            "lr": r["lr_poses"],
            "refine": "on" if r["refine"] else "off",
            "holdout_psnr": (rs.get("holdout") or {}).get("psnr", float("nan")),
            "ate": r.get("ate_rmse_m", float("nan")),
            "rpe": r.get("rpe_rot_rmse_deg", float("nan")),
        })
    cells.append({
        "id": "5",
        "lr": "1e-4",
        "refine": "on (rig-unit)",
        "holdout_psnr": m5_split["splits"]["holdout"]["psnr"],
        "ate": m5_cmp["ate_rmse"],
        "rpe": m5_cmp["rpe_rot_rmse_deg"],
    })

    labels = []
    for c in cells:
        if c["id"] == "5":
            labels.append(f"5\nrig-unit\n6 DoF/ts")
        else:
            labels.append(f"{c['id']}\nlr={c['lr']}\nrefine {c['refine']}")
    holdout = [c["holdout_psnr"] for c in cells]
    ate = [c["ate"] for c in cells]
    rpe = [c["rpe"] for c in cells]

    color_default = "#4C72B0"
    color_m5 = "#DD8452"
    colors = [color_default] * 4 + [color_m5]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5.2))

    panels = [
        (axes[0], holdout, "holdout PSNR (dB)", 2),
        (axes[1], ate, "ATE RMSE (m)", 4),
        (axes[2], rpe, "RPE rot RMSE (°)", 3),
    ]
    for ax, values, ylabel, prec in panels:
        x = np.arange(len(cells))
        bars = ax.bar(x, values, color=colors, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        for bar, v in zip(bars, values):
            if not np.isfinite(v):
                continue
            fmt = f"{v:.{prec}f}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt, ha="center", va="bottom", fontsize=10)
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            ax.set_ylim(0, max(finite) * 1.14)

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".",
                exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
