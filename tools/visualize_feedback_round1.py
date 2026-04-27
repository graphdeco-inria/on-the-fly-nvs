"""Three figures for the feedback-round-1 verification ablation report.

Inputs (under --root, default results/feedback_round1):
  ablation_lr_poses.json
  per_view_pnp_stats.json
  seam_residual_analysis.json

Outputs:
  figures/fig_3_5_1_lr_poses_ablation.png   1×3, holdout PSNR / ATE / RPE rot
  figures/fig_3_6_1_per_view_pnp.png        1×2, mean ew / mean inliers
  figures/fig_3_7_1_seam_residual.png       scatter, seam dist × mean residual

No code changes; reads the JSON aggregates produced by
tools/analyze_feedback_round1.py. Run from on-the-fly-nvs/ root.
"""

from __future__ import annotations

import argparse
import json
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np


def load_json(path: str):
    with open(path) as fh:
        return json.load(fh)


def fmt_value(x: float, p: int = 3) -> str:
    if abs(x) < 0.01 and x != 0:
        return f"{x:.{p+1}f}"
    return f"{x:.{p}f}"


# ---------------------------------------------------------------------------
# fig 3.5.1 — lr_poses ablation
# ---------------------------------------------------------------------------
def fig_lr_poses_ablation(rows: List[Dict], out_path: str):
    rows = sorted(rows, key=lambda r: r["id"])
    ids = [r["id"] for r in rows]
    holdout = [(r.get("render_split") or {}).get("holdout", {}).get("psnr",
                                                                    np.nan)
               for r in rows]
    ate = [r.get("ate_rmse_m", np.nan) for r in rows]
    rpe = [r.get("rpe_rot_rmse_deg", np.nan) for r in rows]

    labels = {
        "A": "A\nlr=1e-4\nrefine on",
        "B": "B\nlr=0\nrefine on",
        "C": "C\nlr=1e-4\nrefine off",
        "D": "D\nlr=0\nrefine off",
    }
    xticklabels = [labels.get(i, i) for i in ids]
    color = "#4C72B0"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    panels = [
        (axes[0], holdout, "holdout PSNR (dB)", 2),
        (axes[1], ate, "ATE RMSE (m)", 4),
        (axes[2], rpe, "RPE rot RMSE (°)", 3),
    ]
    for ax, values, ylabel, prec in panels:
        x = np.arange(len(ids))
        bars = ax.bar(x, values, color=color, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(xticklabels, fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(axis="y", alpha=0.3)
        # value labels
        for bar, v in zip(bars, values):
            if np.isnan(v):
                continue
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    fmt_value(v, prec),
                    ha="center", va="bottom", fontsize=10)
        # add headroom for value labels
        finite = [v for v in values if np.isfinite(v)]
        if finite:
            top = max(finite)
            ax.set_ylim(0, top * 1.12)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# fig 3.6.1 — per-view PnP
# ---------------------------------------------------------------------------
def fig_per_view_pnp(rows: List[Dict], seam_rows: List[Dict], out_path: str):
    # merge azimuth from seam_rows so we can annotate "rel_R angle"
    az_by_view = {r["view"]: r.get("view_azimuth_deg") for r in seam_rows}

    rows = sorted(rows, key=lambda r: r["mean_effective_weight"], reverse=True)
    views = [r["view"] for r in rows]
    weights = [r["mean_effective_weight"] for r in rows]
    inliers = [r["mean_n_inliers"] for r in rows]
    succ = [r["success_rate"] for r in rows]

    def label(view: str) -> str:
        az = az_by_view.get(view)
        if az is None or not np.isfinite(az):
            return view
        return f"{view} ({az:.0f}°)"

    ylabels = [label(v) for v in views]
    color = "#4C72B0"
    color_warn = "#DD8452"

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # subplot 1: mean effective_weight
    ax = axes[0]
    y = np.arange(len(views))
    bars = ax.barh(y, weights, color=color, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("mean effective_weight")
    ax.grid(axis="x", alpha=0.3)
    for bar, v, s in zip(bars, weights, succ):
        ax.text(bar.get_width() + max(weights) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.3f}",
                va="center", ha="left", fontsize=9)
        # mark sub-1.0 success rates
        if s < 1.0:
            ax.text(0.001, bar.get_y() + bar.get_height() / 2,
                    f"succ={s:.2f}", va="center", ha="left",
                    fontsize=8, color=color_warn)
    ax.set_xlim(0, max(weights) * 1.18)

    # subplot 2: mean n_inliers (same view ordering as subplot 1)
    ax = axes[1]
    bars = ax.barh(y, inliers, color=color, edgecolor="black", linewidth=0.5)
    ax.set_yticks(y)
    ax.set_yticklabels(ylabels, fontsize=10)
    ax.invert_yaxis()
    ax.set_xlabel("mean n_inliers")
    ax.grid(axis="x", alpha=0.3)
    for bar, v in zip(bars, inliers):
        ax.text(bar.get_width() + max(inliers) * 0.01,
                bar.get_y() + bar.get_height() / 2,
                f"{v:.0f}",
                va="center", ha="left", fontsize=9)
    ax.set_xlim(0, max(inliers) * 1.15)

    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ---------------------------------------------------------------------------
# fig 3.7.1 — seam distance × mean residual
# ---------------------------------------------------------------------------
def fig_seam_residual(rows: List[Dict], out_path: str):
    finite = [r for r in rows
              if np.isfinite(r["seam_azimuth_distance_deg"])
              and np.isfinite(r["mean_residual_px"])]
    xs = np.array([r["seam_azimuth_distance_deg"] for r in finite])
    ys = np.array([r["mean_residual_px"] for r in finite])
    names = [r["view"] for r in finite]

    if xs.std() > 1e-6 and ys.std() > 1e-6:
        corr = float(np.corrcoef(xs, ys)[0, 1])
    else:
        corr = float("nan")
    # linear regression in log-y space (since y is log-scaled in the plot)
    log_ys = np.log10(ys)
    a, b = np.polyfit(xs, log_ys, 1)
    xs_line = np.linspace(xs.min(), xs.max(), 50)
    ys_line = 10 ** (a * xs_line + b)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(xs, ys, s=70, color="#4C72B0", edgecolor="black",
               linewidth=0.6, zorder=3)

    # annotate point names
    for x, y, n in zip(xs, ys, names):
        ax.annotate(n, xy=(x, y),
                    xytext=(6, 6), textcoords="offset points",
                    fontsize=9)

    ax.plot(xs_line, ys_line, linestyle="--", color="#888888",
            linewidth=1.2, label="lin. fit (log y)")

    ax.set_yscale("log")
    ax.set_xlabel("seam azimuthal distance (°)")
    ax.set_ylabel("mean reprojection residual (px, log scale)")
    ax.grid(True, which="both", alpha=0.3)
    # Headroom at top so the highest annotation isn't clipped.
    y_top = max(ys) * 1.6
    y_bot = max(min(ys) * 0.5, 1.0)
    ax.set_ylim(y_bot, y_top)

    txt = f"Pearson r = {corr:.3f}"
    ax.text(0.03, 0.97, txt, transform=ax.transAxes,
            ha="left", va="top",
            fontsize=11,
            bbox=dict(facecolor="white", edgecolor="black", alpha=0.85,
                      boxstyle="round,pad=0.3"))

    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/feedback_round1")
    args = p.parse_args()

    fig_dir = os.path.join(args.root, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    stage1 = load_json(os.path.join(args.root, "ablation_lr_poses.json"))
    stage2 = load_json(os.path.join(args.root, "per_view_pnp_stats.json"))
    stage3 = load_json(os.path.join(args.root, "seam_residual_analysis.json"))

    fig_lr_poses_ablation(
        stage1, os.path.join(fig_dir, "fig_3_5_1_lr_poses_ablation.png"))
    fig_per_view_pnp(
        stage2, stage3, os.path.join(fig_dir, "fig_3_6_1_per_view_pnp.png"))
    fig_seam_residual(
        stage3, os.path.join(fig_dir, "fig_3_7_1_seam_residual.png"))


if __name__ == "__main__":
    main()
