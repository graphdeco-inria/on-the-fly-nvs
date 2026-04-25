"""Build the two figures for the 4/16 follow-up: OTF vs COLMAP Sim(3)
alignment evidence + scale gauge origin / N-run reproducibility.

Reads from compare/run_{1..N}/{metrics,sim3,aligned_otf_traj,colmap_traj}
and one raw results/multirun/run_1/trajectory_otf.txt for the "before
alignment" plot. Writes:
  compare/figures/fig1_alignment.png
  compare/figures/fig2_scale_origin.png
  compare/figures/summary.md

Usage:
  python tools/visualize_alignment.py
  python tools/visualize_alignment.py --runs 5 --raw_run 1
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as Rscipy


def read_tum(path):
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            p = line.split()
            ts = int(float(p[0]))
            t = np.array([float(x) for x in p[1:4]])
            q = np.array([float(x) for x in p[4:8]])
            R = Rscipy.from_quat(q).as_matrix()
            rows.append((ts, R, t))
    rows.sort(key=lambda r: r[0])
    return rows


def centres(rows):
    return np.stack([(-R.T @ t) for _, R, t in rows])


def rot_angle_deg(R):
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


def per_step_ate(otf_aligned_centres, col_centres):
    return np.linalg.norm(otf_aligned_centres - col_centres, axis=1)


def per_step_rpe_rot(otf_aligned, col):
    out = []
    for i in range(len(otf_aligned) - 1):
        dR_o = otf_aligned[i + 1][1] @ otf_aligned[i][1].T
        dR_c = col[i + 1][1] @ col[i][1].T
        out.append(rot_angle_deg(dR_o @ dR_c.T))
    return np.array(out)


def apply_sim3_to_centre(C, s, R, t):
    return s * (R @ C.T).T + t


def apply_sim3_to_traj(rows, s, R, t):
    """Transform world frame: C' = s R C + t. New w2c poses preserve
    obs (R_w2c' = R_w2c @ R^T, t_w2c' = -R_new @ C_new)."""
    out = []
    for ts, R_wc, t_wc in rows:
        C = -R_wc.T @ t_wc
        C_new = s * R @ C + t
        R_new = R_wc @ R.T
        t_new = -R_new @ C_new
        out.append((ts, R_new, t_new))
    return out


def plot_3d_traj(ax, col_C, otf_C, title, otf_label):
    ax.plot(col_C[:, 0], col_C[:, 1], col_C[:, 2], "b.-",
            label="COLMAP", linewidth=1.2, markersize=4)
    ax.plot(otf_C[:, 0], otf_C[:, 1], otf_C[:, 2], "r.-",
            label=otf_label, linewidth=1.2, markersize=4)
    # Start (●) and end (★) markers
    ax.scatter(col_C[0, 0], col_C[0, 1], col_C[0, 2], c="b", marker="o", s=80,
               edgecolor="black", linewidth=0.5, zorder=5)
    ax.scatter(col_C[-1, 0], col_C[-1, 1], col_C[-1, 2], c="b", marker="*",
               s=140, edgecolor="black", linewidth=0.5, zorder=5)
    ax.scatter(otf_C[0, 0], otf_C[0, 1], otf_C[0, 2], c="r", marker="o", s=80,
               edgecolor="black", linewidth=0.5, zorder=5)
    ax.scatter(otf_C[-1, 0], otf_C[-1, 1], otf_C[-1, 2], c="r", marker="*",
               s=140, edgecolor="black", linewidth=0.5, zorder=5)
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend(loc="best", fontsize=9)
    ax.set_title(title, fontsize=11)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--compare_root", default="compare")
    ap.add_argument("--multirun_root", default="results/multirun")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--raw_run", type=int, default=1,
                    help="Which run's raw OTF trajectory to use for "
                         "the before-alignment plot.")
    ap.add_argument("--out_dir", default="compare/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # Load shared COLMAP traj (use run 1's; should be identical across runs)
    col = read_tum(os.path.join(args.compare_root, "run_1", "colmap_traj.txt"))
    col_C = centres(col)

    # Load each run's compare outputs
    runs = []
    for i in range(1, args.runs + 1):
        d = os.path.join(args.compare_root, f"run_{i}")
        with open(os.path.join(d, "metrics.json")) as f:
            m = json.load(f)
        with open(os.path.join(d, "sim3.json")) as f:
            s3 = json.load(f)
        aligned = read_tum(os.path.join(d, "aligned_otf_traj.txt"))
        runs.append({
            "id": i,
            "metrics": m,
            "sim3": s3,
            "aligned": aligned,
            "aligned_C": centres(aligned),
        })

    # Raw OTF for the before-alignment plot (no Sim(3) applied)
    raw_traj_path = os.path.join(args.multirun_root,
                                 f"run_{args.raw_run}", "trajectory_otf.txt")
    otf_raw = read_tum(raw_traj_path)
    otf_raw_C = centres(otf_raw)

    # ----------- Figure 1: 2x2 -----------
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle("OTF vs COLMAP — Sim(3) alignment evidence", fontsize=14)

    ax00 = fig.add_subplot(2, 2, 1, projection="3d")
    plot_3d_traj(ax00, col_C, otf_raw_C,
                 "Before Sim(3) — different gauges",
                 "OTF (raw)")

    ax01 = fig.add_subplot(2, 2, 2, projection="3d")
    plot_3d_traj(ax01, col_C, runs[args.raw_run - 1]["aligned_C"],
                 "After Sim(3) — aligned",
                 f"OTF aligned (run {args.raw_run})")

    # Per-timestep ATE (mean ± std across runs)
    ate_per_step = np.stack([per_step_ate(r["aligned_C"], col_C) for r in runs])
    mean_ate = ate_per_step.mean(axis=0)
    std_ate = ate_per_step.std(axis=0)
    rmse_ate = np.sqrt((mean_ate ** 2).mean())

    ax10 = fig.add_subplot(2, 2, 3)
    x = np.arange(len(mean_ate))
    ax10.plot(x, mean_ate, "b-", linewidth=1.4, label="mean")
    ax10.fill_between(x, mean_ate - std_ate, mean_ate + std_ate,
                      alpha=0.3, color="b", label="±1σ")
    ax10.axhline(rmse_ate, color="k", linestyle="--", linewidth=1,
                 label=f"RMSE = {rmse_ate:.4f}")
    ax10.set_xlabel("timestep")
    ax10.set_ylabel("‖ΔC‖ (COLMAP units)")
    ax10.set_title(f"Per-timestep ATE ({args.runs}-run mean ± 1σ)",
                   fontsize=11)
    ax10.legend(loc="best", fontsize=9)
    ax10.grid(True, alpha=0.3)

    # Per-timestep RPE rot
    rpe_per_step = np.stack(
        [per_step_rpe_rot(r["aligned"], col) for r in runs]
    )
    mean_rpe = rpe_per_step.mean(axis=0)
    std_rpe = rpe_per_step.std(axis=0)
    rmse_rpe = np.sqrt((mean_rpe ** 2).mean())

    ax11 = fig.add_subplot(2, 2, 4)
    xr = np.arange(len(mean_rpe))
    ax11.plot(xr, mean_rpe, "g-", linewidth=1.4, label="mean")
    ax11.fill_between(xr, mean_rpe - std_rpe, mean_rpe + std_rpe,
                      alpha=0.3, color="g", label="±1σ")
    ax11.axhline(rmse_rpe, color="k", linestyle="--", linewidth=1,
                 label=f"RMSE = {rmse_rpe:.4f}°")
    ax11.set_xlabel("timestep")
    ax11.set_ylabel("Δ rot (deg)")
    ax11.set_title(f"Per-timestep RPE rotation ({args.runs}-run mean ± 1σ)",
                   fontsize=11)
    ax11.legend(loc="best", fontsize=9)
    ax11.grid(True, alpha=0.3)

    plt.tight_layout(rect=(0, 0, 1, 0.96))
    fig1_path = os.path.join(args.out_dir, "fig1_alignment.png")
    plt.savefig(fig1_path, dpi=160)
    plt.close()
    print(f"[fig1] {fig1_path}")

    # ----------- Figure 2: 1x2 -----------
    # (0) Path length comparison (computed first so suptitle can use the
    # measured ratio instead of a stale literal).
    otf_path = float(np.linalg.norm(np.diff(otf_raw_C, axis=0), axis=1).sum())
    col_path = float(np.linalg.norm(np.diff(col_C, axis=0), axis=1).sum())
    ratio = col_path / otf_path
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(
        f"Scale {ratio:.1f}× origin + N-run reproducibility",
        fontsize=14,
    )

    bars = ["OTF (raw, post-bootstrap gauge)",
            "COLMAP (own gauge)"]
    vals = [otf_path, col_path]
    cols = ["#3b6dc7", "#c74040"]
    y = np.arange(len(bars))
    ax0.barh(y, vals, color=cols)
    ax0.set_yticks(y); ax0.set_yticklabels(bars)
    ax0.invert_yaxis()
    for i, v in enumerate(vals):
        ax0.text(v, i, f"  {v:.3f}", va="center", fontsize=10)
    ax0.set_xlabel("Total path length over 23 timesteps")
    ax0.set_title(f"Why scale = {ratio:.2f}× (gauge origin)", fontsize=11)
    ax0.text(
        0.02, -0.32,
        "OTF gauge: bootstrap normalises consecutive median\n"
        "to 0.1 → total path ≈ 22 × 0.1 = 2.2 unit\n"
        "COLMAP gauge: own scale (e.g. first-pair baseline ≈ 1)\n"
        f"→ scale ratio = COLMAP / OTF = {col_path:.3f} / {otf_path:.3f} ≈ {ratio:.2f}",
        transform=ax0.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f8f8e0", edgecolor="#999"),
        verticalalignment="top",
    )

    # (1) N-run consistency: scale, ATE_RMSE, RPE_rot_RMSE
    metrics_arr = {
        "Sim(3) scale": [r["sim3"]["s"] for r in runs],
        "ATE RMSE": [r["metrics"]["ate_rmse"] for r in runs],
        "RPE rot RMSE [°]": [r["metrics"]["rpe_rot_rmse_deg"] for r in runs],
    }
    labels = list(metrics_arr.keys())
    data = [metrics_arr[k] for k in labels]
    means = [np.mean(d) for d in data]
    stds = [np.std(d) for d in data]
    norm_data = [(np.array(d) - m) / max(m, 1e-12) for d, m in zip(data, means)]

    bp = ax1.boxplot(norm_data, labels=labels, showmeans=True,
                     widths=0.5, patch_artist=True)
    for patch, c in zip(bp["boxes"], ["#a8c8ff", "#ffb380", "#a8e0a8"]):
        patch.set_facecolor(c)
    ax1.axhline(0, color="k", linewidth=0.6, alpha=0.5)
    ax1.set_ylabel("(value − mean) / mean   (relative deviation)")
    ax1.set_title(f"{args.runs}-run consistency (relative)", fontsize=11)
    ax1.grid(True, alpha=0.3, axis="y")
    summary_lines = []
    for k, m, s, d in zip(labels, means, stds, data):
        rel = s / max(m, 1e-12)
        summary_lines.append(f"{k}: {m:.6f} ± {s:.2e}  (rel σ={100*rel:.3f}%)")
    ax1.text(
        0.02, -0.18,
        "\n".join(summary_lines),
        transform=ax1.transAxes,
        fontsize=9,
        bbox=dict(boxstyle="round", facecolor="#f8f8e0", edgecolor="#999"),
        verticalalignment="top",
        family="monospace",
    )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    fig2_path = os.path.join(args.out_dir, "fig2_scale_origin.png")
    plt.savefig(fig2_path, dpi=160)
    plt.close()
    print(f"[fig2] {fig2_path}")

    # ----------- summary.md -----------
    scene_scale = float(np.linalg.norm(col_C - col_C.mean(axis=0), axis=1).mean())
    ate_pct = 100 * rmse_ate / max(scene_scale, 1e-12)
    md_path = os.path.join(args.out_dir, "summary.md")
    with open(md_path, "w") as f:
        f.write(
            "# 4/16 미팅 후속 — Sim(3) 정렬로 OTF=COLMAP 입증\n\n"
            f"- 정렬 후 ATE RMSE = **{rmse_ate:.4f}** "
            f"(COLMAP scene scale {scene_scale:.3f} 대비 **{ate_pct:.2f}%**)\n"
            f"- 정렬 후 RPE rot RMSE = **{rmse_rpe:.3f}°**\n"
            f"- Scale {ratio:.2f}× = OTF의 bootstrap gauge (consecutive "
            "median = 0.1)와 COLMAP 자체 gauge의 차이일 뿐\n"
            f"- N={args.runs} run에서 Sim(3) scale 변동 = "
            f"**{stds[0]:.2e}** (relative σ {100*stds[0]/means[0]:.3f}%) "
            "→ BA gauge 결정성 입증\n\n"
            "## Figure 1 — Sim(3) alignment evidence\n\n"
            "![fig1](fig1_alignment.png)\n\n"
            "## Figure 2 — Scale origin + N-run reproducibility\n\n"
            "![fig2](fig2_scale_origin.png)\n\n"
            "## Per-run metrics\n\n"
            "| run | Sim(3) scale | ATE RMSE | RPE rot RMSE (°) |\n"
            "|---:|---:|---:|---:|\n"
        )
        for r in runs:
            f.write(
                f"| {r['id']} | {r['sim3']['s']:.6f} | "
                f"{r['metrics']['ate_rmse']:.6f} | "
                f"{r['metrics']['rpe_rot_rmse_deg']:.6f} |\n"
            )
    print(f"[summary] {md_path}")


if __name__ == "__main__":
    sys.exit(main() or 0)
