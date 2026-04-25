"""Build the 4/16 follow-up runtime + qualitative-render report.

Reads results/report/iter100/runtimes.json and results/report/iter500/runtimes.json
plus the per-keyframe render images from results/report/iter500/render_eval/
and writes:

  results/report/runtime.md           comparison table + analysis
  results/report/figures/render_grid.png         4×4 train + holdout
  results/report/figures/render_structural.png   1×4 Cam02 vs Cam01 (optional)

Usage:
  python tools/render_report.py
"""

import argparse
import json
import os
import statistics
import sys

import cv2
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_runtime(path):
    with open(path) as f:
        return json.load(f)


def fmt_ms(seconds):
    return f"{1000 * seconds:.1f}"


def runtime_table_md(rA, rB):
    sA = rA["stages"]
    sB = rB["stages"]
    arr_A = np.array(rA["incremental_ts_times_s"])
    arr_B = np.array(rB["incremental_ts_times_s"])

    def per_kf_stats(arr):
        if len(arr) == 0:
            return ("—", "—", "—")
        if len(arr) == 1:
            return (f"{1000*arr[0]:.0f}", f"{1000*arr[0]:.0f}", "0")
        return (
            f"{1000*arr.mean():.0f}",
            f"{1000*float(np.median(arr)):.0f}",
            f"{1000*float(arr.std()):.0f}",
        )

    avg_A, med_A, std_A = per_kf_stats(arr_A)
    avg_B, med_B, std_B = per_kf_stats(arr_B)

    def stage_row(key):
        a_total = sA.get(key, {}).get("total_s", 0)
        a_n = sA.get(key, {}).get("n_calls", 0)
        b_total = sB.get(key, {}).get("total_s", 0)
        b_n = sB.get(key, {}).get("n_calls", 0)
        a_pct = 100 * a_total / max(rA["wall_clock_s"], 1e-12)
        b_pct = 100 * b_total / max(rB["wall_clock_s"], 1e-12)
        a_str = (f"{a_total:.2f}s ({a_pct:.1f}%, n={a_n})"
                 if a_n > 0 else "—")
        b_str = (f"{b_total:.2f}s ({b_pct:.1f}%, n={b_n})"
                 if b_n > 0 else "—")
        return f"| {key} | {a_str} | {b_str} |"

    rows = [
        f"| Wall clock | {rA['wall_clock_s']:.2f}s | {rB['wall_clock_s']:.2f}s |",
        f"| Bootstrap (BAB) | "
        f"{sA.get('BAB',{}).get('total_s',0):.2f}s "
        f"({sA.get('BAB',{}).get('n_calls',0)}회) | "
        f"{sB.get('BAB',{}).get('total_s',0):.2f}s "
        f"({sB.get('BAB',{}).get('n_calls',0)}회) |",
        f"| Bootstrap full (incl. matching) | "
        f"{rA['bootstrap_full_s']:.2f}s | {rB['bootstrap_full_s']:.2f}s |",
        f"| Per-keyframe avg | {avg_A} ms | {avg_B} ms |",
        f"| Per-keyframe median | {med_A} ms | {med_B} ms |",
        f"| Per-keyframe std | {std_A} ms | {std_B} ms |",
    ]
    for k in ["Load", "tri", "BAI", "Add", "Init", "Opt", "anc"]:
        rows.append(stage_row(k))
    rows.extend([
        f"| Peak GPU mem | {rA['gpu_peak_gb']:.2f} GB | {rB['gpu_peak_gb']:.2f} GB |",
        f"| Final n_gauss | {rA['n_gauss_final']:,} | {rB['n_gauss_final']:,} |",
    ])
    return "\n".join(rows)


def stage_share_analysis(rA, rB):
    """Return Markdown explaining the dominant stages."""
    out = []
    for tag, r in [("iter=100 (Run A)", rA), ("iter=500 (Run B)", rB)]:
        wall = r["wall_clock_s"]
        shares = sorted(
            (
                (k, v["total_s"], 100 * v["total_s"] / max(wall, 1e-12))
                for k, v in r["stages"].items() if v["n_calls"] > 0
            ),
            key=lambda x: x[1], reverse=True,
        )
        top = ", ".join(
            f"{k} {pct:.1f}%" for k, _t, pct in shares[:3]
        )
        out.append(f"- **{tag}** wall={wall:.1f}s — top 3 stages: {top}")
    return "\n".join(out)


def per_kf_curve(arr_A, arr_B, out_path):
    fig, ax = plt.subplots(figsize=(10, 4))
    if len(arr_A) > 0:
        ax.plot(np.arange(len(arr_A)), 1000 * np.array(arr_A),
                "b.-", label=f"iter=100 (n={len(arr_A)})")
    if len(arr_B) > 0:
        ax.plot(np.arange(len(arr_B)), 1000 * np.array(arr_B),
                "r.-", label=f"iter=500 (n={len(arr_B)})")
    ax.set_xlabel("incremental timestep index")
    ax.set_ylabel("wall clock (ms)")
    ax.set_title("Per-incremental-timestep wall time")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140)
    plt.close()


def load_render_panel(eval_dir, view, ts):
    """Find a render_eval file matching the (view, ts) and load it.

    Files are saved as side-by-side `rendered | gt` so we split them.
    Returns (rendered_rgb, gt_rgb) as uint8 H×W×3 arrays in RGB order.
    """
    name = f"{view}__ts{ts:05d}.png"
    p = os.path.join(eval_dir, name)
    if not os.path.exists(p):
        # eval images use raw key without extension fallback
        name2 = f"{view}__ts{ts:05d}"
        p = os.path.join(eval_dir, name2 + ".png")
        if not os.path.exists(p):
            return None, None
    img = cv2.imread(p, cv2.IMREAD_COLOR)
    if img is None:
        return None, None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    H, W2, _ = img.shape
    W = W2 // 2
    rendered = img[:, :W]
    gt = img[:, W:]
    return rendered, gt


def load_metrics(eval_dir):
    p = os.path.join(eval_dir, "metrics.json")
    if not os.path.exists(p):
        return None
    with open(p) as f:
        return json.load(f)


def panel_psnr(metrics, view, ts):
    if metrics is None:
        return None
    name = f"{view}__ts{ts:05d}.png"
    for entry in metrics.get("per_frame", []):
        if entry["name"] == name:
            return entry["psnr"]
    return None


def build_render_grid(model_dir, out_path):
    """4×4 grid: rows of train rendered/GT (top) and holdout rendered/GT (bot)."""
    eval_dir = os.path.join(model_dir, "render_eval")
    metrics = load_metrics(eval_dir)

    train_views = ["High_Cam07", "High_Cam06", "Low_Cam08", "Low_Cam07"]
    train_ts = [0, 1, 2, 3]
    holdout_view = "High_Cam01"
    holdout_ts = [18, 19, 20, 22]

    fig, axes = plt.subplots(4, 4, figsize=(20, 16))
    fig.suptitle(
        "Rendering Quality — train (top 2 rows) vs holdout (bottom 2 rows)",
        fontsize=14,
    )

    train_psnrs = []
    holdout_psnrs = []

    # rows 1-2: train rendered, train GT
    for col, (v, ts) in enumerate(zip(train_views, train_ts)):
        rendered, gt = load_render_panel(eval_dir, v, ts)
        psnr = panel_psnr(metrics, v, ts)
        if psnr is not None:
            train_psnrs.append(psnr)
        for row, img in [(0, rendered), (1, gt)]:
            ax = axes[row, col]
            if img is not None:
                ax.imshow(img)
            ax.set_axis_off()
            label = "train" if row == 0 else "GT"
            title = f"ts={ts}, {v}, {label}"
            if row == 0 and psnr is not None:
                title += f"  (PSNR {psnr:.2f})"
            ax.set_title(title, fontsize=10)

    # rows 3-4: holdout rendered, holdout GT
    for col, ts in enumerate(holdout_ts):
        rendered, gt = load_render_panel(eval_dir, holdout_view, ts)
        psnr = panel_psnr(metrics, holdout_view, ts)
        if psnr is not None:
            holdout_psnrs.append(psnr)
        for row, img in [(2, rendered), (3, gt)]:
            ax = axes[row, col]
            if img is not None:
                ax.imshow(img)
            ax.set_axis_off()
            label = "holdout" if row == 2 else "GT"
            title = f"ts={ts}, {holdout_view}, {label}"
            if row == 2 and psnr is not None:
                title += f"  (PSNR {psnr:.2f})"
            ax.set_title(title, fontsize=10)

    plt.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(out_path, dpi=130)
    plt.close()

    return {
        "train_psnr_mean": (sum(train_psnrs) / len(train_psnrs)
                            if train_psnrs else None),
        "holdout_psnr_mean": (sum(holdout_psnrs) / len(holdout_psnrs)
                              if holdout_psnrs else None),
        "n_train_panels": len(train_psnrs),
        "n_holdout_panels": len(holdout_psnrs),
    }


def build_structural_example(model_dir, out_path, ts=10):
    """1×4 row: Cam02 rendered/GT + Cam01 rendered/GT at the same ts."""
    eval_dir = os.path.join(model_dir, "render_eval")
    metrics = load_metrics(eval_dir)
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    fig.suptitle(
        f"Structural-limit example — Cam02 (train, underdetermined seed) "
        f"vs Cam01 (train, normal) at ts={ts}",
        fontsize=12,
    )
    panels = [
        ("High_Cam02", ts, "rendered"),
        ("High_Cam02", ts, "GT"),
        ("High_Cam06", ts, "rendered"),
        ("High_Cam06", ts, "GT"),
    ]
    for ax, (view, ts_, label) in zip(axes, panels):
        rendered, gt = load_render_panel(eval_dir, view, ts_)
        img = rendered if label == "rendered" else gt
        if img is not None:
            ax.imshow(img)
        ax.set_axis_off()
        psnr = panel_psnr(metrics, view, ts_)
        title = f"{view}, {label}"
        if label == "rendered" and psnr is not None:
            title += f" (PSNR {psnr:.2f})"
        ax.set_title(title, fontsize=10)
    plt.tight_layout(rect=(0, 0, 1, 0.93))
    plt.savefig(out_path, dpi=130)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report_root", default="results/report")
    ap.add_argument("--out_dir", default="results/report/figures")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    rA = load_runtime(os.path.join(args.report_root, "iter100", "runtimes.json"))
    rB = load_runtime(os.path.join(args.report_root, "iter500", "runtimes.json"))

    table = runtime_table_md(rA, rB)
    analysis = stage_share_analysis(rA, rB)

    per_kf_path = os.path.join(args.out_dir, "per_kf_curve.png")
    per_kf_curve(rA["incremental_ts_times_s"], rB["incremental_ts_times_s"],
                 per_kf_path)

    # 4×4 render grid + optional structural example from Run B
    grid_path = os.path.join(args.out_dir, "render_grid.png")
    grid_stats = build_render_grid(
        os.path.join(args.report_root, "iter500"), grid_path,
    )
    structural_path = os.path.join(args.out_dir, "render_structural.png")
    build_structural_example(os.path.join(args.report_root, "iter500"),
                             structural_path)

    # Markdown report
    md_path = os.path.join(args.report_root, "runtime.md")
    with open(md_path, "w") as f:
        f.write("# Runtime + 정성 평가 보고서 (4/16 후속)\n\n")
        f.write("## 1. Runtime 비교 — Run A (iter=100) vs Run B (iter=500)\n\n")
        f.write("| 단계 | iter=100 (Run A) | iter=500 (Run B) |\n")
        f.write("|---|---|---|\n")
        f.write(table + "\n\n")
        f.write("### Stage 비중 분석\n\n")
        f.write(analysis + "\n\n")
        f.write(f"![per-kf curve]({os.path.relpath(per_kf_path, args.report_root)})\n\n")
        f.write("Per-incremental-timestep wall clock — timestep 인덱스에 따른 누적 비용\n\n")
        f.write("## 2. 정성 평가 (Run B = iter=500 결과)\n\n")
        f.write(f"- 전체 train (n=184) avg PSNR: **{rB.get('post_hoc_train_psnr', '21.36 (from log)')}** \n")
        f.write(f"- 전체 holdout (n=23) avg PSNR: **{rB.get('post_hoc_holdout_psnr', '19.83 (from log)')}** \n")
        if grid_stats["train_psnr_mean"] is not None:
            f.write(f"- 그리드 panel avg (sampled, 데이터셋 평균은 아님): "
                    f"train {grid_stats['train_psnr_mean']:.2f} "
                    f"(n={grid_stats['n_train_panels']}), "
                    f"holdout {grid_stats['holdout_psnr_mean']:.2f} "
                    f"(n={grid_stats['n_holdout_panels']})\n\n")
        f.write(f"![grid]({os.path.relpath(grid_path, args.report_root)})\n\n")
        f.write(f"![structural]({os.path.relpath(structural_path, args.report_root)})\n\n")
        f.write("## 3. Phase 2 속도 최적화 우선순위\n\n")
        f.write("- iter=100에서 **Opt 50.1% + Add 20.1%**로 incremental optimization 단계가 70% 차지. "
                "Phase 2 1차 표적 확정.\n")
        f.write("- iter=500에서는 **Opt 79.3%**로 거의 압도적. Opt 단계의 tile rasterizer + "
                "gradient buffer 개선이 직접적 wall clock 감소.\n")
        f.write("- BAB (bootstrap)는 iter=100에서 11.9%, iter=500에서 4.7%. "
                "Bootstrap은 1회 비용이라 학습 길어질수록 비중 줄어듦.\n")
        f.write("- 23 ts × 9 view 환경에서 BAI(incremental BA)는 3.8–7.2%. "
                "Add(keyframe 추가, 9-view spawn 포함)는 6.7–20.1%. "
                "두 stage 모두 Opt에 비해 marginal target.\n\n")
        f.write("## 4. 부수 관찰\n\n")
        f.write("- Run B의 n_gauss = 0.40M < Run A의 1.23M. iter=500에서 photometric "
                "prune이 더 적극 작동 + anchor가 1→3개로 분할 → 한 anchor당 점이 적어짐.\n")
        f.write("- Per-keyframe std: iter=100 406 ms vs iter=500 2,182 ms. iter↑일수록 "
                "timestep마다 비용 변동 ↑. anchor 분할 + densification timing 의존.\n")
        f.write("- GPU peak 8.99 GB (iter=100) vs 8.15 GB (iter=500). 적은 Gaussian으로 "
                "더 작은 메모리 — densification/prune이 잘 균형.\n")
    print(f"[md] {md_path}")
    print(f"[grid] {grid_path}  (train={grid_stats['train_psnr_mean']}, "
          f"holdout={grid_stats['holdout_psnr_mean']})")
    print(f"[structural] {structural_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
