"""Aggregate Stage 1-3 ablation outputs into tables + JSON for SUMMARY.md.

Inputs (under --root, default results/feedback_round1/):
  run_A/, run_B/, run_C/, run_D/        Stage 1: 4 ablation runs
    runtimes.json, render_eval/metrics.json, trajectory_otf.txt
  run_A_pnp_stats.json                  Stage 2 + 3: per-ts × per-view stats
                                        (mean_residual_px, n_inliers, ew, …)

Outputs (under --root):
  ablation_lr_poses.json                Stage 1 4-cell matrix
  per_view_pnp_stats.json               Stage 2 view-level summary
  seam_residual_analysis.json           Stage 3 seam-distance × residual
  SUMMARY.md                            human-readable rollup

Stage-1 metrics:
  - train_psnr / holdout_psnr / all_psnr derived by splitting per_frame on
    is_test (is_test=True == holdout in --rig_holdout_view path).
  - ATE / RPE rot / Sim(3) scale: invokes tools/compare_with_colmap.py per
    run if compare/<run>/metrics.json doesn't already exist.

Run from on-the-fly-nvs/ root.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from typing import Dict, List, Optional

import numpy as np


# ---------------------------------------------------------------------------
# rig geometry helper for Stage 3 seam azimuth.
#
# blender_rig.json schema: list of tiers (High / Low), each with `cameras`,
# each camera has `angle_offset` (radians) — its azimuthal placement around
# the rig vertical axis. We compute each view's azimuth relative to ref view
# and the wrap-around distance to the EQR stitching seam (seam at 180°
# from ref forward, i.e. directly behind the panorama centre).
# ---------------------------------------------------------------------------
def load_view_azimuth_offsets(rig_config_path: str) -> Dict[str, float]:
    """Returns {view_name: angle_offset_rad} from blender_rig.json."""
    with open(rig_config_path) as fh:
        rig = json.load(fh)
    out: Dict[str, float] = {}
    if isinstance(rig, list):
        tiers = rig
    elif isinstance(rig, dict) and "rigs" in rig:
        tiers = rig["rigs"]
    else:
        tiers = [rig]
    for tier in tiers:
        rot_off = float(tier.get("rotation_offset", 0.0))
        for cam in tier.get("cameras", []):
            name = cam["name"]
            out[name] = float(cam["angle_offset"]) + rot_off
    return out


def seam_azimuth_distance_deg(view_az_rad: float, ref_az_rad: float) -> float:
    """Wrap-around angular distance to the seam.
    Seam is at +π from ref forward (directly opposite, at panorama back).
    """
    rel = (view_az_rad - ref_az_rad) % (2 * math.pi)
    seam_offset = math.pi
    d = abs(rel - seam_offset)
    d = min(d, 2 * math.pi - d)
    return math.degrees(d)


def view_azimuth_relative_deg(view_az_rad: float, ref_az_rad: float) -> float:
    """View azimuth in degrees, with ref at 0° (range -180..180)."""
    rel = (view_az_rad - ref_az_rad + math.pi) % (2 * math.pi) - math.pi
    return math.degrees(rel)


# ---------------------------------------------------------------------------
# Stage-1 helpers
# ---------------------------------------------------------------------------
def split_metrics(per_frame: List[dict]) -> Dict[str, Dict[str, float]]:
    train, holdout = [], []
    for f in per_frame:
        tgt = holdout if f.get("is_test") else train
        tgt.append(f)
    out = {}
    for name, group in [("train", train), ("holdout", holdout), ("all", per_frame)]:
        if not group:
            out[name] = {"psnr": float("nan"), "ssim": float("nan"),
                        "lpips": float("nan"), "n": 0}
            continue
        out[name] = {
            "psnr": float(np.mean([f["psnr"] for f in group])),
            "ssim": float(np.mean([f["ssim"] for f in group])),
            "lpips": float(np.mean([f["lpips"] for f in group])),
            "n": len(group),
        }
    return out


def maybe_run_compare(run_dir: str, root: str, colmap_dir: str, ref_view: str
                     ) -> Optional[Dict]:
    """Invoke tools/compare_with_colmap.py if compare metrics are missing."""
    out_dir = os.path.join("compare", os.path.basename(run_dir))
    metrics_path = os.path.join(out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        with open(metrics_path) as fh:
            return json.load(fh)
    otf_traj = os.path.join(run_dir, "trajectory_otf.txt")
    if not os.path.exists(otf_traj):
        return None
    cmd = [
        sys.executable, "tools/compare_with_colmap.py",
        "--otf_traj", otf_traj,
        "--colmap_dir", colmap_dir,
        "--ref_view", ref_view,
        "--out", out_dir,
    ]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True)
    except subprocess.CalledProcessError as exc:
        print(f"[warn] compare failed for {run_dir}: {exc.stderr}")
        return None
    if not os.path.exists(metrics_path):
        return None
    with open(metrics_path) as fh:
        return json.load(fh)


def collect_stage1(root: str, colmap_dir: str, ref_view: str,
                  run_specs: List[Dict]) -> List[Dict]:
    rows = []
    for spec in run_specs:
        run_id = spec["id"]
        run_dir = os.path.join(root, f"run_{run_id}")
        row = {"run": run_id, **spec}
        # render_eval
        metrics_p = os.path.join(run_dir, "render_eval", "metrics.json")
        if os.path.exists(metrics_p):
            with open(metrics_p) as fh:
                m = json.load(fh)
            row["render_split"] = split_metrics(m.get("per_frame", []))
        else:
            row["render_split"] = None
        # runtimes
        rt_p = os.path.join(run_dir, "runtimes.json")
        if os.path.exists(rt_p):
            with open(rt_p) as fh:
                rt = json.load(fh)
            row["wall_clock_s"] = rt.get("wall_clock_s", float("nan"))
        else:
            row["wall_clock_s"] = float("nan")
        # compare
        cmp_metrics = maybe_run_compare(run_dir, root, colmap_dir, ref_view)
        if cmp_metrics is not None:
            row["ate_rmse_m"] = cmp_metrics.get("ate_rmse", float("nan"))
            row["rpe_rot_rmse_deg"] = cmp_metrics.get("rpe_rot_rmse_deg",
                                                     float("nan"))
            row["sim3_scale"] = cmp_metrics.get("sim3_scale", float("nan"))
        else:
            row["ate_rmse_m"] = float("nan")
            row["rpe_rot_rmse_deg"] = float("nan")
            row["sim3_scale"] = float("nan")
        rows.append(row)
    return rows


# ---------------------------------------------------------------------------
# Stage-2 / Stage-3 from PnP stats
# ---------------------------------------------------------------------------
def collect_stage2_3(pnp_stats_path: str, rig_config_path: str,
                    ref_view: str) -> Dict:
    with open(pnp_stats_path) as fh:
        data = json.load(fh)
    per_ts = data.get("per_ts", [])
    if not per_ts:
        return {"stage2": [], "stage3": []}

    # collect view names from first record
    view_names = list(per_ts[0]["views"].keys())

    # aggregate across timesteps
    agg: Dict[str, Dict[str, list]] = {
        v: {"n_corr": [], "n_inl": [], "ew": [], "succ": [], "res": []}
        for v in view_names
    }
    for rec in per_ts:
        for v, s in rec["views"].items():
            agg[v]["n_corr"].append(s["n_correspondences"])
            agg[v]["n_inl"].append(s["n_inliers"])
            agg[v]["ew"].append(s["effective_weight"])
            agg[v]["succ"].append(1 if s["success"] else 0)
            r = s.get("mean_residual_px", float("nan"))
            if math.isfinite(r):
                agg[v]["res"].append(r)

    # rig geometry: load per-view azimuth offsets and pick ref.
    az_offsets = load_view_azimuth_offsets(rig_config_path)
    ref_az = az_offsets.get(ref_view, 0.0)

    stage2_rows = []
    stage3_rows = []
    for v in view_names:
        a = agg[v]
        n_ts = len(a["succ"])
        succ_rate = float(np.mean(a["succ"]))
        mean_inl = float(np.mean(a["n_inl"])) if a["n_inl"] else 0.0
        mean_corr = float(np.mean(a["n_corr"])) if a["n_corr"] else 0.0
        mean_ew = float(np.mean(a["ew"])) if a["ew"] else 0.0
        mean_res = float(np.mean(a["res"])) if a["res"] else float("nan")
        std_res = float(np.std(a["res"])) if len(a["res"]) > 1 else 0.0

        stage2_rows.append({
            "view": v,
            "n_ts": n_ts,
            "success_rate": succ_rate,
            "mean_n_correspondences": mean_corr,
            "mean_n_inliers": mean_inl,
            "mean_effective_weight": mean_ew,
        })

        if v in az_offsets:
            view_az = view_azimuth_relative_deg(az_offsets[v], ref_az)
            seam_d = seam_azimuth_distance_deg(az_offsets[v], ref_az)
        else:
            view_az = float("nan")
            seam_d = float("nan")
        stage3_rows.append({
            "view": v,
            "view_azimuth_deg": view_az,
            "seam_azimuth_distance_deg": seam_d,
            "mean_residual_px": mean_res,
            "residual_std_px": std_res,
            "n_residual_samples": len(a["res"]),
        })

    return {"stage2": stage2_rows, "stage3": stage3_rows}


# ---------------------------------------------------------------------------
# Markdown emit
# ---------------------------------------------------------------------------
def fmt(x, p=3):
    if x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x))):
        return "—"
    if isinstance(x, float):
        if abs(x) < 0.01:
            return f"{x:.{p+1}f}"
        return f"{x:.{p}f}"
    return str(x)


def emit_summary(out_path: str, stage1_rows: List[Dict],
                stage2_rows: List[Dict], stage3_rows: List[Dict]):
    L = []
    L.append("# Feedback round 1 — verification ablation\n")
    L.append("Generated by `tools/analyze_feedback_round1.py`. "
             "Source data: `results/feedback_round1/`.\n")

    L.append("## Stage 1 — lr_poses ablation (4-cell matrix)\n")
    L.append("Same dataset, iter=100, --rig_holdout_view High_Cam01.\n")
    L.append("| run | lr_poses | refine | wall (s) | train PSNR | "
             "holdout PSNR | all PSNR | ATE (m) | RPE rot (°) | Sim3 scale |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for row in stage1_rows:
        rs = row.get("render_split") or {}
        train_psnr = (rs.get("train") or {}).get("psnr", float("nan"))
        hold_psnr = (rs.get("holdout") or {}).get("psnr", float("nan"))
        all_psnr = (rs.get("all") or {}).get("psnr", float("nan"))
        L.append(
            f"| {row['run']} | {row['lr_poses']} | "
            f"{'on' if row['refine'] else 'off'} | "
            f"{fmt(row.get('wall_clock_s'), 2)} | "
            f"{fmt(train_psnr, 2)} | {fmt(hold_psnr, 2)} | "
            f"{fmt(all_psnr, 2)} | {fmt(row.get('ate_rmse_m'), 4)} | "
            f"{fmt(row.get('rpe_rot_rmse_deg'), 3)} | "
            f"{fmt(row.get('sim3_scale'), 3)} |"
        )
    L.append("")
    # delta block
    by_id = {r["run"]: r for r in stage1_rows}

    def get_metric(rid, key):
        r = by_id.get(rid)
        if r is None:
            return float("nan")
        if key == "train_psnr":
            return ((r.get("render_split") or {}).get("train") or {}).get("psnr", float("nan"))
        if key == "holdout_psnr":
            return ((r.get("render_split") or {}).get("holdout") or {}).get("psnr", float("nan"))
        return r.get(key, float("nan"))

    def delta(a, b, key):
        va = get_metric(a, key)
        vb = get_metric(b, key)
        if not (math.isfinite(va) and math.isfinite(vb)):
            return float("nan")
        return va - vb

    L.append("### Channel decomposition (vs A baseline)\n")
    L.append("| comparison | meaning | Δ holdout PSNR | Δ ATE (m) | Δ RPE rot (°) |")
    L.append("|---|---|---|---|---|")
    pairs = [
        ("A", "B", "lr_poses single channel (off → on)"),
        ("A", "C", "refinement single channel (off → on)"),
        ("A", "D", "both channels (off → on)"),
    ]
    for a, b, desc in pairs:
        L.append(
            f"| {a} − {b} | {desc} | "
            f"{fmt(delta(a, b, 'holdout_psnr'), 3)} | "
            f"{fmt(delta(a, b, 'ate_rmse_m'), 4)} | "
            f"{fmt(delta(a, b, 'rpe_rot_rmse_deg'), 3)} |"
        )
    L.append("")

    # Stage 1 conclusion line (quick numeric summary)
    if stage1_rows:
        d_b = delta("A", "B", "holdout_psnr")
        d_c = delta("A", "C", "holdout_psnr")
        d_d = delta("A", "D", "holdout_psnr")
        biggest = max(
            [("lr_poses", d_b), ("refinement", d_c), ("both", d_d)],
            key=lambda kv: abs(kv[1]) if math.isfinite(kv[1]) else -1,
        )
        L.append(f"**Conclusion (Stage 1):** disabling {biggest[0]} drops "
                 f"holdout PSNR by {fmt(biggest[1], 2)} dB. "
                 f"Per-channel: lr_poses Δ={fmt(d_b, 2)}, "
                 f"refinement Δ={fmt(d_c, 2)}, both Δ={fmt(d_d, 2)}.\n")

    L.append("## Stage 2 — per-view PnP inlier / effective weight\n")
    L.append("Run A (full system) per-timestep × per-view aggregation, "
             "averaged across rig timesteps.\n")
    L.append("| view | n_ts | success rate | mean n_corr | "
             "mean n_inliers | mean effective_weight |")
    L.append("|---|---|---|---|---|---|")
    for r in stage2_rows:
        L.append(
            f"| {r['view']} | {r['n_ts']} | "
            f"{fmt(r['success_rate'], 3)} | "
            f"{fmt(r['mean_n_correspondences'], 1)} | "
            f"{fmt(r['mean_n_inliers'], 1)} | "
            f"{fmt(r['mean_effective_weight'], 3)} |"
        )
    L.append("")

    # Stage 2 conclusion: pick view with lowest contribution
    if stage2_rows:
        ews = [(r["view"], r["mean_effective_weight"]) for r in stage2_rows]
        lo = min(ews, key=lambda x: x[1])
        hi = max(ews, key=lambda x: x[1])
        succ_min = min(stage2_rows, key=lambda r: r["success_rate"])
        L.append(f"**Conclusion (Stage 2):** lowest-contribution view is "
                 f"{lo[0]} (mean ew={fmt(lo[1], 3)}); highest is "
                 f"{hi[0]} ({fmt(hi[1], 3)}). Success rates: "
                 f"min={fmt(succ_min['success_rate'], 3)} "
                 f"@ {succ_min['view']}.\n")

    L.append("## Stage 3 — seam-adjacent residual\n")
    L.append("Reprojection residual (final rig_pose) per view vs azimuthal "
             "distance to EQR seam (assumed at 180° from ref forward).\n")
    L.append("| view | view azimuth (°) | seam distance (°) | "
             "mean residual (px) | std (px) | n samples |")
    L.append("|---|---|---|---|---|---|")
    for r in sorted(stage3_rows, key=lambda x: x["seam_azimuth_distance_deg"]
                    if math.isfinite(x["seam_azimuth_distance_deg"]) else 1e9):
        L.append(
            f"| {r['view']} | {fmt(r['view_azimuth_deg'], 1)} | "
            f"{fmt(r['seam_azimuth_distance_deg'], 1)} | "
            f"{fmt(r['mean_residual_px'], 3)} | "
            f"{fmt(r['residual_std_px'], 3)} | "
            f"{r['n_residual_samples']} |"
        )
    L.append("")

    # Stage 3 conclusion: correlation between seam distance and residual
    if stage3_rows:
        finite = [r for r in stage3_rows
                  if math.isfinite(r["seam_azimuth_distance_deg"])
                  and math.isfinite(r["mean_residual_px"])]
        if len(finite) >= 3:
            ds = np.array([r["seam_azimuth_distance_deg"] for r in finite])
            rs = np.array([r["mean_residual_px"] for r in finite])
            if ds.std() > 1e-6 and rs.std() > 1e-6:
                corr = float(np.corrcoef(ds, rs)[0, 1])
            else:
                corr = float("nan")
            # Also compare seam-near (dist < 45°) vs seam-far (dist >= 45°).
            near = [r for r in finite if r["seam_azimuth_distance_deg"] < 45]
            far = [r for r in finite if r["seam_azimuth_distance_deg"] >= 45]
            if near and far:
                near_mean = float(np.mean([r["mean_residual_px"] for r in near]))
                far_mean = float(np.mean([r["mean_residual_px"] for r in far]))
                rel = (near_mean - far_mean) / max(far_mean, 1e-9) * 100
                L.append(
                    f"**Conclusion (Stage 3):** Pearson r(seam_dist, residual) "
                    f"= {fmt(corr, 3)}. Seam-near (<45°) mean residual "
                    f"{fmt(near_mean, 2)} px vs seam-far (≥45°) "
                    f"{fmt(far_mean, 2)} px ({fmt(rel, 1)}% diff)."
                )
            else:
                L.append(f"**Conclusion (Stage 3):** Pearson r(seam_dist, "
                         f"residual) = {fmt(corr, 3)} "
                         f"(insufficient bins for near/far split).")

    with open(out_path, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="results/feedback_round1")
    p.add_argument("--colmap_dir",
                   default="/opt/ftp/files/260411/colmap_result/sparse/0")
    p.add_argument("--ref_view", default="High_Cam07")
    p.add_argument("--rig_config",
                   default="/opt/ftp/files/260411/blender_rig.json")
    args = p.parse_args()

    run_specs = [
        {"id": "A", "lr_poses": "1e-4", "refine": True},
        {"id": "B", "lr_poses": "0", "refine": True},
        {"id": "C", "lr_poses": "1e-4", "refine": False},
        {"id": "D", "lr_poses": "0", "refine": False},
    ]
    stage1 = collect_stage1(args.root, args.colmap_dir, args.ref_view,
                            run_specs)

    pnp_path = os.path.join(args.root, "run_A_pnp_stats.json")
    if os.path.exists(pnp_path):
        s23 = collect_stage2_3(pnp_path, args.rig_config, args.ref_view)
    else:
        print(f"[warn] missing {pnp_path}; Stage 2/3 sections will be empty")
        s23 = {"stage2": [], "stage3": []}

    with open(os.path.join(args.root, "ablation_lr_poses.json"), "w") as fh:
        json.dump(stage1, fh, indent=2)
    with open(os.path.join(args.root, "per_view_pnp_stats.json"), "w") as fh:
        json.dump(s23["stage2"], fh, indent=2)
    with open(os.path.join(args.root, "seam_residual_analysis.json"), "w") as fh:
        json.dump(s23["stage3"], fh, indent=2)
    emit_summary(os.path.join(args.root, "SUMMARY.md"),
                 stage1, s23["stage2"], s23["stage3"])


if __name__ == "__main__":
    main()
