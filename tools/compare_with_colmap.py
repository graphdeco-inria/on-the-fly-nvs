"""Compare on-the-fly-nvs-rig output with COLMAP rig-constrained result.

Both pipelines have arbitrary world gauge (scale + rotation + origin), so
direct subtraction is meaningless. We align with a 7-DoF Sim(3) using
camera centres (Umeyama 1991) and report ATE / RPE.

Inputs:
  --otf_traj      results/<run>/trajectory_otf.txt   (TUM, dumped by train.py)
  --colmap_dir    /opt/ftp/files/260411/colmap_result/sparse/0
                  (or --colmap_traj if you already have a TUM-format COLMAP traj)
  --ref_view      filter COLMAP images to this view only (default High_Cam07)

Outputs (under --out, default ./compare):
  colmap_traj.txt        TUM, world-to-cam, ref_view only
  aligned_otf_traj.txt   TUM, OTF poses after Sim(3) alignment to COLMAP
  sim3.json              {"s": float, "R": 3x3 list, "t": 3 list}
  metrics.json           ATE / RPE numbers + cross-checks
  traj_3d.png            matplotlib 3D plot
  error_heatmap.png      per-timestep alignment error

Usage (run from on-the-fly-nvs/ root):
  python tools/compare_with_colmap.py --sanity
  python tools/compare_with_colmap.py \\
      --otf_traj results/<run>/trajectory_otf.txt \\
      --colmap_dir /path/to/colmap_result/sparse/0 \\
      --ref_view High_Cam07 \\
      --out compare/<run>
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from typing import Dict, List, Tuple

import numpy as np
from scipy.spatial.transform import Rotation as Rscipy

# ---------------------------------------------------------------------------
# Bring in COLMAP's read_model from the sibling dataloaders module
# (image.bin/images.txt parser).
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_OTF_DATALOADERS = os.path.normpath(os.path.join(_HERE, "..", "dataloaders"))
if os.path.isdir(_OTF_DATALOADERS):
    sys.path.insert(0, _OTF_DATALOADERS)
try:
    from read_write_model import read_model, qvec2rotmat  # type: ignore
except Exception as e:  # pragma: no cover
    print(
        "[fatal] could not import read_write_model from "
        f"{_OTF_DATALOADERS}: {e}",
        file=sys.stderr,
    )
    raise


# ---------------------------------------------------------------------------
# Quaternion / rotation helpers (TUM uses xyzw order)
# ---------------------------------------------------------------------------
def quat_xyzw_from_R(R: np.ndarray) -> np.ndarray:
    return Rscipy.from_matrix(R).as_quat()  # (x, y, z, w)


def R_from_quat_xyzw(q: np.ndarray) -> np.ndarray:
    return Rscipy.from_quat(q).as_matrix()


def rot_angle_deg(R: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))


# ---------------------------------------------------------------------------
# TUM IO
# ---------------------------------------------------------------------------
def read_tum(path: str) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """TUM file -> {ts: (R_w2c, t_w2c)}."""
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) != 8:
                raise ValueError(f"bad TUM row: {line!r}")
            ts = int(float(parts[0]))
            t = np.array([float(x) for x in parts[1:4]])
            q = np.array([float(x) for x in parts[4:8]])  # xyzw
            R = R_from_quat_xyzw(q)
            out[ts] = (R, t)
    return out


def write_tum(path: str, traj: Dict[int, Tuple[np.ndarray, np.ndarray]],
              header_lines: List[str] | None = None) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("# TUM format: timestamp tx ty tz qx qy qz qw\n")
        for h in header_lines or []:
            f.write(f"# {h}\n")
        for ts in sorted(traj):
            R, t = traj[ts]
            qx, qy, qz, qw = quat_xyzw_from_R(R)
            f.write(
                f"{ts} {t[0]:.9f} {t[1]:.9f} {t[2]:.9f} "
                f"{qx:.9f} {qy:.9f} {qz:.9f} {qw:.9f}\n"
            )


# ---------------------------------------------------------------------------
# COLMAP loader
# ---------------------------------------------------------------------------
_FRAME_NUM_RE = re.compile(r"(\d+)")


def _frame_num(name: str) -> int:
    m = _FRAME_NUM_RE.search(os.path.basename(name))
    return int(m.group(1)) if m else -1


def load_colmap_ref_view(
    colmap_dir: str, ref_view: str
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Read COLMAP sparse model, keep only images whose name starts with the
    ref_view directory or matches `<ref_view>__ts<NN>`. Returns {ts: (R_w2c, t_w2c)}.

    COLMAP poses are world-to-camera (qvec, tvec), same convention as OTF.
    Timestep is derived from the numeric part of the filename divided by the
    inter-keyframe stride; we infer the stride from the smallest non-zero
    delta so we don't need to know the stride upfront.
    """
    _, images, _ = read_model(colmap_dir)
    rows: List[Tuple[int, np.ndarray, np.ndarray, str]] = []
    for img in images.values():
        name = img.name
        # Two layouts: "<ref_view>/<frame>.png" or "<ref_view>__ts00001.png"
        belongs_to_ref = (
            name.startswith(f"{ref_view}/")
            or name.startswith(f"{ref_view}__ts")
            or os.path.basename(os.path.dirname(name)) == ref_view
        )
        if not belongs_to_ref:
            continue
        R = qvec2rotmat(img.qvec)
        t = np.array(img.tvec, dtype=np.float64)
        rows.append((_frame_num(name), R, t, name))

    if not rows:
        raise RuntimeError(
            f"No COLMAP images matched ref_view={ref_view!r}; "
            f"saw {len(images)} total images."
        )

    rows.sort(key=lambda r: r[0])
    fnums = [r[0] for r in rows]
    diffs = sorted({fnums[i + 1] - fnums[i] for i in range(len(fnums) - 1)})
    diffs = [d for d in diffs if d > 0]
    stride = diffs[0] if diffs else 1
    base = fnums[0]

    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    for fnum, R, t, _name in rows:
        ts = (fnum - base) // stride
        out[ts] = (R, t)
    return out


# ---------------------------------------------------------------------------
# Convention check: world-to-camera should produce a finite, sensible
# camera centre c = -R^T t. Anything else (NaN, or |c| absurdly large)
# means the input is cam-to-world or degenerate.
# ---------------------------------------------------------------------------
def assert_w2c_consistent(traj: Dict[int, Tuple[np.ndarray, np.ndarray]],
                          label: str) -> None:
    bad_det = 0
    centres = []
    for _ts, (R, t) in traj.items():
        d = np.linalg.det(R)
        if abs(d - 1.0) > 1e-3:
            bad_det += 1
        c = -R.T @ t
        centres.append(c)
    centres = np.stack(centres)
    if bad_det > 0:
        print(f"[warn] {label}: {bad_det}/{len(traj)} rotations have det(R)≠+1 "
              "(possibly reflection or invalid Rt).")
    if not np.isfinite(centres).all():
        raise RuntimeError(f"{label}: non-finite camera centres detected.")


# ---------------------------------------------------------------------------
# Sim(3) via Umeyama (1991)
# ---------------------------------------------------------------------------
def umeyama(src: np.ndarray, dst: np.ndarray) -> Tuple[float, np.ndarray, np.ndarray]:
    """Find (s, R, t) minimizing ||s R src_i + t - dst_i||^2.

    src, dst: (N, 3). Returns (s, R, t).
    """
    assert src.shape == dst.shape and src.ndim == 2 and src.shape[1] == 3
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = (dc.T @ sc) / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    s = float(np.trace(np.diag(D) @ S) / max(var_s, 1e-12))
    t = mu_d - s * R @ mu_s
    return s, R, t


def apply_sim3_to_traj(
    traj: Dict[int, Tuple[np.ndarray, np.ndarray]],
    s: float, R: np.ndarray, t: np.ndarray,
) -> Dict[int, Tuple[np.ndarray, np.ndarray]]:
    """Transform a w2c trajectory by Sim(3) acting on world frame.

    A camera centre transforms as C' = s R C + t.
    The world-to-camera pose (R_w2c, t_w2c) becomes
        R_w2c' = R_w2c @ R.T
        t_w2c' = (1/s) (t_w2c - R_w2c @ R.T @ t)
    """
    out: Dict[int, Tuple[np.ndarray, np.ndarray]] = {}
    Rt_inv = R.T
    for ts, (R_wc, t_wc) in traj.items():
        R_new = R_wc @ Rt_inv
        # New camera centre after Sim(3): C' = s R C + t.
        # We need t_w2c' such that -R_new^T t_new = C'.
        C = -R_wc.T @ t_wc
        C_new = s * R @ C + t
        t_new = -R_new @ C_new
        out[ts] = (R_new, t_new)
    return out


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def cam_centres(traj: Dict[int, Tuple[np.ndarray, np.ndarray]],
                keys: List[int]) -> np.ndarray:
    return np.stack([(-traj[k][0].T @ traj[k][1]) for k in keys])


def compute_ate_rpe(
    otf_aligned: Dict[int, Tuple[np.ndarray, np.ndarray]],
    col: Dict[int, Tuple[np.ndarray, np.ndarray]],
    common: List[int],
) -> Dict[str, float]:
    Cs_o = cam_centres(otf_aligned, common)
    Cs_c = cam_centres(col, common)
    err = np.linalg.norm(Cs_o - Cs_c, axis=1)
    ate_rmse = float(np.sqrt((err ** 2).mean()))
    ate_mean = float(err.mean())
    ate_max = float(err.max())

    # RPE (translation): consecutive relative motion
    if len(common) >= 2:
        d_o = Cs_o[1:] - Cs_o[:-1]
        d_c = Cs_c[1:] - Cs_c[:-1]
        rpe_t = np.linalg.norm(d_o - d_c, axis=1)
        rpe_t_rmse = float(np.sqrt((rpe_t ** 2).mean()))
    else:
        rpe_t_rmse = 0.0

    # RPE (rotation): per-step relative rotation difference
    rot_diffs = []
    for i in range(len(common) - 1):
        Ro_a, _ = otf_aligned[common[i]]
        Ro_b, _ = otf_aligned[common[i + 1]]
        Rc_a, _ = col[common[i]]
        Rc_b, _ = col[common[i + 1]]
        dR_o = Ro_b @ Ro_a.T
        dR_c = Rc_b @ Rc_a.T
        rot_diffs.append(rot_angle_deg(dR_o @ dR_c.T))
    rpe_rot_rmse_deg = (
        float(np.sqrt(np.mean(np.array(rot_diffs) ** 2))) if rot_diffs else 0.0
    )

    return {
        "ate_rmse": ate_rmse,
        "ate_mean": ate_mean,
        "ate_max": ate_max,
        "rpe_trans_rmse": rpe_t_rmse,
        "rpe_rot_rmse_deg": rpe_rot_rmse_deg,
        "n_pairs": len(common),
    }


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def make_plots(
    otf_aligned: Dict[int, Tuple[np.ndarray, np.ndarray]],
    col: Dict[int, Tuple[np.ndarray, np.ndarray]],
    common: List[int],
    out_dir: str,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    Cs_o = cam_centres(otf_aligned, common)
    Cs_c = cam_centres(col, common)

    # 3D trajectory plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(Cs_c[:, 0], Cs_c[:, 1], Cs_c[:, 2], "b.-",
            label="COLMAP", linewidth=1, markersize=4)
    ax.plot(Cs_o[:, 0], Cs_o[:, 1], Cs_o[:, 2], "r.-",
            label="OTF aligned", linewidth=1, markersize=4)
    for i in range(len(common)):
        ax.plot(
            [Cs_c[i, 0], Cs_o[i, 0]],
            [Cs_c[i, 1], Cs_o[i, 1]],
            [Cs_c[i, 2], Cs_o[i, 2]],
            "k-", alpha=0.2, linewidth=0.5,
        )
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.legend()
    ax.set_title("Trajectories after Sim(3) alignment")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "traj_3d.png"), dpi=140)
    plt.close()

    # Error heatmap (per timestep)
    err = np.linalg.norm(Cs_o - Cs_c, axis=1)
    fig, ax = plt.subplots(figsize=(10, 3))
    im = ax.imshow(err[None, :], aspect="auto", cmap="viridis")
    ax.set_yticks([])
    ax.set_xticks(range(0, len(err), max(1, len(err) // 10)))
    ax.set_xticklabels([str(common[i]) for i in range(0, len(err),
                                                       max(1, len(err) // 10))])
    ax.set_xlabel("timestep")
    ax.set_title(f"Per-timestep alignment error (mean={err.mean():.4f}, max={err.max():.4f})")
    plt.colorbar(im, ax=ax, label="‖ΔC‖")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "error_heatmap.png"), dpi=140)
    plt.close()


# ---------------------------------------------------------------------------
# Sanity check (Umeyama round-trip)
# ---------------------------------------------------------------------------
def sanity_check() -> int:
    rng = np.random.default_rng(0)
    n = 30
    # Random camera centres + random rotations.
    base_C = rng.normal(size=(n, 3))
    base_R = Rscipy.random(num=n, random_state=rng).as_matrix()

    traj = {}
    for i in range(n):
        R_wc = base_R[i]
        t_wc = -R_wc @ base_C[i]
        traj[i] = (R_wc, t_wc)

    # Apply random Sim(3): scale 0.3, random rotation, random translation.
    s_true = 0.3
    R_true = Rscipy.random(random_state=rng).as_matrix()
    t_true = rng.normal(size=3) * 5.0

    # Build "transformed" trajectory: world frame moves by (s, R, t).
    transformed = apply_sim3_to_traj(traj, 1.0 / s_true, R_true.T,
                                     -R_true.T @ t_true / s_true)
    # The above produces a w2c traj whose centres equal s*R@C + t for the original.
    # Now run Umeyama recovering that Sim(3).
    src = cam_centres(traj, list(range(n)))
    dst = cam_centres(transformed, list(range(n)))
    s, R, t = umeyama(src, dst)
    aligned = apply_sim3_to_traj(traj, s, R, t)
    metrics = compute_ate_rpe(aligned, transformed, list(range(n)))

    print(f"[sanity] recovered s={s:.6f} (expected {s_true:.6f})")
    print(f"[sanity] R diff = {rot_angle_deg(R @ R_true.T):.6f}°")
    print(f"[sanity] t diff = {np.linalg.norm(t - t_true):.6e}")
    print(f"[sanity] ATE RMSE = {metrics['ate_rmse']:.6e} (target < 1e-3)")
    if metrics["ate_rmse"] > 1e-3:
        print("[sanity] FAIL")
        return 1
    print("[sanity] OK")
    return 0


# ---------------------------------------------------------------------------
# Optional evo cross-check
# ---------------------------------------------------------------------------
def evo_cross_check(colmap_traj_path: str, aligned_otf_path: str) -> Dict[str, float] | None:
    if not shutil.which("evo_ape"):
        return None
    try:
        result = subprocess.run(
            ["evo_ape", "tum", colmap_traj_path, aligned_otf_path, "-as"],
            capture_output=True, text=True, timeout=60, check=True,
        )
    except Exception as e:
        print(f"[evo] cross-check failed: {e}")
        return None
    text = result.stdout
    out: Dict[str, float] = {}
    for key in ("rmse", "mean", "median", "std", "max"):
        m = re.search(rf"^\s*{key}\s+([\d.eE+\-]+)", text, re.MULTILINE)
        if m:
            out[key] = float(m.group(1))
    return out or None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--otf_traj")
    p.add_argument("--colmap_dir")
    p.add_argument("--colmap_traj")
    p.add_argument("--ref_view", default="High_Cam07")
    p.add_argument("--out", default="compare")
    p.add_argument("--sanity", action="store_true",
                   help="Run the Umeyama round-trip sanity check and exit.")
    args = p.parse_args()

    if args.sanity:
        return sanity_check()

    if not args.otf_traj:
        p.error("--otf_traj is required (or use --sanity)")
    if not (args.colmap_dir or args.colmap_traj):
        p.error("provide --colmap_dir or --colmap_traj")

    os.makedirs(args.out, exist_ok=True)

    # Load OTF
    otf = read_tum(args.otf_traj)
    assert_w2c_consistent(otf, label="OTF")

    # Load COLMAP
    if args.colmap_traj:
        col = read_tum(args.colmap_traj)
    else:
        col = load_colmap_ref_view(args.colmap_dir, args.ref_view)
        # Persist the COLMAP traj we used so the comparison is auditable.
        write_tum(
            os.path.join(args.out, "colmap_traj.txt"), col,
            header_lines=[
                f"source: {args.colmap_dir}",
                f"ref_view: {args.ref_view}",
                f"convention: world-to-camera",
            ],
        )
    assert_w2c_consistent(col, label="COLMAP")

    # Match by timestep
    common = sorted(set(otf) & set(col))
    if not common:
        raise RuntimeError("No common timesteps between OTF and COLMAP.")
    only_otf = sorted(set(otf) - set(col))
    only_col = sorted(set(col) - set(otf))
    if only_otf or only_col:
        print(f"[warn] timestep mismatch — OTF-only: {only_otf}, "
              f"COLMAP-only: {only_col}. proceeding with {len(common)} pairs.")

    # Sim(3)
    src = cam_centres(otf, common)
    dst = cam_centres(col, common)
    s, R, t = umeyama(src, dst)
    sim3_path = os.path.join(args.out, "sim3.json")
    with open(sim3_path, "w") as f:
        json.dump({"s": s, "R": R.tolist(), "t": t.tolist(),
                   "n_pairs": len(common), "ref_view": args.ref_view}, f, indent=2)
    print(f"[sim3] s={s:.6f}  |R diff from I|={rot_angle_deg(R):.2f}°  "
          f"|t|={np.linalg.norm(t):.4f}  → {sim3_path}")

    # Aligned trajectory
    otf_aligned = apply_sim3_to_traj(otf, s, R, t)
    aligned_path = os.path.join(args.out, "aligned_otf_traj.txt")
    write_tum(
        aligned_path, otf_aligned,
        header_lines=[f"aligned to: {args.colmap_dir or args.colmap_traj}",
                      f"sim3.json: scale={s:.6f}"],
    )

    # Metrics
    metrics = compute_ate_rpe(otf_aligned, col, common)
    metrics.update({
        "sim3_scale": s,
        "sim3_R_angle_deg": rot_angle_deg(R),
        "sim3_t_norm": float(np.linalg.norm(t)),
        "n_otf": len(otf), "n_colmap": len(col),
        "n_common": len(common),
        "only_otf": only_otf, "only_colmap": only_col,
    })

    # Sanity warn
    scene_scale = float(np.linalg.norm(dst - dst.mean(axis=0), axis=1).mean())
    if scene_scale > 0 and metrics["ate_rmse"] > 0.5 * scene_scale:
        print(f"[warn] ATE RMSE ({metrics['ate_rmse']:.4f}) is >50% of scene "
              f"scale ({scene_scale:.4f}) — alignment may be poor.")
    metrics["scene_scale_colmap"] = scene_scale

    print(f"[ate]  rmse={metrics['ate_rmse']:.4f}  mean={metrics['ate_mean']:.4f}  "
          f"max={metrics['ate_max']:.4f}")
    print(f"[rpe]  trans_rmse={metrics['rpe_trans_rmse']:.4f}  "
          f"rot_rmse={metrics['rpe_rot_rmse_deg']:.2f}°")

    # evo cross-check
    evo_metrics = evo_cross_check(
        os.path.join(args.out, "colmap_traj.txt") if args.colmap_dir else args.colmap_traj,
        aligned_path,
    )
    if evo_metrics is not None:
        metrics["evo_ape"] = evo_metrics
        print(f"[evo]  rmse={evo_metrics.get('rmse', float('nan')):.4f}  "
              f"(should match [ate] rmse)")

    # Plots
    make_plots(otf_aligned, col, common, args.out)
    print(f"[plot] {args.out}/traj_3d.png  {args.out}/error_heatmap.png")

    # Save metrics
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
