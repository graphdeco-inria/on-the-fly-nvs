"""σ_center sanity check — measures whether the rig invariant
(all 9 views share a common optical centre) holds at the end of training.

Reads metadata.json (produced by train.py at save time) which lists keyframes
with their final (4×4) Rt, groups by ts_idx (extracted from name), and reports
per-ts max deviation of camera centres from their mean. Under the rig
assumption (rotation-only relative_Rt, rel_t=0) this should be ≈ 0; any
non-zero value comes from view-unit photometric drift.

Usage:
  python tools/analyze_view_drift.py --metadata <run>/metadata.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--metadata", required=True)
    args = p.parse_args()

    with open(args.metadata) as fh:
        m = json.load(fh)
    kfs = m.get("keyframes", [])
    if not kfs:
        print("[error] no keyframes in metadata", file=sys.stderr)
        sys.exit(1)

    groups = {}
    for kf in kfs:
        name = kf["info"]["name"]
        match = re.search(r"ts(\d+)", name)
        if not match:
            continue
        ts = int(match.group(1))
        Rt = np.array(kf["Rt"])
        R, t = Rt[:3, :3], Rt[:3, 3]
        c = -R.T @ t
        groups.setdefault(ts, []).append((kf["info"]["name"], c))

    sigmas = []
    for ts, items in sorted(groups.items()):
        centers = np.array([c for _, c in items])
        if len(centers) < 2:
            continue
        d = np.linalg.norm(centers - centers.mean(axis=0), axis=1)
        sigmas.append({"ts": ts, "n_views": len(centers),
                       "max_dev": float(d.max()),
                       "rms_dev": float(np.sqrt((d ** 2).mean()))})

    all_centers = np.array([c for _, c in sum(groups.values(), [])])
    scene_scale = float(np.linalg.norm(
        all_centers - all_centers.mean(axis=0), axis=1).mean())

    max_devs = np.array([s["max_dev"] for s in sigmas])
    summary = {
        "scene_scale": scene_scale,
        "n_ts": len(sigmas),
        "sigma_center_mean": float(max_devs.mean()),
        "sigma_center_max": float(max_devs.max()),
        "sigma_center_min": float(max_devs.min()),
        "sigma_over_scene_pct": float(max_devs.mean() / scene_scale * 100),
    }

    print(f"scene_scale (mean dist from centroid): {scene_scale:.6f}")
    print(f"σ_center per-ts max deviation across views:")
    print(f"  n_ts={len(sigmas)}")
    print(f"  mean = {summary['sigma_center_mean']:.3e}")
    print(f"  max  = {summary['sigma_center_max']:.3e}")
    print(f"  min  = {summary['sigma_center_min']:.3e}")
    print(f"σ/scene_scale (mean) = {summary['sigma_over_scene_pct']:.3e}%")

    out_path = args.metadata.replace(".json", "_view_drift.json")
    if out_path != args.metadata:
        with open(out_path, "w") as fh:
            json.dump({"summary": summary, "per_ts": sigmas}, fh, indent=2)
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
