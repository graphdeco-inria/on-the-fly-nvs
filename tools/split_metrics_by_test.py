"""Split render_eval/metrics.json into train/holdout/all on `is_test`.

Reads:
  <run>/render_eval/metrics.json  (per_frame list with `is_test`)
  <run>/runtimes.json             (wall_clock_s, n_gauss_final)
  <run>/metadata.json             (num anchors)

Writes:
  <run>/render_eval/split_summary.json with per-split mean PSNR/SSIM/LPIPS.

Run from on-the-fly-nvs/ root:
  python tools/split_metrics_by_test.py <run>
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir", help="Path to run directory (e.g. results/foo).")
    p.add_argument("--out", default=None,
                   help="Output JSON path (default <run>/render_eval/split_summary.json)")
    args = p.parse_args()

    metrics_p = os.path.join(args.run_dir, "render_eval", "metrics.json")
    if not os.path.exists(metrics_p):
        sys.exit(f"[error] missing {metrics_p}")
    with open(metrics_p) as fh:
        m = json.load(fh)
    pf = m.get("per_frame", [])

    splits = {"train": [], "holdout": [], "all": pf}
    for f in pf:
        splits["holdout" if f.get("is_test") else "train"].append(f)

    out = {"splits": {}}
    for name, group in splits.items():
        if not group:
            out["splits"][name] = {"n": 0, "psnr": float("nan"),
                                   "ssim": float("nan"), "lpips": float("nan")}
            continue
        out["splits"][name] = {
            "n": len(group),
            "psnr": float(np.mean([f["psnr"] for f in group])),
            "ssim": float(np.mean([f["ssim"] for f in group])),
            "lpips": float(np.mean([f["lpips"] for f in group])),
        }

    rt_p = os.path.join(args.run_dir, "runtimes.json")
    if os.path.exists(rt_p):
        with open(rt_p) as fh:
            rt = json.load(fh)
        out["wall_clock_s"] = rt.get("wall_clock_s")
        out["n_gauss_final"] = rt.get("n_gauss_final")
        out["gpu_peak_gb"] = rt.get("gpu_peak_gb")

    md_p = os.path.join(args.run_dir, "metadata.json")
    if os.path.exists(md_p):
        with open(md_p) as fh:
            md = json.load(fh)
        out["num_anchors"] = md.get("num anchors")

    out_path = args.out or os.path.join(args.run_dir, "render_eval",
                                        "split_summary.json")
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)

    for name in ("train", "holdout", "all"):
        s = out["splits"][name]
        print(f"  {name:8s} n={s['n']:4d}  PSNR={s['psnr']:.3f}  "
              f"SSIM={s['ssim']:.3f}  LPIPS={s['lpips']:.3f}")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
