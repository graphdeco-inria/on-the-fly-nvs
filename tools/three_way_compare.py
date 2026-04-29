"""3-way per-frame quality comparison between GT, gsplat baseline, and the
on-the-fly-nvs (rig-unit B-1) renderings.

Pairs frames using the same logic as compute_metrics.py, then computes
mean PSNR/SSIM/LPIPS per (split, method) cell.

Inputs:
  --gt_dir       e.g. /opt/ftp/files/260411/images
  --gsplat_dir   e.g. /opt/ftp/files/260429_gsplat/images_rendered
  --otf_dir      e.g. results/b1_v2/run_B1/render_eval
  --metadata     <run>/metadata.json (for is_test split per ts × view)

GT and gsplat are organized by view subdirectories (View/frame_NNNNN.png);
OTF render_eval has a flat directory of <View>__tsNNNNN.png.

Outputs:
  --out          JSON with per-cell + per-frame numbers
  --out_table    Markdown table for the report (§12)

Run from on-the-fly-nvs/ root.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# -- helpers from compute_metrics.py kept self-contained -------------------
def _load(path, device, crop_left_half=False):
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    if crop_left_half:
        # OTF render_eval saves [render | gt] side-by-side (W=2*orig).
        w = img.shape[1]
        img = img[:, : w // 2, :]
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def _resize_to(t, h, w):
    if t.shape[-2:] == (h, w):
        return t
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


def _psnr(pred, gt):
    mse = ((pred - gt) ** 2).mean().item()
    return 99.0 if mse < 1e-12 else float(-10 * np.log10(mse))


def _ssim(pred, gt):
    sigma, ksize = 1.5, 11
    coords = torch.arange(ksize, device=pred.device, dtype=pred.dtype) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2)); g = g / g.sum()
    kernel = (g[None, :] * g[:, None]).expand(3, 1, ksize, ksize)
    pad = ksize // 2
    def conv(x): return F.conv2d(x, kernel, padding=pad, groups=3)
    mu1, mu2 = conv(pred), conv(gt)
    sigma1_sq = conv(pred * pred) - mu1 * mu1
    sigma2_sq = conv(gt * gt) - mu2 * mu2
    sigma12 = conv(pred * gt) - mu1 * mu2
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / (
        (mu1 * mu1 + mu2 * mu2 + C1) * (sigma1_sq + sigma2_sq + C2)
    )
    return float(ssim_map.mean().item())


def _build_split_table(metadata_path: str) -> Dict[Tuple[str, int], bool]:
    with open(metadata_path) as fh:
        m = json.load(fh)
    out = {}
    for kf in m.get("keyframes", []):
        name = kf["info"]["name"]
        is_test = bool(kf["info"].get("is_test", False))
        match = re.match(r"(.+?)__ts(\d+)", name)
        if match:
            view = match.group(1)
            ts = int(match.group(2))
            out[(view, ts)] = is_test
    return out


def _walk_views(root: str) -> List[str]:
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)))


def _list_frames(view_dir: str):
    return sorted(f for f in os.listdir(view_dir)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))


def _frame_to_ts_map(gt_view_dir: str) -> Dict[str, int]:
    """Map gt frame filename → ts index (sorted order)."""
    files = _list_frames(gt_view_dir)
    return {f: i for i, f in enumerate(files)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--gsplat_dir", required=True)
    parser.add_argument("--otf_dir", required=True)
    parser.add_argument("--metadata", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--out_table", required=True)
    parser.add_argument("--no_lpips", action="store_true")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as exc:
            print(f"[warn] LPIPS unavailable: {exc}")

    split_table = _build_split_table(args.metadata)
    if not split_table:
        sys.exit(f"[error] no keyframes parsed from {args.metadata}")

    methods = ["gsplat", "otf"]
    per_frame_records = []

    views = _walk_views(args.gt_dir)
    for view in views:
        gt_view = os.path.join(args.gt_dir, view)
        gsp_view = os.path.join(args.gsplat_dir, view)
        if not os.path.isdir(gsp_view):
            print(f"[warn] gsplat missing view {view}; skip")
            continue
        # frame → ts mapping comes from sorted gt files
        gt_files = _list_frames(gt_view)
        for ts, gt_name in enumerate(gt_files):
            split = "holdout" if split_table.get((view, ts), False) else "train"
            gt_img = _load(os.path.join(gt_view, gt_name), device)
            for method in methods:
                if method == "gsplat":
                    pred_path = os.path.join(gsp_view, gt_name)
                else:
                    pred_path = os.path.join(args.otf_dir, f"{view}__ts{ts:05d}.png")
                if not os.path.isfile(pred_path):
                    continue
                # OTF post-hoc renders are [render | gt] composites; crop.
                pred = _load(pred_path, device,
                             crop_left_half=(method == "otf"))
                if pred.shape[-2:] != gt_img.shape[-2:]:
                    pred = _resize_to(pred, gt_img.shape[-2], gt_img.shape[-1])
                psnr = _psnr(pred, gt_img)
                ssim = _ssim(pred, gt_img)
                lp = float("nan")
                if lpips_model is not None:
                    with torch.no_grad():
                        lp = float(lpips_model(pred * 2 - 1, gt_img * 2 - 1).item())
                per_frame_records.append({
                    "view": view, "ts": ts, "method": method,
                    "split": split, "psnr": psnr, "ssim": ssim, "lpips": lp,
                })

    # aggregate by (split, method)
    agg = {}
    for r in per_frame_records:
        key = (r["split"], r["method"])
        agg.setdefault(key, []).append(r)

    summary = {}
    for (split, method), rows in agg.items():
        s = {"n": len(rows)}
        for k in ("psnr", "ssim", "lpips"):
            vs = [r[k] for r in rows if not np.isnan(r[k])]
            s[k] = float(np.mean(vs)) if vs else float("nan")
        summary[f"{split}__{method}"] = s

    out = {"summary": summary, "per_frame": per_frame_records}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    # markdown table
    L = []
    L.append("| split | n | metric | gsplat | OTF (rig-unit) | Δ (OTF − gsplat) |")
    L.append("|---|---:|---|---:|---:|---:|")
    for split in ("train", "holdout"):
        for metric in ("psnr", "ssim", "lpips"):
            g = summary.get(f"{split}__gsplat", {}).get(metric, float("nan"))
            o = summary.get(f"{split}__otf", {}).get(metric, float("nan"))
            n = summary.get(f"{split}__gsplat", {}).get("n", 0)
            d = o - g if not (np.isnan(g) or np.isnan(o)) else float("nan")
            L.append(f"| {split} | {n} | {metric.upper()} | "
                     f"{g:.3f} | {o:.3f} | {d:+.3f} |")
    with open(args.out_table, "w") as fh:
        fh.write("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nwrote {args.out}\nwrote {args.out_table}")


if __name__ == "__main__":
    main()
