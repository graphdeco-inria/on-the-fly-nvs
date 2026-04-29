"""3-way render figure grid: GT × original 3DGS × on-the-fly-nvs (rig-unit B-1).

Layout: rows = (frame × view), columns = (GT, gsplat, OTF). PSNR shown
underneath each non-GT panel.

Usage:
  python tools/three_way_grid.py \
      --gt_dir       /opt/ftp/files/260411/images \
      --gsplat_dir   /opt/ftp/files/260429_gsplat/images_rendered \
      --otf_dir      results/b1_v2/run_B1/render_eval \
      --views        High_Cam01,High_Cam07 \
      --ts_idx       0,5,10,15,22 \
      --out          figures/3way_render_grid.png
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from PIL import Image


def _load_np(path, crop_left_half=False):
    img = np.array(Image.open(path).convert("RGB"))
    if crop_left_half:
        # OTF render_eval saves [render | gt] side-by-side; crop to render.
        w = img.shape[1]
        img = img[:, : w // 2, :]
    return img


def _psnr_np(a, b):
    a = a.astype(np.float32) / 255.0
    b = b.astype(np.float32) / 255.0
    mse = ((a - b) ** 2).mean()
    return 99.0 if mse < 1e-12 else float(-10 * np.log10(mse))


def _list_frames(d):
    return sorted(f for f in os.listdir(d)
                  if f.lower().endswith((".png", ".jpg", ".jpeg")))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--gsplat_dir", required=True)
    parser.add_argument("--otf_dir", required=True)
    parser.add_argument("--views", required=True, help="comma-separated view names")
    parser.add_argument("--ts_idx", required=True, help="comma-separated ts indices")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    views = [v.strip() for v in args.views.split(",")]
    ts_indices = [int(t) for t in args.ts_idx.split(",")]

    n_rows = len(views) * len(ts_indices)
    n_cols = 3  # GT, gsplat, OTF

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 3.2, n_rows * 3.4),
        squeeze=False,
    )
    col_titles = ["GT", "original 3DGS", "OTF (rig-unit B-1)"]

    row = 0
    for view in views:
        gt_view = os.path.join(args.gt_dir, view)
        gsp_view = os.path.join(args.gsplat_dir, view)
        if not (os.path.isdir(gt_view) and os.path.isdir(gsp_view)):
            print(f"[warn] missing dirs for {view}")
            continue
        gt_files = _list_frames(gt_view)

        for ts in ts_indices:
            if ts >= len(gt_files):
                print(f"[warn] ts={ts} out of range for {view} ({len(gt_files)} frames)")
                continue
            gt_name = gt_files[ts]
            gt_path = os.path.join(gt_view, gt_name)
            gsp_path = os.path.join(gsp_view, gt_name)
            otf_path = os.path.join(args.otf_dir, f"{view}__ts{ts:05d}.png")

            try:
                gt_img = _load_np(gt_path)
            except Exception as exc:
                print(f"[warn] failed GT {gt_path}: {exc}")
                row += 1; continue

            cols_data = [(gt_img, None), (None, None), (None, None)]
            for col, path in enumerate([gsp_path, otf_path], start=1):
                if not os.path.isfile(path):
                    cols_data[col] = (None, None); continue
                try:
                    # column 2 (otf) is a side-by-side composite; crop.
                    img = _load_np(path, crop_left_half=(col == 2))
                    if img.shape != gt_img.shape:
                        # quick resize via torch
                        t = torch.from_numpy(img).permute(2, 0, 1)[None].float()
                        t = F.interpolate(t, size=gt_img.shape[:2],
                                          mode="bilinear", align_corners=False)
                        img = t[0].permute(1, 2, 0).clamp(0, 255).byte().numpy()
                    psnr = _psnr_np(img, gt_img)
                    cols_data[col] = (img, psnr)
                except Exception as exc:
                    print(f"[warn] failed {path}: {exc}")

            for col in range(n_cols):
                ax = axes[row, col]
                img, psnr = cols_data[col]
                if img is None:
                    ax.text(0.5, 0.5, "—", ha="center", va="center",
                            transform=ax.transAxes, fontsize=14)
                else:
                    ax.imshow(img)
                ax.set_xticks([]); ax.set_yticks([])
                # row label on first column
                if col == 0:
                    ax.set_ylabel(f"{view}\nts={ts}", fontsize=9)
                # column label on first row
                if row == 0:
                    ax.set_title(col_titles[col], fontsize=10)
                # PSNR caption beneath panel
                if col > 0 and psnr is not None:
                    ax.text(0.5, -0.06, f"PSNR={psnr:.2f} dB",
                            ha="center", va="top", transform=ax.transAxes,
                            fontsize=8)
            row += 1

    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    fig.savefig(args.out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
