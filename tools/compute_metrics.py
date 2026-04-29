"""Frame-by-frame PSNR/SSIM/LPIPS between a predictions directory and a GT
directory.

Pairing rule (priority):
  1. If --pairs is given (CSV: pred_basename,gt_basename), use that.
  2. Else if both dirs share filenames, pair by exact name match.
  3. Else, pair by sorted order with `--ts_to_frame` step (default 40).
     This handles the case where preds use ts indexing
     (`<view>__ts000NN.png`) and GT uses absolute frame numbers
     (`frame_NNNNN.png`).

Outputs JSON:
  {"per_frame": [{name, psnr, ssim, lpips}, …],
   "summary":   {n, psnr, ssim, lpips}}

Usage:
  python tools/compute_metrics.py \
      --pred_dir <pred> --gt_dir <gt> --out metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def _load_image(path: str, device: str = "cuda") -> torch.Tensor:
    img = np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)


def _resize_to(t: torch.Tensor, h: int, w: int) -> torch.Tensor:
    if t.shape[-2:] == (h, w):
        return t
    return F.interpolate(t, size=(h, w), mode="bilinear", align_corners=False)


def _psnr(pred: torch.Tensor, gt: torch.Tensor) -> float:
    mse = ((pred - gt) ** 2).mean().item()
    if mse <= 1e-12:
        return 99.0
    return float(-10.0 * np.log10(mse))


def _ssim(pred: torch.Tensor, gt: torch.Tensor) -> float:
    # 11×11 Gaussian SSIM (single-scale, simplified)
    from math import exp
    sigma = 1.5
    ksize = 11
    coords = torch.arange(ksize, device=pred.device, dtype=pred.dtype) - (ksize - 1) / 2
    g = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    g = g / g.sum()
    kernel_2d = g[None, :] * g[:, None]
    kernel = kernel_2d.expand(3, 1, ksize, ksize)
    pad = ksize // 2

    def conv(x):
        return F.conv2d(x, kernel, padding=pad, groups=3)

    mu1, mu2 = conv(pred), conv(gt)
    mu1_sq, mu2_sq, mu12 = mu1 * mu1, mu2 * mu2, mu1 * mu2
    sigma1_sq = conv(pred * pred) - mu1_sq
    sigma2_sq = conv(gt * gt) - mu2_sq
    sigma12 = conv(pred * gt) - mu12
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    ssim_map = (
        (2 * mu12 + C1) * (2 * sigma12 + C2)
    ) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))
    return float(ssim_map.mean().item())


def _make_pairs(pred_dir: str, gt_dir: str, ts_to_frame_step: int
                ) -> List[Tuple[str, str]]:
    pred_files = sorted(f for f in os.listdir(pred_dir)
                        if f.lower().endswith((".png", ".jpg", ".jpeg")))
    gt_files = sorted(f for f in os.listdir(gt_dir)
                      if f.lower().endswith((".png", ".jpg", ".jpeg")))
    if not pred_files or not gt_files:
        return []

    # exact-name path
    common = sorted(set(pred_files) & set(gt_files))
    if common and len(common) >= max(1, len(pred_files) // 2):
        return [(c, c) for c in common]

    # ts → frame_(1 + step*i):05d.png path
    pairs: List[Tuple[str, str]] = []
    for i, p in enumerate(pred_files):
        frame_num = 1 + ts_to_frame_step * i
        for ext in (".png", ".jpg", ".jpeg"):
            cand = f"frame_{frame_num:05d}{ext}"
            if cand in gt_files:
                pairs.append((p, cand))
                break
    return pairs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--pred_dir", required=True)
    parser.add_argument("--gt_dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--ts_to_frame_step", type=int, default=40)
    parser.add_argument("--max_frames", type=int, default=None,
                        help="optional cap for fast iteration")
    parser.add_argument("--no_lpips", action="store_true")
    args = parser.parse_args()

    pairs = _make_pairs(args.pred_dir, args.gt_dir, args.ts_to_frame_step)
    if not pairs:
        sys.exit(f"[error] no frame pairs from {args.pred_dir} ↔ {args.gt_dir}")
    if args.max_frames:
        pairs = pairs[: args.max_frames]

    device = "cuda" if torch.cuda.is_available() else "cpu"

    lpips_model = None
    if not args.no_lpips:
        try:
            import lpips
            lpips_model = lpips.LPIPS(net="alex", verbose=False).to(device).eval()
        except Exception as e:
            print(f"[warn] LPIPS init failed ({e}); skipping LPIPS")

    per_frame = []
    psnrs, ssims, lpips_vals = [], [], []

    for pred_name, gt_name in pairs:
        pred = _load_image(os.path.join(args.pred_dir, pred_name), device)
        gt = _load_image(os.path.join(args.gt_dir, gt_name), device)
        if pred.shape[-2:] != gt.shape[-2:]:
            gt = _resize_to(gt, pred.shape[-2], pred.shape[-1])

        psnr = _psnr(pred, gt)
        ssim = _ssim(pred, gt)
        lp = float("nan")
        if lpips_model is not None:
            with torch.no_grad():
                # LPIPS expects [-1, 1]
                lp = float(lpips_model(pred * 2 - 1, gt * 2 - 1).item())

        per_frame.append({"pred": pred_name, "gt": gt_name,
                          "psnr": psnr, "ssim": ssim, "lpips": lp})
        psnrs.append(psnr)
        ssims.append(ssim)
        if not np.isnan(lp):
            lpips_vals.append(lp)

    summary = {
        "n": len(per_frame),
        "psnr": float(np.mean(psnrs)),
        "ssim": float(np.mean(ssims)),
        "lpips": float(np.mean(lpips_vals)) if lpips_vals else float("nan"),
    }
    out = {"summary": summary, "per_frame": per_frame,
           "config": {"pred_dir": args.pred_dir, "gt_dir": args.gt_dir,
                      "ts_to_frame_step": args.ts_to_frame_step}}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(out, fh, indent=2)

    print(f"  n={summary['n']}  PSNR={summary['psnr']:.3f}  "
          f"SSIM={summary['ssim']:.3f}  LPIPS={summary['lpips']:.3f}")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
