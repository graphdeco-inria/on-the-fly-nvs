"""Rig-aware PnP: method A from TODO.md §3.3.2.

Each view runs PnP independently, the view poses are lifted to candidate
rig poses via the known fixed relative transforms, and the candidates are
combined on SE(3) (weighted Fréchet mean).
"""

from typing import Dict, Iterable, Optional, Tuple

import cv2
import numpy as np
import torch
from torch import Tensor

from rig.rig_loader import RigConfig
from rig.se3_utils import se3_robust_mean


def _to_numpy(x):
    return x.detach().cpu().numpy() if isinstance(x, Tensor) else np.asarray(x)


def rig_pnp_per_view(
    correspondences: Dict[str, Tuple[Tensor, Tensor]],
    rig_config: RigConfig,
    K: Tensor,
    min_correspondences: int = 6,
    reproj_error_px: float = 4.0,
    scene_scale: float = 0.1,
) -> Tuple[Optional[Tensor], Dict[str, dict]]:
    """
    correspondences: {view_name: (pts_2d[N,2], pts_3d[N,3])}
    K: (3, 3) camera intrinsic matrix (assumed identical across views).
    scene_scale: expected translation magnitude between neighboring rig
        poses. Used by the robust mean's translation Huber threshold so the
        kernel stays meaningful across different dataset scales.
    Returns: (rig_world2cam 4x4 or None, per_view_stats dict).
    """
    K_np = _to_numpy(K).astype(np.float64)
    rig_candidates = []
    weights = []
    stats: Dict[str, dict] = {}

    for view_name, (pts2d, pts3d) in correspondences.items():
        n = pts2d.shape[0] if pts2d is not None else 0
        if n < min_correspondences:
            stats[view_name] = {"success": False, "n_points": n, "n_inliers": 0}
            continue
        pts2d_np = _to_numpy(pts2d).astype(np.float64).reshape(-1, 1, 2)
        pts3d_np = _to_numpy(pts3d).astype(np.float64).reshape(-1, 1, 3)
        try:
            ok, rvec, tvec, inliers = cv2.solvePnPRansac(
                pts3d_np, pts2d_np, K_np, None,
                flags=cv2.SOLVEPNP_EPNP,
                reprojectionError=reproj_error_px,
                iterationsCount=2000,
                confidence=0.999,
            )
        except cv2.error:
            ok, inliers = False, None

        if not ok or inliers is None or len(inliers) < min_correspondences:
            stats[view_name] = {"success": False, "n_points": n,
                                "n_inliers": 0 if inliers is None else len(inliers)}
            continue

        R_view, _ = cv2.Rodrigues(rvec)
        view_w2c = np.eye(4, dtype=np.float64)
        view_w2c[:3, :3] = R_view
        view_w2c[:3, 3] = tvec.reshape(3)
        view_w2c_t = torch.from_numpy(view_w2c).to(dtype=torch.float32, device=K.device)

        rel = rig_config.relative_Rt[view_name].to(view_w2c_t)
        # view_w2c = rel @ rig_w2c  =>  rig_w2c = rel^{-1} @ view_w2c
        rig_w2c = torch.linalg.inv(rel) @ view_w2c_t

        rig_candidates.append(rig_w2c)
        weights.append(float(len(inliers)))
        stats[view_name] = {"success": True, "n_points": n, "n_inliers": int(len(inliers))}

    if not rig_candidates:
        return None, stats

    Ts = torch.stack(rig_candidates, dim=0)
    w = torch.tensor(weights, dtype=torch.float32, device=Ts.device)
    rig_mean, eff_w = se3_robust_mean(Ts, w, scene_scale=scene_scale)
    # Expose the effective (post-Huber) weights so callers can see which
    # views ended up trusted.
    view_to_eff_w = {}
    candidate_views = [v for v, s in stats.items() if s.get("success")]
    for v, ew in zip(candidate_views, eff_w.tolist()):
        stats[v]["effective_weight"] = float(ew)
    return rig_mean, stats
