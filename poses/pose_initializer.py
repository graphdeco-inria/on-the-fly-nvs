#
# Copyright (C) 2025, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os

import torch
import math

import numpy as np

from poses.feature_detector import DescribedKeypoints
from poses.mini_ba import MiniBA
from poses.mini_ba_rig import MiniBARig
from utils import fov2focal, depth2points, sixD2mtx, mtx2sixD, sample
from scene.mono_depth import relative_idepth_to_depth
from scene.keyframe import Keyframe
from poses.ransac import RANSACEstimator, EstimatorType
from rig.rig_pnp import rig_pnp_per_view


def _print_matcher_diag(records, view_names, n_ts):
    """Inlier-ratio histogram per matcher call. Splits losses by stage:
    NN match (semantic descriptor agreement) vs RANSAC fundamental (geometric)."""
    if not records:
        print("[diag] no matcher records.")
        return
    n_calls = len(records)
    avg_kpts = (sum(r["n_kpts1"] for r in records) +
                sum(r["n_kpts2"] for r in records)) / (2 * n_calls)
    avg_match_nn = sum(r["n_match_nn"] for r in records) / n_calls
    avg_inliers = sum(r["n_inliers"] for r in records) / n_calls
    nn_ratio = avg_match_nn / max(avg_kpts, 1)
    inlier_ratio_of_match = avg_inliers / max(avg_match_nn, 1)
    inlier_ratio_of_kpts = avg_inliers / max(avg_kpts, 1)

    print(f"[diag/matcher] {n_calls} pair calls across {len(view_names)} views, "
          f"{n_ts} ts (C(N_ts,2)={n_ts * (n_ts - 1) // 2} per view)")
    print(f"[diag/matcher] avg_kpts/img={avg_kpts:.0f}  "
          f"avg_NN_match={avg_match_nn:.0f} ({100 * nn_ratio:.1f}% of kpts)  "
          f"avg_inliers={avg_inliers:.0f} ({100 * inlier_ratio_of_match:.1f}% of NN, "
          f"{100 * inlier_ratio_of_kpts:.1f}% of kpts)")

    # Distribution buckets of inliers per call
    counts = sorted(r["n_inliers"] for r in records)
    bins = [0, 10, 50, 100, 200, 500, 1000, 99999]
    bin_labels = ["0", "1-9", "10-49", "50-99", "100-199", "200-499", "500-999", "1000+"]
    hist = [0] * len(bin_labels)
    for c in counts:
        for i in range(len(bins) - 1):
            if bins[i] <= c < bins[i + 1]:
                hist[i] += 1
                break
    print(f"[diag/matcher] inliers/call distribution:")
    for lbl, n in zip(bin_labels, hist):
        bar = "#" * int(40 * n / max(n_calls, 1))
        print(f"  {lbl:>9}: {n:>4} ({100 * n / n_calls:>5.1f}%) {bar}")

    # Per-view breakdown (which view is weakest?). Records are per-call across
    # all 9 views; group by lower-numbered kID's view.
    n_views = len(view_names)
    per_view_inliers = {v: [] for v in view_names}
    for r in records:
        v_idx = (min(r["kID"], r["kID_other"])) % n_views
        per_view_inliers[view_names[v_idx]].append(r["n_inliers"])
    print(f"[diag/matcher] per-view mean inliers/call:")
    for v in view_names:
        arr = per_view_inliers[v]
        if not arr:
            continue
        m = sum(arr) / len(arr)
        print(f"  {v:>14}: {m:>6.0f}  (n_calls={len(arr)})")


def _print_e0_rejected_obs(uv, r, mask, view_names, n_ts, n_views, actual_npts,
                           centre, width, height):
    """Where do BA-rejected observations live?

    For each (point, ts, view) slot: was uv valid AND BA dropped it?
    Splits the loss along four axes so we can tell whether the BA filter is
    being unfairly aggressive on a uniform tail (-> relax mad_scale) or
    correctly killing junk concentrated in image corners / a bad timestep /
    a single noisy view (-> fix data, not the filter).
    """
    n_obs = n_ts * n_views
    # uv: (actual_npts, n_ts, n_views, 2). valid_before = where matcher gave us
    # a real observation.
    valid_before = (uv >= 0).all(dim=-1)                     # (npts, n_ts, n_views)
    mask_view = mask.view(actual_npts, n_obs, 2).any(dim=-1) # (npts, n_obs)
    accepted = mask_view.view(actual_npts, n_ts, n_views)
    rejected = valid_before & ~accepted

    n_valid = int(valid_before.sum().item())
    n_rej = int(rejected.sum().item())
    if n_valid == 0:
        print("[diag/E0] no valid observations to analyse.")
        return

    print(f"[diag/E0] BA-rejected observations: "
          f"{n_rej}/{n_valid} ({100 * n_rej / n_valid:.1f}%) "
          f"valid uv → BA mask=False")

    # ---- Axis 1: per-timestep (uniform vs. concentrated)
    rej_per_ts = rejected.sum(dim=(0, 2)).cpu().numpy()
    val_per_ts = valid_before.sum(dim=(0, 2)).cpu().numpy()
    print(f"[diag/E0] per-timestep rejection rate (rejected / valid):")
    for ts in range(n_ts):
        v = int(val_per_ts[ts])
        rj = int(rej_per_ts[ts])
        ratio = rj / max(v, 1)
        bar = "#" * int(40 * ratio / max(rej_per_ts.max() / max(val_per_ts.max(), 1), 1e-9))
        print(f"  ts={ts}: {rj:>4}/{v:>4} ({100 * ratio:>5.1f}%) {bar}")

    # ---- Axis 2: per-view (single bad camera?)
    rej_per_view = rejected.sum(dim=(0, 1)).cpu().numpy()
    val_per_view = valid_before.sum(dim=(0, 1)).cpu().numpy()
    print(f"[diag/E0] per-view rejection rate (rejected / valid):")
    for v_idx, v_name in enumerate(view_names):
        v = int(val_per_view[v_idx])
        rj = int(rej_per_view[v_idx])
        ratio = rj / max(v, 1)
        print(f"  {v_name:>14}: {rj:>4}/{v:>4} ({100 * ratio:>5.1f}%)")

    # ---- Axis 3: residual distribution of rejected vs accepted obs
    r_view = r.view(actual_npts, n_obs, 2)
    res_norm = r_view.norm(dim=-1)                              # (npts, n_obs)
    res_norm = res_norm.view(actual_npts, n_ts, n_views)        # (npts, n_ts, n_views)
    rej_residuals = res_norm[rejected].cpu().numpy()
    acc_residuals = res_norm[accepted].cpu().numpy()
    bins = [0, 1, 2, 5, 10, 50, 100, 1e9]
    bin_lbl = ["<1", "1-2", "2-5", "5-10", "10-50", "50-100", "100+"]
    print(f"[diag/E0] residual distribution (px), accepted vs rejected:")
    print(f"  {'bin':>9} {'accepted':>10} {'rejected':>10}")
    for lo, hi, lbl in zip(bins[:-1], bins[1:], bin_lbl):
        a = int(((acc_residuals >= lo) & (acc_residuals < hi)).sum())
        rj = int(((rej_residuals >= lo) & (rej_residuals < hi)).sum())
        print(f"  {lbl:>9} {a:>10} {rj:>10}")
    if rej_residuals.size > 0:
        rq = np.quantile(rej_residuals, [0.25, 0.5, 0.75, 0.95])
        print(f"  rejected quartiles: q25={rq[0]:.2f}  q50={rq[1]:.2f}  "
              f"q75={rq[2]:.2f}  q95={rq[3]:.2f}")

    # ---- Axis 4: keypoint position (corner concentration?)
    # Corners are far from image centre; we bin by normalized distance from
    # centre (0 = centre, 1 = farthest corner of an axis-aligned image).
    cx, cy = float(centre[0].item()), float(centre[1].item())
    half_diag = float(np.sqrt(cx ** 2 + cy ** 2))
    rej_idx = rejected.nonzero(as_tuple=False)
    if rej_idx.numel() > 0:
        # uv: (npts, n_ts, n_views, 2)
        rej_uv = uv[rej_idx[:, 0], rej_idx[:, 1], rej_idx[:, 2]].cpu().numpy()
        d_norm = np.sqrt(((rej_uv - np.array([cx, cy])) ** 2).sum(axis=-1)) / half_diag
        d_acc = (
            uv[accepted.nonzero(as_tuple=False)[:, 0],
               accepted.nonzero(as_tuple=False)[:, 1],
               accepted.nonzero(as_tuple=False)[:, 2]].cpu().numpy()
        )
        d_acc_norm = (
            np.sqrt(((d_acc - np.array([cx, cy])) ** 2).sum(axis=-1)) / half_diag
        )
        bins = [0.0, 0.25, 0.5, 0.75, 1.0]
        labels = ["centre", "inner-mid", "outer-mid", "corner"]
        print(f"[diag/E0] kpt position distribution (normalized dist from centre):")
        print(f"  {'ring':>10} {'accepted':>10} {'rejected':>10} "
              f"{'rej%':>8}")
        for lo, hi, lbl in zip(bins[:-1], bins[1:], labels):
            a = int(((d_acc_norm >= lo) & (d_acc_norm < hi)).sum())
            rj = int(((d_norm >= lo) & (d_norm < hi)).sum())
            ratio = rj / max(a + rj, 1)
            print(f"  {lbl:>10} {a:>10} {rj:>10} {100 * ratio:>7.1f}%")


def _print_obs_histograms(uv, view_names, npts_per_view, n_ts):
    """Per-point observation count histogram. uv: (npts_total, n_ts, n_views, 2)."""
    valid = (uv >= 0).all(dim=-1)              # (npts, n_ts, n_views)
    obs_count = valid.any(dim=-1).sum(dim=-1)  # (npts,) — # of ts where ANY view sees it
    bins = [(0, 1, "0"), (1, 2, "1"), (2, 3, "2"), (3, 6, "3-5"),
            (6, 11, "6-10"), (11, 99999, "11+")]
    total = obs_count.numel()
    print(f"[diag/obs] per-point timestep-observation histogram (n={total}):")
    for lo, hi, lbl in bins:
        n = int(((obs_count >= lo) & (obs_count < hi)).sum().item())
        bar = "#" * int(40 * n / max(total, 1))
        print(f"  ts_obs {lbl:>5}: {n:>5} ({100 * n / total:>5.1f}%) {bar}")
    print(f"[diag/obs] {'-' * 40}")

    # Per-view: each chunk of npts_per_view belongs to one view.
    print(f"[diag/obs] per-view few_obs (<2 ts) ratio:")
    for v_idx, v in enumerate(view_names):
        lo = v_idx * npts_per_view
        hi = lo + npts_per_view
        view_obs = obs_count[lo:hi]
        few = int((view_obs < 2).sum().item())
        print(f"  {v:>14}: {few:>4}/{npts_per_view} ({100 * few / npts_per_view:>5.1f}%)")


class PoseInitializer():
    """Fast pose initializer using MiniBA and the previous frames."""
    def __init__(self, width, height, triangulator, matcher, max_pnp_error, args, rig_config=None):
        self.width = width
        self.height = height
        self.triangulator = triangulator
        self.max_pnp_error = max_pnp_error
        self.matcher = matcher

        self.centre = torch.tensor([(width - 1) / 2, (height - 1) / 2], device='cuda')
        self.num_pts_miniba_bootstrap = args.num_pts_miniba_bootstrap
        self.num_kpts = args.num_kpts

        self.num_pts_pnpransac = 2 * args.num_pts_miniba_incr
        self.num_pts_miniba_incr = args.num_pts_miniba_incr
        self.min_num_inliers = args.min_num_inliers

        # Initialize the focal length
        if args.init_focal > 0:
            self.f_init = args.init_focal
        elif args.init_fov > 0:
            self.f_init = fov2focal(args.init_fov * math.pi / 180, width)
        else:
            self.f_init = 0.7 * width

        # Initialize MiniBA models
        self.miniba_bootstrap = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  not args.fix_focal, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniba_rebooting = MiniBA(
            1, args.num_keyframes_miniba_bootstrap, 0, args.num_pts_miniba_bootstrap,  False, True,
            make_cuda_graph=True, iters=args.iters_miniba_bootstrap)
        self.miniBA_incr = MiniBA(
            1, 1, 0, args.num_pts_miniba_incr, optimize_focal=False, optimize_3Dpts=False,
            make_cuda_graph=True, iters=args.iters_miniba_incr)
        
        self.PnPRANSAC = RANSACEstimator(args.pnpransac_samples, self.max_pnp_error, EstimatorType.P4P)

        # Rig-aware MiniBA for bootstrap (Option A, TODO §3.3.1).
        self.rig_config = rig_config
        if rig_config is not None:
            self._rig_view_names = list(rig_config.view_names)
            n_views = len(self._rig_view_names)
            # MiniBARig seeds 3D points from one observation per view, so the
            # effective npts is a multiple of n_views.
            self._rig_npts_per_view = args.num_pts_miniba_bootstrap // n_views
            self._rig_bootstrap_npts = self._rig_npts_per_view * n_views
            self.miniba_bootstrap_rig = MiniBARig(
                n_ts=args.num_keyframes_miniba_bootstrap,
                n_views=n_views,
                npts=self._rig_bootstrap_npts,
                optimize_focal=(not args.fix_focal),
                optimize_3Dpts=True,
                iters=args.iters_miniba_bootstrap,
            )
            # Issue #5 refinement: 1-timestep MiniBARig for per-step rig_pose
            # refinement after rig_pnp_per_view. Pose-only (3D pts fixed).
            self._rig_incr_npts = (args.num_pts_miniba_incr // n_views) * n_views
            self.miniba_incr_rig = MiniBARig(
                n_ts=1,
                n_views=n_views,
                npts=self._rig_incr_npts,
                optimize_focal=False,
                optimize_3Dpts=False,
                iters=args.iters_miniba_incr,
            )
            # Track incremental refinement health. >10% fallback rate is a
            # signal to investigate BA hyperparameters or matcher quality.
            self._refine_call_count = 0
            self._refine_fail_count = 0

    def build_problem(self,
                      desc_kpts_list: list[DescribedKeypoints],
                      npts: int,
                      n_cams: int,
                      n_primary_cam: int,
                      min_n_matches: int,
                      kfId_list: list[int],
    ):
        """Build the problem for mini ba by organizing the matches between the keypoints of the cameras."""
        npts_per_primary_cam = npts // n_primary_cam
        uvs = torch.zeros(npts, n_cams, 2, device='cuda') - 1
        xyz_indices = torch.zeros(npts, n_cams, dtype=torch.int64, device='cuda') - 1
        unused_kpts_mask = torch.ones((n_cams, desc_kpts_list[0].kpts.shape[0]), device='cuda', dtype=torch.bool)
        for k in range(n_primary_cam):
            idx_occurrences = torch.zeros(self.num_kpts, device="cuda", dtype=torch.int)
            for match in desc_kpts_list[k].matches.values():
                idx_occurrences[match.idx] += 1
            idx_occurrences *= unused_kpts_mask[k]
            if idx_occurrences.sum() == 0:
                print("No matches.")
                continue
            idx_occurrences = idx_occurrences > 0
            selected_indices = torch.multinomial(idx_occurrences.float(), npts_per_primary_cam, replacement=False)

            selected_mask = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
            selected_mask[selected_indices] = True
            aligned_ids = torch.arange(npts_per_primary_cam, device="cuda")
            all_aligned_ids = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
            all_aligned_ids[selected_indices] = aligned_ids

            uvs_k = uvs[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam, :, :]
            xyz_indices_k = xyz_indices[k*npts_per_primary_cam:(k+1)*npts_per_primary_cam]
            for l in range(n_cams):
                if l == k:
                    uvs_k[:, l, :] = desc_kpts_list[l].kpts[selected_indices]
                    xyz_indices_k[:, l] = selected_indices
                else:
                    lId = kfId_list[l]
                    if lId in desc_kpts_list[k].matches:
                        idxk = desc_kpts_list[k].matches[lId].idx
                        idxl = desc_kpts_list[k].matches[lId].idx_other

                        mask = selected_mask[idxk] 
                        idxk = idxk[mask]
                        idxl = idxl[mask]

                        set_idx = all_aligned_ids[idxk]
                        unused_kpts_mask[l, idxl] = False
                        uvs_k[set_idx, l, :] = desc_kpts_list[l].kpts[idxl]
                        xyz_indices_k[set_idx, l] = idxl

                        selected_indices_l = idxl.clone()
                        selected_mask_l = torch.zeros(self.num_kpts, device='cuda', dtype=torch.bool)
                        selected_mask_l[selected_indices_l] = True
                        all_aligned_ids_l = torch.zeros(self.num_kpts, device="cuda", dtype=aligned_ids.dtype)
                        all_aligned_ids_l[selected_indices_l] = set_idx.clone()

                        for m in range(l + 1, n_cams):
                            mId = kfId_list[m]
                            if mId in desc_kpts_list[l].matches:
                                idxl = desc_kpts_list[l].matches[mId].idx
                                idxm = desc_kpts_list[l].matches[mId].idx_other

                                mask = selected_mask_l[idxl] 
                                idxl = idxl[mask]
                                idxm = idxm[mask]

                                set_idx = all_aligned_ids_l[idxl]
                                set_mask = uvs_k[set_idx, m, 0] == -1
                                uvs_k[set_idx[set_mask], m, :] = desc_kpts_list[m].kpts[idxm[set_mask]]

        n_valid = (uvs >= 0).all(dim=-1).sum(dim=-1)
        mask = n_valid < min_n_matches
        uvs[mask, :, :] = -1
        xyz_indices[mask, :] = -1
        return uvs, xyz_indices

    @torch.no_grad()
    def initialize_bootstrap(self, desc_kpts_list: list[DescribedKeypoints], rebooting=False):
        """
        Estimate focal and initialize the poses of the frames corresponding to desc_kpts_list. 
        """
        n_cams = len(desc_kpts_list)
        npts = self.num_pts_miniba_bootstrap

        ## Exhaustive matching
        for i in range(n_cams):
            for j in range(i + 1, n_cams):
                _ = self.matcher(desc_kpts_list[i], desc_kpts_list[j], remove_outliers=True, update_kpts_flag="inliers", kID=i, kID_other=j)
        
        ## Build the problem by organizing matches
        uvs, xyz_indices = self.build_problem(desc_kpts_list, npts, n_cams, n_cams, 2, list(range(n_cams)))

        ## Initialize for miniBA (poses at identity, 3D points with rand depth)
        f_init = (torch.tensor([self.f_init], device="cuda"))
        Rs6D_init = torch.eye(3, 2, device="cuda")[None].repeat(n_cams, 1, 1)
        ts_init = torch.zeros(n_cams, 3, device="cuda")

        xyz_init = torch.zeros(npts, 3, device="cuda")
        for k in range(n_cams):
            mask = (uvs[:, k, :] >= 0).all(dim=-1)
            xyz_init[mask] += depth2points(uvs[mask, k, :], 1, f_init, self.centre)
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        ## Run miniBA, estimating 3D points, camera focal and poses
        if rebooting:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_rebooting(Rs6D_init, ts_init, self.f, xyz_init, self.centre, uvs.view(-1))
        else:
            Rs6D, ts, f, xyz, r, r_init, mask = self.miniba_bootstrap(Rs6D_init, ts_init, f_init, xyz_init, self.centre, uvs.view(-1))
        final_residual = (r * mask).abs().sum()/mask.sum()

        self.f = f
        self.intrinsics = torch.cat([f, self.centre], dim=0)

        ## Scale to 0.1 average translation
        rel_ts = ts[:-1] - ts[1:]
        scale = 0.1 / rel_ts.norm(dim=-1).mean()
        ts *= scale
        xyz = scale * xyz.clone()
        Rts = torch.eye(4, device="cuda")[None].repeat(n_cams, 1, 1)
        Rts[:, :3, :3] = sixD2mtx(Rs6D)
        Rts[:, :3, 3] = ts

        return Rts, f, final_residual

    @torch.no_grad()
    def initialize_incremental(self, keyframes: list[Keyframe], curr_desc_kpts: DescribedKeypoints, index: int, is_test: bool, curr_img):
        """
        Initialize the pose of the frame given by curr_desc_kpts and index using the previously registered keyframes.
        """
        
        # Match the current frame with previous keyframes
        xyz = []
        uvs = []
        confs = []
        match_indices = []
        for keyframe in keyframes:
            matches = self.matcher(curr_desc_kpts, keyframe.desc_kpts, remove_outliers=True, update_kpts_flag="all", kID=index, kID_other=keyframe.index)

            mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
            xyz.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
            uvs.append(matches.kpts[mask])
            confs.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            match_indices.append(matches.idx[mask])

        xyz = torch.cat(xyz, dim=0)
        uvs = torch.cat(uvs, dim=0)
        confs = torch.cat(confs, dim=0)
        match_indices = torch.cat(match_indices, dim=0)

        # Subsample the points if there are too many
        if len(xyz) > self.num_pts_pnpransac:
            selected_indices = torch.multinomial(confs, self.num_pts_miniba_incr, replacement=False)
            xyz = xyz[selected_indices]
            uvs = uvs[selected_indices]
            confs = confs[selected_indices]
            match_indices = match_indices[selected_indices]

        # Estimate an initial camera pose and inliers using PnP RANSAC
        Rs6D_init = keyframes[0].rW2C
        ts_init = keyframes[0].tW2C
        Rt, inliers = self.PnPRANSAC(uvs, xyz, self.f, self.centre, Rs6D_init, ts_init, confs)

        xyz = xyz[inliers]
        uvs = uvs[inliers]
        confs = confs[inliers]
        match_indices = match_indices[inliers]

        # Subsample the points if there are too many
        if len(xyz) >= self.num_pts_miniba_incr:
            selected_indices = torch.topk(torch.rand_like(xyz[..., 0]), self.num_pts_miniba_incr, dim=0, largest=False)[1]
            xyz_ba = xyz[selected_indices]
            uvs_ba = uvs[selected_indices]
        elif len(xyz) < self.num_pts_miniba_incr:
            xyz_ba = torch.cat([xyz, torch.zeros(self.num_pts_miniba_incr - len(xyz), 3, device="cuda")], dim=0)
            uvs_ba = torch.cat([uvs, -torch.ones(self.num_pts_miniba_incr - len(uvs), 2, device="cuda")], dim=0)

        # Run the initialization
        Rs6D, ts = Rt[:3, :2][None], Rt[:3, 3][None]
        Rs6D, ts, _, _, r, r_init, mask = self.miniBA_incr(Rs6D, ts, self.f, xyz_ba, self.centre, uvs_ba.view(-1))
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = sixD2mtx(Rs6D)[0]
        Rt[:3, 3] = ts[0]

        # Check if we have sufficiently many inliers
        if is_test or mask.sum() > self.min_num_inliers:
            # Return the pose of the current frame
            return Rt
        else:
            print("Too few inliers for pose initialization")
            # Remove matches as we prevent the current frame from being registered
            for keyframe in keyframes:
                keyframe.desc_kpts.matches.pop(index, None)
            return None

    @torch.no_grad()
    def initialize_bootstrap_rig(self, desc_kpts_per_ts_per_view, rig_config,
                                 mono_idepth_per_ts_per_view=None):
        """Rig-aware bootstrap (Option A, TODO §3.3.1).

        Args:
            desc_kpts_per_ts_per_view: list length N_ts of dict
                {view_name: DescribedKeypoints}. All 9 views present per ts.
            rig_config: RigConfig with fixed relative_Rt per view.
            mono_idepth_per_ts_per_view: optional list[dict[view_name,
                Tensor (1, 1, H, W)]] of monocular *inverse* depth maps,
                already interpolated to image (H, W). When provided, xyz_init
                seeds use the per-pixel mono depth instead of unit depth;
                None falls back to the original unit-depth behaviour.

        Returns:
            rig_Rts (N_ts, 4, 4), f, final_residual, xyz_out (npts, 3),
            view_names (list, matches rig_config.view_names ordering).
        """
        view_names = list(rig_config.view_names)
        N_views = len(view_names)
        N_ts = len(desc_kpts_per_ts_per_view)
        npts_per_view = self.num_pts_miniba_bootstrap // N_views
        actual_npts = npts_per_view * N_views

        per_view_desc_kpts_list = {
            v: [desc_kpts_per_ts_per_view[ts][v] for ts in range(N_ts)]
            for v in view_names
        }

        def make_kID(v_idx, ts):
            return ts * N_views + v_idx

        # Time-axis exhaustive matching within each view — TODO.md §3.3.1.
        # Cross-view matching is intentionally skipped (rotation-only rig has
        # zero baseline intra-timestep, so cross-view matches carry no depth info).
        from poses.matcher import (
            _diag_enabled as _matcher_diag_on,
            reset_matcher_diag, get_matcher_diag,
        )
        diag = _matcher_diag_on()
        if diag:
            reset_matcher_diag()

        for v_idx, v_name in enumerate(view_names):
            dk_list = per_view_desc_kpts_list[v_name]
            for i in range(N_ts):
                for j in range(i + 1, N_ts):
                    _ = self.matcher(
                        dk_list[i], dk_list[j],
                        remove_outliers=True, update_kpts_flag="inliers",
                        kID=make_kID(v_idx, i), kID_other=make_kID(v_idx, j),
                    )

        if diag:
            _print_matcher_diag(get_matcher_diag(), view_names, N_ts)

        uv = torch.full((actual_npts, N_ts, N_views, 2), -1.0, device="cuda")
        xyz_init = torch.zeros(actual_npts, 3, device="cuda")
        f_init_t = torch.tensor([self.f_init], device="cuda")

        rel_R_all = torch.stack([
            rig_config.relative_Rt[v][:3, :3].cuda() for v in view_names
        ])
        rel_t_all = torch.stack([
            rig_config.relative_Rt[v][:3, 3].cuda() for v in view_names
        ])

        # Pre-convert idepth → depth per (ts, view) once so the per-point
        # seed loop just samples. Each (1, 1, H, W) map.
        mono_depth_maps = None
        if mono_idepth_per_ts_per_view is not None:
            mono_depth_maps = [
                {v: relative_idepth_to_depth(ts_dict[v]).float()
                 for v in view_names}
                for ts_dict in mono_idepth_per_ts_per_view
            ]
        n_seeded_mono = 0
        n_seeded_unit = 0

        for v_idx, v_name in enumerate(view_names):
            dk_list = per_view_desc_kpts_list[v_name]
            kIDs = [make_kID(v_idx, i) for i in range(N_ts)]
            uvs_v, _ = self.build_problem(dk_list, npts_per_view, N_ts, N_ts, 2, kIDs)
            lo = v_idx * npts_per_view
            hi = (v_idx + 1) * npts_per_view
            uv[lo:hi, :, v_idx, :] = uvs_v

            # Seed xyz from each point's first valid observation. Depth comes
            # from the pre-computed mono depth map (sampled at the keypoint
            # pixel) when available, else falls back to unit depth.
            valid = (uvs_v >= 0).all(dim=-1)  # (npts_per_view, N_ts)
            for p in range(npts_per_view):
                ts_hits = valid[p].nonzero(as_tuple=False).flatten()
                if ts_hits.numel() == 0:
                    continue
                ts_idx = int(ts_hits[0].item())
                uv_pt = uvs_v[p, ts_idx]

                depth_pt = 1.0
                if mono_depth_maps is not None:
                    depth_map = mono_depth_maps[ts_idx][v_name]
                    sampled = sample(
                        depth_map, uv_pt.view(1, 1, 1, 2),
                        self.width, self.height,
                    )
                    depth_pt = float(sampled[0, 0, 0, 0].item())
                    if not (depth_pt > 0 and torch.isfinite(torch.tensor(depth_pt))):
                        depth_pt = 1.0
                        n_seeded_unit += 1
                    else:
                        n_seeded_mono += 1
                else:
                    n_seeded_unit += 1

                local = depth2points(uv_pt[None], depth_pt, f_init_t, self.centre)[0]
                xyz_init[lo + p] = rel_R_all[v_idx].T @ local

        if mono_depth_maps is not None and bool(int(os.environ.get("RIG_DIAG", "0"))):
            tot = n_seeded_mono + n_seeded_unit
            ratio = (100 * n_seeded_mono / max(tot, 1)) if tot else 0.0
            print(f"[diag/seed] mono-depth seeded {n_seeded_mono}/{tot} "
                  f"({ratio:.1f}%); unit-depth fallback {n_seeded_unit}")

        if diag:
            _print_obs_histograms(uv, view_names, npts_per_view, N_ts)

        # KNOWN ISSUE (E0 diagnostic): for views with rel_R angle > 90° from
        # ref (High_Cam02 at 135°, Low_Cam02 at 108°), the seeded ray's
        # world-z is negative, and the next two lines reflect those points
        # across z=0 to anchor them on the +1 plane. This costs us the
        # geometric meaning of those views' seeds (BA rejects 95% of Cam02
        # observations) but the alternative — keeping the negative-z
        # placement — makes BA's projection gradient ill-conditioned and
        # blows up the LM step (verified empirically: 100% rejection if we
        # drop the force-z=1 anchor without a deeper rewrite of the
        # parameterisation, e.g. inverse depth or per-view world frames).
        # Treat this as accepted lossy behaviour until a structural fix.
        xyz_init /= xyz_init[..., -1:].clamp_min(1)
        xyz_init[..., -1] = 1
        xyz_init *= 1 + torch.randn_like(xyz_init[:, :1]).abs()

        rig_R_init = torch.eye(3, 2, device="cuda")[None].repeat(N_ts, 1, 1).contiguous()
        rig_t_init = torch.zeros(N_ts, 3, device="cuda")

        rig_R_out, rig_t_out, f_out, xyz_out, r, r_init, mask = self.miniba_bootstrap_rig(
            rig_R_init, rig_t_init, f_init_t, xyz_init,
            rel_R_all, rel_t_all, self.centre, uv.reshape(-1).contiguous(),
        )

        if diag:
            _print_e0_rejected_obs(
                uv, r, mask, view_names, N_ts, N_views, actual_npts,
                self.centre, self.width, self.height,
            )

        final_residual = (r * mask).abs().sum() / mask.sum().clamp_min(1)

        self.f = f_out
        self.intrinsics = torch.cat([f_out, self.centre], dim=0)

        # Scale so consecutive rig centers have median distance 0.1.
        # Median is robust to a single bad timestep that bootstrap BA might
        # leave at an outlier position; mean would be dragged by it.
        rel_rig_t = rig_t_out[:-1] - rig_t_out[1:]
        rel_rig_t_med = rel_rig_t.norm(dim=-1).median()
        scale = 0.1 / rel_rig_t_med.clamp_min(1e-6)
        rig_t_out = rig_t_out * scale
        xyz_out = xyz_out * scale

        # Prune BA outlier xyz. The right signal is *reprojection error per
        # point*, not raw distance; far points may still be correct, while
        # close points can be junk if their residual is high. Distance and
        # observation count are kept as a safety net (OR-combined).
        n_obs = N_ts * N_views
        r_view = r.view(actual_npts, n_obs, 2)
        mask_view = mask.view(actual_npts, n_obs, 2)
        r_abs = (r_view * mask_view).abs()
        valid_count = mask_view.sum(dim=(-1, -2)).clamp_min(1)
        per_pt_err = r_abs.sum(dim=(-1, -2)) / valid_count
        per_pt_err_med = per_pt_err.median().clamp_min(1.0)
        outlier_repr = per_pt_err > 5.0 * per_pt_err_med
        xyz_norm = xyz_out.norm(dim=-1)
        xyz_median = xyz_norm.median().clamp_min(0.01)
        outlier_dist = xyz_norm > 30.0 * xyz_median
        outlier_few = mask_view.any(dim=-1).sum(dim=-1) < 2
        outlier_mask = outlier_repr | outlier_dist | outlier_few
        n_outliers = int(outlier_mask.sum().item())
        if n_outliers > 0:
            xyz_out[outlier_mask] = float("nan")
        repr_and_dist = int((outlier_repr & outlier_dist).sum())
        dist_only = int((outlier_dist & ~outlier_repr).sum())
        repr_only = int((outlier_repr & ~outlier_dist).sum())
        print(
            f"[bootstrap_rig] outliers pruned: "
            f"repr={int(outlier_repr.sum())}, "
            f"dist={int(outlier_dist.sum())}, "
            f"few_obs={int(outlier_few.sum())}, "
            f"total={n_outliers}/{actual_npts}"
        )
        print(
            f"[bootstrap_rig] repr/dist overlap: "
            f"repr_and_dist={repr_and_dist}, "
            f"repr_only={repr_only}, "
            f"dist_only={dist_only}"
        )

        rig_Rts = torch.eye(4, device="cuda")[None].repeat(N_ts, 1, 1)
        rig_Rts[:, :3, :3] = sixD2mtx(rig_R_out)
        rig_Rts[:, :3, 3] = rig_t_out

        return rig_Rts, f_out, final_residual, xyz_out, view_names

    @torch.no_grad()
    def initialize_incremental_rig(
        self,
        keyframes,
        desc_kpts_per_view,
        view_indices,
        rig_config,
    ):
        """Rig-aware incremental pose using method A from TODO §3.3.2.

        Each view is matched independently against the previous keyframes;
        per-view 2D-3D correspondences are fed to `rig_pnp_per_view`, which
        lifts each view's PnP to a rig pose candidate and averages them on
        SE(3) with an IRLS Huber kernel.

        Args:
            keyframes: either a flat list of previous Keyframes (all views
                share the same candidate pool), or a dict
                {view_name: list[Keyframe]} where each view has its own pool.
                The dict form is preferred when the camera is turning — each
                view picks keyframes that best match its own orientation.
            desc_kpts_per_view: dict {view_name: DescribedKeypoints}.
            view_indices: dict {view_name: int} — kID used by the matcher so
                that match dicts stay disjoint per view.
            rig_config: RigConfig with fixed per-view relative Rt.

        Returns: (rig_w2c 4x4 Tensor or None, per-view stats dict).
        """
        # Normalize keyframes argument to a dict.
        if isinstance(keyframes, dict):
            keyframes_per_view = keyframes
        else:
            keyframes_per_view = {v: keyframes for v in desc_kpts_per_view}

        correspondences = {}
        for view_name, curr_desc_kpts in desc_kpts_per_view.items():
            xyz_list, uv_list, conf_list = [], [], []
            my_idx = view_indices[view_name]
            for keyframe in keyframes_per_view[view_name]:
                matches = self.matcher(
                    curr_desc_kpts, keyframe.desc_kpts,
                    remove_outliers=True, update_kpts_flag="all",
                    kID=my_idx, kID_other=keyframe.index,
                )
                mask = keyframe.desc_kpts.has_pt3d[matches.idx_other]
                if mask.sum() == 0:
                    continue
                xyz_list.append(keyframe.desc_kpts.pts3d[matches.idx_other[mask]])
                uv_list.append(matches.kpts[mask])
                conf_list.append(keyframe.desc_kpts.pts_conf[matches.idx_other[mask]])
            if xyz_list:
                xyz = torch.cat(xyz_list, dim=0)
                uv = torch.cat(uv_list, dim=0)
                conf = torch.cat(conf_list, dim=0)
                if len(xyz) > self.num_pts_pnpransac:
                    sel = torch.multinomial(conf, self.num_pts_pnpransac, replacement=False)
                    xyz = xyz[sel]
                    uv = uv[sel]
                correspondences[view_name] = (uv, xyz)
            else:
                correspondences[view_name] = (
                    torch.zeros(0, 2, device="cuda"),
                    torch.zeros(0, 3, device="cuda"),
                )

        f = self.f.item() if isinstance(self.f, torch.Tensor) else float(self.f)
        K = torch.tensor(
            [[f, 0.0, self.centre[0].item()],
             [0.0, f, self.centre[1].item()],
             [0.0, 0.0, 1.0]], dtype=torch.float32, device="cuda",
        )
        rig_pose, stats = rig_pnp_per_view(
            correspondences, rig_config, K,
            min_correspondences=max(self.min_num_inliers // 4, 6),
            reproj_error_px=float(self.max_pnp_error),
        )

        if rig_pose is None:
            for view_name, my_idx in view_indices.items():
                for keyframe in keyframes_per_view[view_name]:
                    keyframe.desc_kpts.matches.pop(my_idx, None)
            return None, stats

        # Issue #5: refine rig_pose with 1-timestep MiniBARig LM (pose-only).
        # Takes PnP+Fréchet mean output as initial guess, tightens it with
        # full reprojection residual across all 9 views simultaneously.
        if hasattr(self, "miniba_incr_rig"):
            self._refine_call_count += 1
            try:
                rig_pose = self._refine_rig_pose_miniba(rig_pose, correspondences, rig_config)
            except Exception as e:
                # Refinement is a best-effort tightening; fall back to PnP
                # result on any numerical hiccup instead of dropping the ts.
                self._refine_fail_count += 1
                fail_rate = self._refine_fail_count / max(self._refine_call_count, 1)
                print(
                    f"rig incr refinement failed ({e}); fallback to PnP. "
                    f"cumulative fail rate: "
                    f"{self._refine_fail_count}/{self._refine_call_count} "
                    f"({100 * fail_rate:.1f}%)"
                )
        return rig_pose, stats

    @torch.no_grad()
    def _refine_rig_pose_miniba(self, rig_pose_init, correspondences, rig_config):
        """1-timestep MiniBARig LM refinement of the rig_pose.

        Layout for MiniBARig (n_ts=1): observation buffer is
        (npts, n_ts=1, n_views, 2) where `-1` means no observation. We fill
        each view's strip with that view's correspondences.
        """
        view_names = list(rig_config.view_names)
        n_views = len(view_names)
        npts = self._rig_incr_npts
        pts_per_view = npts // n_views

        uv = torch.full((npts, 1, n_views, 2), -1.0, device="cuda")
        xyz = torch.zeros(npts, 3, device="cuda")

        any_obs = False
        for v_idx, v_name in enumerate(view_names):
            if v_name not in correspondences:
                continue
            uv_v, xyz_v = correspondences[v_name]
            if uv_v.shape[0] == 0:
                continue
            n = min(pts_per_view, uv_v.shape[0])
            lo = v_idx * pts_per_view
            uv[lo:lo + n, 0, v_idx, :] = uv_v[:n]
            xyz[lo:lo + n] = xyz_v[:n]
            any_obs = True
        if not any_obs:
            return rig_pose_init

        R_init = rig_pose_init[:3, :3].contiguous()
        t_init = rig_pose_init[:3, 3].contiguous()
        R6D_init = mtx2sixD(R_init.unsqueeze(0))  # (1, 3, 2)
        t_in = t_init.unsqueeze(0)                # (1, 3)

        rel_R = torch.stack(
            [rig_config.relative_Rt[v][:3, :3].to("cuda") for v in view_names]
        )
        rel_t = torch.stack(
            [rig_config.relative_Rt[v][:3, 3].to("cuda") for v in view_names]
        )

        f_t = self.f if isinstance(self.f, torch.Tensor) else torch.tensor(
            [float(self.f)], device="cuda"
        )
        R6D_out, t_out, _, _, r, r_init, mask = self.miniba_incr_rig(
            R6D_init, t_in, f_t, xyz, rel_R, rel_t, self.centre,
            uv.reshape(-1).contiguous(),
        )

        # Sanity: if output has NaNs, or if refinement moved the translation
        # by an unreasonable amount relative to the PnP initial (e.g. MiniBA
        # diverged on degenerate inputs), keep the PnP result.
        if not torch.isfinite(t_out).all() or not torch.isfinite(R6D_out).all():
            return rig_pose_init
        t_jump = float((t_out[0] - t_init).norm().item())
        init_t_norm = float(t_init.norm().item())
        # Allow refinement to move up to max(0.2, |t_init|)
        max_jump = max(0.2, init_t_norm * 1.0)
        if t_jump > max_jump:
            return rig_pose_init

        rig_pose_out = torch.eye(4, device="cuda")
        rig_pose_out[:3, :3] = sixD2mtx(R6D_out)[0]
        rig_pose_out[:3, 3] = t_out[0]
        return rig_pose_out