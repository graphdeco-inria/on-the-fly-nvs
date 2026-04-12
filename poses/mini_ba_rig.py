"""Rig-aware MiniBA (TODO.md §3.3.1).

Keeps the structure of mini_ba.MiniBA (autograd jacfwd, LM+Schur, CUDA graph)
but changes the meaning of the "camera" dimension:

- Observations are indexed by (timestep, view). There are n_obs = n_ts * n_views
  observations per 3D point.
- Optimizable camera parameters are `rig_pose` per timestep — there are
  n_opt_rig = n_ts of them (6D rotation + 3D translation = 9 params each).
- Per-view relative transforms `rel_R[view]` / `rel_t[view]` are fixed; each
  observation's world-to-camera pose is `view_w2c = rel[view] @ rig[ts]`.
- Cross-term zeroing: the Jacobian of observation (ts, view) wrt rig_pose[k]
  is non-zero only when k == ts. Same block-diagonal trick as mini_ba.py, but
  keyed on `ts_of_obs = obs_idx // n_views` instead of `cam_idx`.

Focal and translation parts of rel are kept for generality. For the Insta360
X5 rig `rel_t == 0` (rotation-only).
"""

from utils import mtx2sixD, pts2px, sixD2mtx

import torch
import torch.nn as nn
from torch.func import jacfwd, vmap


def project_rig(xyz, rig_R6D_t, rel_R, rel_t, f, centre):
    """Single-observation projection: xyz in world, via rig[ts] composed with rel[view]."""
    rig_R = sixD2mtx(rig_R6D_t[:6].reshape(3, 2))
    rig_t = rig_R6D_t[6:9]
    view_R = rel_R @ rig_R
    view_t = rel_R @ rig_t + rel_t
    xyz_local = view_R @ xyz + view_t
    return pts2px(xyz_local, f, centre)


def get_residual(xyz, rig_R6D_t, rel_R, rel_t, f, centre, uv):
    return project_rig(xyz, rig_R6D_t, rel_R, rel_t, f, centre) - uv


def get_residual2(xyz, rig_R6D_t, rel_R, rel_t, f, centre, uv):
    err = get_residual(xyz, rig_R6D_t, rel_R, rel_t, f, centre, uv)
    return err, err


class MiniBARigInternal(nn.Module):
    def __init__(
        self,
        n_ts,
        n_views,
        npts,
        optimize_focal,
        optimize_3Dpts,
        huber_delta,
        outlier_mad_scale,
        lm,
        ep,
        k,
        iters,
    ):
        super().__init__()
        self.n_ts = n_ts
        self.n_views = n_views
        self.n_obs = n_ts * n_views
        self.npts = npts
        self.optimize_focal = optimize_focal
        self.optimize_3Dpts = optimize_3Dpts
        self.huber_delta = huber_delta
        self.outlier_mad_scale = outlier_mad_scale
        self.lm = lm
        self.ep = ep
        self.k = k
        self.iters = iters
        self.n_cam_params = n_ts * 9 + (1 if optimize_focal else 0)

        # argnums: (xyz=0, rig_R6D_t=1, rel_R=2, rel_t=3, f=4, centre=5, uv=6)
        argnums = (1,)
        self.param2id = {"poses": len(argnums) - 1}
        if optimize_focal:
            argnums += (4,)
            self.param2id["focal"] = len(argnums) - 1
        if optimize_3Dpts:
            argnums += (0,)
            self.param2id["xyz"] = len(argnums) - 1

        self.get_residual_jacobian = vmap(
            vmap(jacfwd(get_residual2, has_aux=True, argnums=argnums))
        )
        self.get_residual = vmap(vmap(get_residual))

        # Precompute the cross-term mask (obs_idx j, rig_idx l): True when ts(j)==l.
        # Shape compatible with the duv_dcam expansion in `optimize`.
        j = torch.arange(self.n_obs).view(1, -1, 1, 1, 1)
        l = torch.arange(self.n_ts).view(1, 1, 1, -1, 1)
        # ts_of_obs[j] = j // n_views
        ts_of_obs = j // self.n_views
        self.register_buffer("_cross_mask", ts_of_obs == l, persistent=False)

    def prepare_for_proj(self, xyz, rig_R6D_t, rel_R, rel_t, f, centre):
        """Expand inputs to (npts, n_obs, ...) for the double-vmap projection."""
        # xyz: (npts, 3) -> (npts, n_obs, 3)
        xyz_e = xyz.unsqueeze(1).expand(-1, self.n_obs, *xyz.shape[1:])
        # rig_R6D_t: (n_ts, 9) -> (npts, n_obs=ts*view, 9) where obs_idx j maps to ts=j//n_views
        #   Build (n_ts, n_views, 9) by view-expansion, then flatten ts*view.
        rig_e = rig_R6D_t.unsqueeze(1).expand(-1, self.n_views, -1).reshape(self.n_obs, 9)
        rig_e = rig_e[None].expand(self.npts, -1, -1)
        # rel_R: (n_views, 3, 3) -> (npts, n_obs, 3, 3); obs j -> view = j % n_views
        rel_R_e = rel_R.unsqueeze(0).expand(self.n_ts, -1, -1, -1).reshape(self.n_obs, 3, 3)
        rel_R_e = rel_R_e[None].expand(self.npts, -1, -1, -1)
        rel_t_e = rel_t.unsqueeze(0).expand(self.n_ts, -1, -1).reshape(self.n_obs, 3)
        rel_t_e = rel_t_e[None].expand(self.npts, -1, -1)
        # f scalar, centre (2,) -> broadcast to (npts, n_obs, ...)
        f_e = f[None, None].expand(self.npts, self.n_obs, *f.shape)
        centre_e = centre[None, None].expand(self.npts, self.n_obs, *centre.shape)
        return xyz_e, rig_e, rel_R_e, rel_t_e, f_e, centre_e

    def get_mask(self, r_in, original_mask2):
        if self.outlier_mad_scale > 0:
            err = torch.linalg.vector_norm(
                r_in.view(-1, 2) * original_mask2.view(-1)[:, None],
                dim=-1, keepdim=True,
            ).view(self.npts, self.n_obs)
            q = 1 - 0.5 * original_mask2.float().mean(0)
            med = torch.quantile(err.T, q)
            mad = torch.quantile(torch.abs(err - med).T, q)
            c = med + self.outlier_mad_scale * mad
            c.clamp_min_(5)
            mask = original_mask2 * (err < c[None])
            mask = mask[..., None].expand(-1, -1, 2).reshape(-1).float()
            return mask
        return original_mask2.expand(-1, -1, 2).reshape(-1).float()

    def get_huber_weights(self, r):
        if self.huber_delta > 0:
            r_abs = r.abs()
            return torch.where(
                r_abs <= self.huber_delta, 1, self.huber_delta / r_abs.sqrt()
            )
        return torch.ones_like(r)

    def optimize(self, rig_R6D, rig_ts, f, xyz, rel_R, rel_t, centre, uv):
        """Run LM iterations. Shapes:
            rig_R6D: (n_ts, 3, 2)   rig_ts: (n_ts, 3)
            rel_R: (n_views, 3, 3)  rel_t: (n_views, 3)
            xyz: (npts, 3)          f: (1,)   centre: (2,)
            uv: (npts * n_obs * 2,) flattened.
        """
        uv = uv.view(self.npts, self.n_obs, 2)
        original_mask2 = (uv >= 0).all(dim=-1)
        lm = self.lm

        for iteration in range(self.iters):
            rig_R6D_t = torch.cat([rig_R6D.view(-1, 6), rig_ts], dim=-1)  # (n_ts, 9)

            jacobian_elements, r_in = self.get_residual_jacobian(
                *self.prepare_for_proj(xyz, rig_R6D_t, rel_R, rel_t, f, centre), uv
            )
            r_in = r_in.view(-1)
            # jacobian_elements[poses] shape: (npts, n_obs, 2, 9) — local Jacobian wrt the
            # rig pose *assigned to this observation's timestep*. We need to place it in
            # the right n_ts-block and zero elsewhere (block-diagonal structure).
            duv_drig = jacobian_elements[self.param2id["poses"]]
            duv_drig = duv_drig.unsqueeze(-2).repeat(1, 1, 1, self.n_ts, 1)  # (npts, n_obs, 2, n_ts, 9)
            duv_drig = torch.where(self._cross_mask, duv_drig, torch.zeros_like(duv_drig))
            duv_drig = duv_drig.reshape(*duv_drig.shape[:-2], -1)  # (npts, n_obs, 2, n_ts*9)

            if self.optimize_focal:
                duv_drig = torch.cat(
                    [duv_drig, jacobian_elements[self.param2id["focal"]]], dim=-1
                )
            if self.optimize_3Dpts:
                duv_dxyz = jacobian_elements[self.param2id["xyz"]]

            if iteration == 0:
                initial_r = r_in.clone()

            weights = self.get_huber_weights(r_in)
            mask = self.get_mask(r_in, original_mask2)
            weights = mask * weights
            r = r_in * weights.reshape(-1)
            duv_drig *= weights.reshape(self.npts, self.n_obs, 2, 1)
            if self.optimize_3Dpts:
                duv_dxyz *= weights.reshape(self.npts, self.n_obs, 2, 1)

            duv_drig_flat = duv_drig.reshape(-1, self.n_cam_params)
            duv_drig_reshaped = duv_drig.reshape(
                self.npts, self.n_obs * 2, self.n_cam_params
            )

            jtj_cam = duv_drig_flat.T @ duv_drig_flat
            if self.optimize_3Dpts:
                duv_dxyz = duv_dxyz.reshape(self.npts, self.n_obs * 2, -1)
                jtj_xyz = torch.bmm(duv_dxyz.transpose(1, 2), duv_dxyz)
                jtj_cam_xyz = (
                    torch.bmm(duv_drig_reshaped.transpose(1, 2), duv_dxyz)
                    .transpose(1, 2)
                    .reshape(-1, self.n_cam_params)
                )

            jtj_cam.diagonal().mul_(1 + lm)
            jtj_cam.diagonal().clamp_min_(self.ep)
            if self.optimize_3Dpts:
                jtj_xyz.diagonal(dim1=-2, dim2=-1).mul_(1 + lm)
                jtj_xyz.diagonal(dim1=-2, dim2=-1).clamp_min_(self.ep)

            jacxr_cam = duv_drig_flat.T @ r
            if self.optimize_3Dpts:
                jacxr_xyz = torch.bmm(
                    duv_dxyz.transpose(1, 2),
                    r.view(self.npts, self.n_obs * 2).unsqueeze(-1),
                ).view(-1)

                jtj_xyz_inv = torch.linalg.inv_ex(jtj_xyz)[0]
                jtj_xyz_inv.nan_to_num_()

                BD = (
                    torch.bmm(jtj_xyz_inv, jtj_cam_xyz.view(self.npts, 3, -1))
                    .view(-1, self.n_cam_params)
                    .T
                )
                BmECm1Et = jtj_cam - BD @ jtj_cam_xyz
                BmECm1Et_inv = torch.linalg.inv_ex(BmECm1Et)[0]
                BmECm1Et_inv.nan_to_num_()

                vmECm1w = jacxr_cam - BD @ jacxr_xyz
                dcam = BmECm1Et_inv @ vmECm1w
                dcam.nan_to_num_()

                b = jacxr_xyz - jtj_cam_xyz @ dcam
                dxyz = torch.bmm(b.view(self.npts, 1, 3), jtj_xyz_inv).view(xyz.shape)
                dxyz.nan_to_num_()
            else:
                dcam = torch.linalg.inv_ex(jtj_cam)[0] @ jacxr_cam
                dcam.nan_to_num_()
                dxyz = 0

            dpose = dcam[: self.n_ts * 9].view(self.n_ts, 9)
            dR = dpose[..., :6].view(-1, 3, 2)
            dt = dpose[..., 6:]
            df = dcam[-1] if self.optimize_focal else 0

            rig_R6D_tmp = rig_R6D.clone() - dR
            rig_ts_tmp = rig_ts.clone() - dt
            f_tmp = f.clone() - df
            xyz_tmp = xyz - dxyz

            rig_R6D_t_tmp = torch.cat([rig_R6D_tmp.view(-1, 6), rig_ts_tmp], dim=-1)
            new_r = self.get_residual(
                *self.prepare_for_proj(xyz_tmp, rig_R6D_t_tmp, rel_R, rel_t, f_tmp, centre),
                uv,
            ).view(-1)
            weights = self.get_huber_weights(new_r) * mask
            new_r = new_r * weights
            success_mask = ((new_r ** 2).mean() < (r ** 2).mean()) * (f_tmp > 0)

            rig_R6D = rig_R6D - success_mask * dR
            rig_ts = rig_ts - success_mask * dt
            f = f - success_mask * df
            xyz = xyz - success_mask * dxyz

            lm *= (1 / self.k) * success_mask + self.k * (1 - success_mask.to(rig_ts))
            rig_R6D = mtx2sixD(sixD2mtx(rig_R6D))

        rig_R6D_t_final = torch.cat([rig_R6D.view(-1, 6), rig_ts], dim=-1)
        r = self.get_residual(
            *self.prepare_for_proj(xyz, rig_R6D_t_final, rel_R, rel_t, f, centre), uv
        ).view(-1)
        mask = self.get_mask(r, original_mask2)
        return rig_R6D, rig_ts, f, xyz, r, initial_r, mask

    def forward(self, rig_R6D, rig_ts, f, xyz, rel_R, rel_t, centre, uv):
        return self.optimize(rig_R6D, rig_ts, f, xyz, rel_R, rel_t, centre, uv)


class MiniBARig:
    """Eager wrapper. No CUDA graph capture yet — rig sizes vary between bootstrap
    and incremental phases, and the tighter integration can be added later once
    this version is validated."""

    @torch.no_grad()
    def __init__(
        self,
        n_ts,
        n_views,
        npts,
        optimize_focal,
        optimize_3Dpts,
        huber_delta=1.0,
        outlier_mad_scale=4,
        lm=1e-5,
        ep=1e-2,
        k=2,
        iters=200,
    ):
        self.optimizer = MiniBARigInternal(
            n_ts, n_views, npts, optimize_focal, optimize_3Dpts,
            huber_delta, outlier_mad_scale, lm, ep, k, iters,
        ).eval().cuda()

    @torch.no_grad()
    def __call__(self, rig_R6D, rig_ts, f, xyz, rel_R, rel_t, centre, uv):
        return self.optimizer(rig_R6D, rig_ts, f, xyz, rel_R, rel_t, centre, uv)
