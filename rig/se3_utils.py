"""Minimal SE(3) log/exp and weighted Fréchet mean for pose averaging.

Only what rig_pnp needs. Shapes are 4x4 homogeneous matrices.
"""

import torch
from torch import Tensor


_EPS = 1e-7


def _so3_log(R: Tensor) -> Tensor:
    # Rodrigues inverse. Returns axis-angle vector in R^3.
    cos = torch.clamp((R.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) * 0.5, -1.0, 1.0)
    theta = torch.acos(cos)
    small = theta.abs() < 1e-4
    # Near identity: use 1st-order expansion.
    k = torch.where(small, torch.ones_like(theta) * 0.5,
                    theta / (2.0 * torch.sin(theta).clamp(min=_EPS)))
    w = torch.stack([
        k * (R[..., 2, 1] - R[..., 1, 2]),
        k * (R[..., 0, 2] - R[..., 2, 0]),
        k * (R[..., 1, 0] - R[..., 0, 1]),
    ], dim=-1)
    return w


def _so3_exp(w: Tensor) -> Tensor:
    theta = w.norm(dim=-1, keepdim=True).clamp(min=_EPS)
    k = w / theta
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -k[..., 2]; K[..., 0, 2] =  k[..., 1]
    K[..., 1, 0] =  k[..., 2]; K[..., 1, 2] = -k[..., 0]
    K[..., 2, 0] = -k[..., 1]; K[..., 2, 1] =  k[..., 0]
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(K)
    s = torch.sin(theta).unsqueeze(-1)
    c = torch.cos(theta).unsqueeze(-1)
    return I + s * K + (1 - c) * (K @ K)


def se3_log(T: Tensor) -> Tensor:
    """4x4 -> 6-vector (rotation 3, translation 3).

    Uses the standard closed-form V^{-1} t so that `exp(log(T)) == T`.
    """
    R = T[..., :3, :3]
    t = T[..., :3, 3]
    w = _so3_log(R)
    theta = w.norm(dim=-1, keepdim=True)
    # V = I + (1-cos)/theta^2 * K + (theta-sin)/theta^3 * K^2
    K = torch.zeros_like(R)
    K[..., 0, 1] = -w[..., 2]; K[..., 0, 2] =  w[..., 1]
    K[..., 1, 0] =  w[..., 2]; K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]; K[..., 2, 1] =  w[..., 0]
    th2 = (theta * theta).clamp(min=_EPS)
    th3 = (theta * th2).clamp(min=_EPS)
    A = (1 - torch.cos(theta)) / th2
    B = (theta - torch.sin(theta)) / th3
    # Handle tiny theta with Taylor expansion.
    small = theta.squeeze(-1) < 1e-4
    A = torch.where(small.unsqueeze(-1), torch.full_like(A, 0.5), A)
    B = torch.where(small.unsqueeze(-1), torch.full_like(B, 1.0 / 6.0), B)
    I = torch.eye(3, device=T.device, dtype=T.dtype).expand_as(R)
    V = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)
    u = torch.linalg.solve(V, t.unsqueeze(-1)).squeeze(-1)
    return torch.cat([w, u], dim=-1)


def se3_exp(xi: Tensor) -> Tensor:
    w = xi[..., :3]
    u = xi[..., 3:]
    R = _so3_exp(w)
    theta = w.norm(dim=-1, keepdim=True)
    K = torch.zeros(*w.shape[:-1], 3, 3, device=w.device, dtype=w.dtype)
    K[..., 0, 1] = -w[..., 2]; K[..., 0, 2] =  w[..., 1]
    K[..., 1, 0] =  w[..., 2]; K[..., 1, 2] = -w[..., 0]
    K[..., 2, 0] = -w[..., 1]; K[..., 2, 1] =  w[..., 0]
    th2 = (theta * theta).clamp(min=_EPS)
    th3 = (theta * th2).clamp(min=_EPS)
    A = (1 - torch.cos(theta)) / th2
    B = (theta - torch.sin(theta)) / th3
    small = theta.squeeze(-1) < 1e-4
    A = torch.where(small.unsqueeze(-1), torch.full_like(A, 0.5), A)
    B = torch.where(small.unsqueeze(-1), torch.full_like(B, 1.0 / 6.0), B)
    I = torch.eye(3, device=w.device, dtype=w.dtype).expand_as(R)
    V = I + A.unsqueeze(-1) * K + B.unsqueeze(-1) * (K @ K)
    t = (V @ u.unsqueeze(-1)).squeeze(-1)
    T = torch.eye(4, device=w.device, dtype=w.dtype).repeat(*w.shape[:-1], 1, 1)
    T[..., :3, :3] = R
    T[..., :3, 3] = t
    return T


def se3_weighted_mean(Ts: Tensor, weights: Tensor, iters: int = 3) -> Tensor:
    """Weighted Fréchet mean on SE(3) via iterated tangent-space averaging.

    Ts: (N, 4, 4), weights: (N,) non-negative, not all zero.
    Returns: (4, 4).
    """
    assert Ts.dim() == 3 and Ts.shape[-2:] == (4, 4)
    w = weights.to(Ts) / weights.sum().clamp(min=_EPS)
    # Initialize at the max-weight pose.
    ref_idx = int(torch.argmax(weights).item())
    T_mean = Ts[ref_idx].clone()
    for _ in range(iters):
        T_mean_inv = torch.linalg.inv(T_mean)
        delta = T_mean_inv.unsqueeze(0) @ Ts        # (N, 4, 4), in mean's frame
        xi = se3_log(delta)                          # (N, 6)
        xi_mean = (w.unsqueeze(-1) * xi).sum(dim=0)  # (6,)
        if xi_mean.abs().max() < 1e-8:
            break
        T_mean = T_mean @ se3_exp(xi_mean)
    return T_mean


def se3_robust_mean(Ts: Tensor, weights: Tensor, iters: int = 8,
                    huber_c: float = 0.3) -> Tensor:
    """Iteratively reweighted (IRLS) Fréchet mean with Huber kernel.

    Candidates whose distance in se(3) tangent space from the current mean
    exceeds `huber_c` get down-weighted by `huber_c / dist`, so a few corrupt
    rig-pose candidates cannot dominate the average.

    Returns a pair (T_mean, effective_weights) so callers can inspect which
    candidates were trusted.
    """
    assert Ts.dim() == 3 and Ts.shape[-2:] == (4, 4)
    base_w = weights.to(Ts)
    base_w = base_w / base_w.sum().clamp(min=_EPS)
    ref_idx = int(torch.argmax(weights).item())
    T_mean = Ts[ref_idx].clone()
    eff_w = base_w.clone()
    for _ in range(iters):
        T_mean_inv = torch.linalg.inv(T_mean)
        delta = T_mean_inv.unsqueeze(0) @ Ts
        xi = se3_log(delta)                     # (N, 6)
        # Distance metric: rotation in radians + translation (both in
        # tangent space units; fine for Huber bookkeeping).
        dist = xi.norm(dim=-1)                   # (N,)
        huber = torch.where(
            dist <= huber_c,
            torch.ones_like(dist),
            huber_c / dist.clamp(min=_EPS),
        )
        eff_w = base_w * huber
        eff_w = eff_w / eff_w.sum().clamp(min=_EPS)
        xi_mean = (eff_w.unsqueeze(-1) * xi).sum(dim=0)
        if xi_mean.abs().max() < 1e-8:
            break
        T_mean = T_mean @ se3_exp(xi_mean)
    return T_mean, eff_w
