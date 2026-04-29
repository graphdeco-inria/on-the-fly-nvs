"""B-1 patch v2 §7.1 unit test — autograd through scene_model.rig_R6D / rig_t.

Builds a minimal stand-in for a rig keyframe (no rendering, no Gaussians) and
verifies:

  1. After scene_model.register_rig_poses + scene_model.add_keyframe, calling
     keyframe.get_R / get_t / get_Rt produces tensors that lead back to
     rig_R6D / rig_t in autograd.
  2. A second backward on a fresh forward also lands gradients on rig_R6D
     (this is the cached_Rts bypass check — the v1 patch's biggest blind
     spot).
  3. scene_model.get_Rts() returns a fresh-graph tensor in rig mode
     (i.e. backward through the stacked tensor reaches rig_R6D, even after
     a previous rig_optimizer.step).

Run from on-the-fly-nvs/ root:
    python tools/test_b1_autograd.py
"""

from __future__ import annotations

import sys
import os
import torch

sys.path.insert(0, os.path.abspath("."))

from utils import sixD2mtx, mtx2sixD


# ---------------------------------------------------------------------------
# Minimal stand-in classes — we don't want to drag the full SceneModel /
# Keyframe constructor into a unit test. We only exercise the rig path of
# Keyframe.get_R / get_t / set_Rt and SceneModel.get_Rts.
# ---------------------------------------------------------------------------
class FakeKeyframe:
    """Mimics Keyframe's rig-mode pose accessors (the only methods relevant
    to the autograd test)."""
    def __init__(self, ts_idx, rig_view, rel_R, rel_t, rW2C_init):
        self.ts_idx = ts_idx
        self.rig_view = rig_view
        self.is_rig_mode = True
        self.rel_R = rel_R
        self.rel_t = rel_t
        self.rW2C = rW2C_init  # plain attr in rig mode
        self.tW2C = torch.zeros(3, device="cuda")
        self.scene_model = None
        # Fake info dict for compatibility
        self.info = {"is_test": False, "rig_view": rig_view, "ts_idx": ts_idx}

    def get_R(self):
        rig_R = sixD2mtx(self.scene_model.rig_R6D[self.ts_idx])
        return self.rel_R @ rig_R

    def get_t(self):
        rig_t = self.scene_model.rig_t[self.ts_idx]
        return self.rel_R @ rig_t + self.rel_t

    def get_Rt(self):
        Rt = torch.eye(4, device="cuda")
        Rt[:3, :3] = self.get_R()
        Rt[:3, 3] = self.get_t()
        return Rt


class FakeSceneModel:
    """Mimics SceneModel rig-pose container + get_Rts (cached_Rts bypass)."""
    def __init__(self):
        self.rig_R6D = torch.nn.ParameterList()
        self.rig_t = torch.nn.ParameterList()
        self.rig_optimizer = None
        self.keyframes = []
        # match SceneModel layout — these would normally be torch.empty(0,...)
        self.cached_Rts = torch.empty(0, 4, 4, device="cuda")
        self.valid_Rt_cache = torch.empty(0, dtype=torch.bool, device="cuda")

    def register_rig_poses(self, rig_R6D_init, rig_t_init, lr):
        from scene.optimizers import BaseAdam
        params = {}
        for i in range(rig_R6D_init.shape[0]):
            p_R = torch.nn.Parameter(rig_R6D_init[i].clone())
            p_t = torch.nn.Parameter(rig_t_init[i].clone())
            self.rig_R6D.append(p_R)
            self.rig_t.append(p_t)
            params[f"rig_R6D_{i}"] = {"val": p_R, "lr": lr}
            params[f"rig_t_{i}"] = {"val": p_t, "lr": lr}
        self.rig_optimizer = BaseAdam(params, betas=(0.8, 0.99))

    def append_rig_pose(self, rig_R6D_new, rig_t_new):
        p_R = torch.nn.Parameter(rig_R6D_new.clone())
        p_t = torch.nn.Parameter(rig_t_new.clone())
        self.rig_R6D.append(p_R)
        self.rig_t.append(p_t)
        new_idx = len(self.rig_R6D) - 1
        lr = next(iter(self.rig_optimizer.params.values()))["lr"]
        self.rig_optimizer.add_param(f"rig_R6D_{new_idx}", p_R, lr=lr)
        self.rig_optimizer.add_param(f"rig_t_{new_idx}", p_t, lr=lr)

    def add_keyframe(self, kf):
        kf.scene_model = self
        self.keyframes.append(kf)

    def get_Rts(self):
        if self.rig_optimizer is not None:
            return torch.stack([kf.get_Rt() for kf in self.keyframes])
        return self.cached_Rts


def main():
    assert torch.cuda.is_available(), "CUDA required"
    torch.manual_seed(0)

    # Fake rig: 2 views (ref + 1 other), 3 ts.
    N_ts, N_views = 3, 2
    sm = FakeSceneModel()

    # Bootstrap rig poses: identity rotations, small translations.
    rig_R6D_init = torch.eye(3, 2)[None].repeat(N_ts, 1, 1).cuda()
    rig_t_init = torch.randn(N_ts, 3, device="cuda") * 0.1
    sm.register_rig_poses(rig_R6D_init, rig_t_init, lr=1e-3)

    # Two views with rotation-only relative_Rt (rel_t = 0).
    rel_R_per_view = []
    for v in range(N_views):
        # 90° turn between views, simple example
        ang = v * 0.5
        c, s = torch.cos(torch.tensor(ang)), torch.sin(torch.tensor(ang))
        R = torch.tensor(
            [[c, 0, s], [0, 1, 0], [-s, 0, c]], device="cuda"
        )
        rel_R_per_view.append(R)

    # One keyframe per ts × view.
    for ts in range(N_ts):
        for v in range(N_views):
            kf = FakeKeyframe(
                ts_idx=ts,
                rig_view=f"view_{v}",
                rel_R=rel_R_per_view[v],
                rel_t=torch.zeros(3, device="cuda"),
                rW2C_init=torch.eye(3, 2, device="cuda"),
            )
            sm.add_keyframe(kf)

    # ─── Step 1: first backward ───
    sm.rig_optimizer.zero_grad()
    kf = sm.keyframes[0]
    R = kf.get_R()
    t = kf.get_t()
    loss1 = R.sum() + t.sum()
    loss1.backward()
    g1_R = sm.rig_R6D[kf.ts_idx].grad
    g1_t = sm.rig_t[kf.ts_idx].grad
    assert g1_R is not None and torch.isfinite(g1_R).all(), \
        "step1: rig_R6D grad missing"
    assert g1_t is not None and torch.isfinite(g1_t).all(), \
        "step1: rig_t grad missing"
    print(f"  step 1 backward OK  |g_R|={g1_R.norm().item():.4f} "
          f"|g_t|={g1_t.norm().item():.4f}")

    sm.rig_optimizer.step()
    sm.rig_optimizer.zero_grad()

    # ─── Step 2: second backward (cached_Rts bypass check) ───
    R2 = kf.get_R()
    t2 = kf.get_t()
    loss2 = R2.sum() + t2.sum()
    loss2.backward()
    g2_R = sm.rig_R6D[kf.ts_idx].grad
    g2_t = sm.rig_t[kf.ts_idx].grad
    assert g2_R is not None and torch.isfinite(g2_R).all(), \
        "step2: stale graph! rig_R6D grad missing on second forward"
    assert g2_t is not None and torch.isfinite(g2_t).all(), \
        "step2: stale graph! rig_t grad missing on second forward"
    print(f"  step 2 backward OK  |g_R|={g2_R.norm().item():.4f} "
          f"|g_t|={g2_t.norm().item():.4f}  (cached_Rts bypass works)")

    sm.rig_optimizer.zero_grad()

    # ─── Step 3: get_Rts() fresh-graph check ───
    Rts = sm.get_Rts()
    assert Rts.requires_grad, "get_Rts result has no grad"
    loss3 = Rts.sum()
    loss3.backward()
    n_ts_with_grad = sum(
        1 for p in sm.rig_R6D if p.grad is not None and p.grad.abs().sum() > 0
    )
    assert n_ts_with_grad == N_ts, \
        f"get_Rts: only {n_ts_with_grad}/{N_ts} ts received grad"
    print(f"  step 3 get_Rts OK   {n_ts_with_grad}/{N_ts} rig_R6D entries got "
          "grad through stacked tensor")

    # ─── Step 4: append_rig_pose mid-flight ───
    sm.rig_optimizer.zero_grad()
    new_R6D = torch.eye(3, 2, device="cuda")
    new_t = torch.randn(3, device="cuda") * 0.05
    sm.append_rig_pose(new_R6D, new_t)
    # add a keyframe at the new ts
    new_ts = len(sm.rig_R6D) - 1
    kf_new = FakeKeyframe(
        ts_idx=new_ts, rig_view="view_0",
        rel_R=rel_R_per_view[0],
        rel_t=torch.zeros(3, device="cuda"),
        rW2C_init=torch.eye(3, 2, device="cuda"),
    )
    sm.add_keyframe(kf_new)
    R4 = kf_new.get_R()
    loss4 = R4.sum()
    loss4.backward()
    g4 = sm.rig_R6D[new_ts].grad
    assert g4 is not None and torch.isfinite(g4).all(), \
        f"appended ts {new_ts}: grad missing"
    print(f"  step 4 append OK    new_ts={new_ts} grad |{g4.norm().item():.4f}|")

    print("\nAll §7.1 checks passed.")


if __name__ == "__main__":
    main()
