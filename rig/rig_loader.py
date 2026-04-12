"""Load blender_rig.json and convert to COLMAP-style relative world-to-camera transforms.

Mirrors the conversion used by eqr_to_pinhole.py so poses stay consistent with
the extracted pinhole images. Translation is forced to zero — the rig is
rotation-only (all views share one optical center).
"""

import json
from dataclasses import dataclass, field
from typing import Dict, List

import torch
from torch import Tensor


_AXIS_FLIP = torch.diag(torch.tensor([1.0, -1.0, -1.0]))


def _quat_to_R(q: Tensor) -> Tensor:
    # Blender quaternion order: (w, x, y, z)
    w, x, y, z = q[0], q[1], q[2], q[3]
    n = torch.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    R = torch.stack([
        torch.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w),     2 * (x * z + y * w)]),
        torch.stack([2 * (x * y + z * w),     1 - 2 * (x * x + z * z), 2 * (y * z - x * w)]),
        torch.stack([2 * (x * z - y * w),     2 * (y * z + x * w),     1 - 2 * (x * x + y * y)]),
    ])
    return R


def _blender_quat_to_world2cam(q: Tensor) -> Tensor:
    R_c2w = _quat_to_R(q)
    return _AXIS_FLIP.to(q) @ R_c2w.T


@dataclass
class RigConfig:
    """Fixed rig geometry, expressed relative to a reference view.

    `relative_Rt[view]` is a 4x4 matrix such that
        view_world2cam = relative_Rt[view] @ rig_world2cam
    where `rig_world2cam` is the world-to-camera transform of the reference view
    (the rig's shared optical center pose).
    """
    view_names: List[str]
    ref_view: str
    relative_Rt: Dict[str, Tensor] = field(default_factory=dict)

    def ref_index(self) -> int:
        return self.view_names.index(self.ref_view)


def load_rig_config(config_path: str, ref_view: str = "High_Cam07",
                    device: str = "cpu") -> RigConfig:
    with open(config_path, "r", encoding="utf-8") as f:
        rings = json.load(f)

    quats: Dict[str, Tensor] = {}
    order: List[str] = []
    for ring in rings:
        for cam in ring["cameras"]:
            order.append(cam["name"])
            quats[cam["name"]] = torch.tensor(cam["rotation"], dtype=torch.float32,
                                              device=device)

    if ref_view not in quats:
        raise ValueError(f"ref_view {ref_view!r} not found in rig config "
                         f"(available: {order})")

    ref_R_w2c = _blender_quat_to_world2cam(quats[ref_view])

    relative: Dict[str, Tensor] = {}
    for name in order:
        R_w2c = _blender_quat_to_world2cam(quats[name])
        rel_R = R_w2c @ ref_R_w2c.T
        Rt = torch.eye(4, dtype=torch.float32, device=device)
        Rt[:3, :3] = rel_R
        # Translation intentionally zero — rotation-only rig.
        relative[name] = Rt

    return RigConfig(view_names=order, ref_view=ref_view, relative_Rt=relative)
