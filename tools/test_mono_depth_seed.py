"""Unit tests for relative_idepth_to_depth — the helper used to seed
bootstrap_rig xyz_init from Depth-Anything-V2 output.

Run: python tools/test_mono_depth_seed.py
"""

import os
import sys
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..")))

from scene.mono_depth import relative_idepth_to_depth


def test_normal_input_positive_and_ordered():
    """Normal idepth (median 0, MAD 1, mixed sign) must yield strictly positive
    depth, and relative ordering must be preserved (smaller idepth → larger
    depth, since idepth is inverse depth)."""
    idepth = torch.tensor([-2.0, -0.5, 0.0, 0.5, 2.0])
    depth = relative_idepth_to_depth(idepth)
    assert (depth > 0).all(), f"non-positive output: {depth}"
    # Inverse-depth ordering: idepth ascending → depth descending.
    assert torch.all(depth[1:] < depth[:-1]), f"ordering broken: {depth}"
    print(f"[ok] normal_input: {idepth.tolist()} -> {depth.tolist()}")


def test_all_zero_idepth():
    """All zeros: idepth_min = 0, shifted = 1, depth = 1. Not the fallback
    branch — but a stable, sane output."""
    idepth = torch.zeros(5)
    depth = relative_idepth_to_depth(idepth)
    assert torch.allclose(depth, torch.ones(5)), f"unexpected: {depth}"
    print(f"[ok] all_zero: {depth.tolist()}")


def test_all_nonfinite_returns_fallback():
    """Every entry non-finite → constant fallback map."""
    idepth = torch.full((5,), float("nan"))
    depth = relative_idepth_to_depth(idepth, fallback_depth=2.5)
    assert torch.allclose(depth, torch.full((5,), 2.5)), f"unexpected: {depth}"
    print(f"[ok] all_nan: -> {depth.tolist()}")


def test_nan_safe_partial():
    """Some NaN, some finite. NaN entries get fallback; finite ones get
    proper conversion."""
    idepth = torch.tensor([-1.0, float("nan"), 0.0, float("inf"), 1.0])
    depth = relative_idepth_to_depth(idepth, fallback_depth=0.5)
    # Finite entries: idepth_min = -1.0 (computed over finite only),
    # shifted = idepth - (-1) + 1 = idepth + 2 → [1, _, 2, _, 3]
    # depth = 1/shifted → [1.0, _, 0.5, _, 1/3]
    expected = torch.tensor([1.0, 0.5, 0.5, 0.5, 1.0 / 3.0])
    assert torch.allclose(depth, expected, atol=1e-5), (
        f"got {depth.tolist()}, expected {expected.tolist()}"
    )
    print(f"[ok] nan_safe: {idepth.tolist()} -> {depth.tolist()}")


def test_2d_map_shape_preserved():
    """Helper must work on (H, W) maps too, not just 1D."""
    idepth = torch.randn(8, 8)
    depth = relative_idepth_to_depth(idepth)
    assert depth.shape == idepth.shape
    assert (depth > 0).all()
    print(f"[ok] 2d_map shape={tuple(depth.shape)} all_positive")


if __name__ == "__main__":
    test_normal_input_positive_and_ordered()
    test_all_zero_idepth()
    test_all_nonfinite_returns_fallback()
    test_nan_safe_partial()
    test_2d_map_shape_preserved()
    print("\nALL TESTS PASSED")
