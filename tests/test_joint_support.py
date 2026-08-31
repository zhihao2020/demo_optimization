"""Joint grid-coupled support is not a cartesian product of device boxes."""
from __future__ import annotations

from actions.joint_support import (
    GridCoupling,
    decode_joint_numpy,
    predict_p_grid_u,
    thermal_window,
    tighten_caes_modes,
)


def _ctx(**kw) -> GridCoupling:
    base = dict(p_thermal=1.0, p_battery=1.0, p_caes=1.0, residual=0.0, g_min=-0.25, g_max=0.25)
    base.update(kw)
    return GridCoupling(**base)


def test_cartesian_can_violate_grid_but_decoder_does_not():
    ctx = _ctx()
    # Device boxes are [-1,1]; u_T=1, u_B=-1, u_C=0 → p_grid=2, way outside ±0.25.
    assert predict_p_grid_u(ctx, 1.0, -1.0, 0.0) > 0.25
    u_t, u_b = decode_joint_numpy(ctx, -1.0, 1.0, -1.0, 1.0, 0.0, 1.0, -1.0)
    pg = predict_p_grid_u(ctx, u_t, u_b, 0.0)
    assert -0.25 - 1e-9 <= pg <= 0.25 + 1e-9


def test_empty_caes_mode_is_masked_when_grid_window_misses():
    ctx = _ctx(g_min=10.0, g_max=11.0, residual=0.0)  # impossible band
    dis, chg, ad, ai, ac = tighten_caes_modes(ctx, 0.3, 1.0, -1.0, 1.0, (-1.0, -0.33), (0.86, 1.0), True)
    assert dis is None and chg is None
    assert ad is False and ac is False
    assert ai is False


def test_thermal_window_nonempty_for_idle_on_wide_grid():
    ctx = _ctx(g_min=-5.0, g_max=5.0)
    win = thermal_window(ctx, 0.0, 0.33, 1.0, -1.0, 1.0)
    assert win is not None
    assert win[0] <= win[1]
