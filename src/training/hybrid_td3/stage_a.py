"""Stage A: 10k decoded actions must lie in A_f(s). No FMU."""

from __future__ import annotations

import torch

from actions.caes_u import is_legal_u_caes, mode_from_u
from actions.joint_support import GridCoupling, predict_p_grid_u
from actions.types import CaesMode
from training.hybrid_td3.actor import HybridActor


def run_stage_a_support(*, n: int = 10_000, seed: int = 0, obs_dim: int = 8) -> dict:
    """Sample n actor actions on synthetic dynamic intervals; assert support.

    Returns:
        Dict with illegal_mode, bound_violation, nan counts and status.
    """
    torch.manual_seed(int(seed))
    actor = HybridActor(obs_dim, parameterized_caes=True, use_dynamic_support=True)
    actor.eval()
    illegal = 0
    bound_viol = 0
    grid_viol = 0
    nan = 0
    batch = 256
    remaining = int(n)
    dis_lo_v, dis_hi_v = -0.9, -0.4
    chg_lo_v, chg_hi_v = 0.88, 0.99
    ctx = GridCoupling(
        p_thermal=1.5e8,
        p_battery=1.0e8,
        p_caes=1.5e8,
        residual=0.0,
        g_min=-5.0e8,
        g_max=5.0e8,
    )
    while remaining > 0:
        b = min(batch, remaining)
        obs = torch.randn(b, obs_dim)
        mask = torch.ones(b, 3, dtype=torch.bool)
        dis_lo = torch.full((b,), dis_lo_v)
        dis_hi = torch.full((b,), dis_hi_v)
        chg_lo = torch.full((b,), chg_lo_v)
        chg_hi = torch.full((b,), chg_hi_v)
        tp_lo = torch.full((b,), 0.3)
        tp_hi = torch.ones(b)
        bat_lo = -torch.ones(b)
        bat_hi = torch.ones(b)
        with torch.no_grad():
            out = actor.act(
                obs,
                tp_lo,
                tp_hi,
                bat_lo,
                bat_hi,
                mask,
                deterministic=False,
                dis_lo=dis_lo,
                dis_hi=dis_hi,
                chg_lo=chg_lo,
                chg_hi=chg_hi,
                grid_residual=torch.zeros(b),
                grid_g_min=torch.full((b,), ctx.g_min),
                grid_g_max=torch.full((b,), ctx.g_max),
                p_cap_thermal=torch.full((b,), ctx.p_thermal),
                p_cap_battery=torch.full((b,), ctx.p_battery),
                p_cap_caes=torch.full((b,), ctx.p_caes),
            )
        u_tp = out["u_tp"]
        u_bat = out["u_battery"]
        u = out["u_caes"]
        nan += int((~torch.isfinite(u)).sum().item())
        for j, val in enumerate(u.tolist()):
            if not is_legal_u_caes(float(val)):
                illegal += 1
            m = mode_from_u(float(val))
            if m == CaesMode.DISCHARGE and not (dis_lo_v - 1e-4 <= float(val) <= dis_hi_v + 1e-4):
                bound_viol += 1
            if m == CaesMode.CHARGE and not (chg_lo_v - 1e-4 <= float(val) <= chg_hi_v + 1e-4):
                bound_viol += 1
            pg = predict_p_grid_u(ctx, float(u_tp[j]), float(u_bat[j]), float(val))
            if pg < ctx.g_min - 1e-3 or pg > ctx.g_max + 1e-3:
                grid_viol += 1
        remaining -= b
    ok = illegal == 0 and bound_viol == 0 and grid_viol == 0 and nan == 0
    return {
        "status": "completed" if ok else "failed",
        "method": "stage_a_support",
        "n": int(n),
        "illegal_caes_mode": illegal,
        "dynamic_bound_violation": bound_viol,
        "grid_violation": grid_viol,
        "nan": nan,
        "seed": int(seed),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_stage_a_support(), indent=2))
