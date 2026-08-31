"""Stage A: 10k decoded actions must lie in A_f(s). No FMU."""

from __future__ import annotations

import torch

from actions.caes_u import is_legal_u_caes, mode_from_u
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
    nan = 0
    batch = 256
    remaining = int(n)
    dis_lo_v, dis_hi_v = -0.9, -0.4
    chg_lo_v, chg_hi_v = 0.88, 0.99
    while remaining > 0:
        b = min(batch, remaining)
        obs = torch.randn(b, obs_dim)
        mask = torch.ones(b, 3, dtype=torch.bool)
        dis_lo = torch.full((b,), dis_lo_v)
        dis_hi = torch.full((b,), dis_hi_v)
        chg_lo = torch.full((b,), chg_lo_v)
        chg_hi = torch.full((b,), chg_hi_v)
        with torch.no_grad():
            out = actor.act(
                obs,
                torch.full((b,), 0.3),
                torch.ones(b),
                -torch.ones(b),
                torch.ones(b),
                mask,
                deterministic=False,
                dis_lo=dis_lo,
                dis_hi=dis_hi,
                chg_lo=chg_lo,
                chg_hi=chg_hi,
            )
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
        remaining -= b
    ok = illegal == 0 and bound_viol == 0 and nan == 0
    return {
        "status": "completed" if ok else "failed",
        "method": "stage_a_support",
        "n": int(n),
        "illegal_caes_mode": illegal,
        "dynamic_bound_violation": bound_viol,
        "nan": nan,
        "seed": int(seed),
    }


if __name__ == "__main__":
    import json

    print(json.dumps(run_stage_a_support(), indent=2))
