"""Cui 2024 OCTD3 Table 4 knobs that transfer to same-hour hybrid SAC.

Cui et al., Applied Energy 374 (2024) 123950, Table 4. OCTD3 is option-HRL;
FS-HSAC does **not** copy options / initiation / κ. Shared DRL numbers:

  ε_max=1.0, ε_min=0.05, Δε=6e-6 on a 2e5-step run
  lr=1e-4, τ=0.005, batch=64, γ=0.99, replay=10_000

Our fair week is 5000×168=840_000 steps. ε and replay are scaled so they
hit Cui's schedule at the same *fraction* of the horizon (Δε hits ε_min at
158_333/200_000 ≈ 0.792). TD3 delay D=2, σ²=0.1, option temperature κ=1,
and PFI/CCI weight w=0.2 are not SAC knobs.
"""

from __future__ import annotations

CUI_EPS_MAX = 1.0
CUI_EPS_MIN = 0.05
CUI_DELTA_EPS = 6.0e-6
CUI_TRAIN_STEPS_REF = 200_000
CUI_LR = 1.0e-4
CUI_TAU = 0.005
CUI_BATCH = 64
CUI_GAMMA = 0.99
CUI_REPLAY = 10_000


def cui_eps_horizon(total: int) -> int:
    """Steps until ε_min, matching Cui's 158333/200000 fraction of ``total``."""
    hit = (CUI_EPS_MAX - CUI_EPS_MIN) / CUI_DELTA_EPS
    frac = hit / float(CUI_TRAIN_STEPS_REF)
    return max(int(max(int(total), 1) * frac), 1)


def explore_epsilon(
    step: int,
    total: int,
    *,
    eps_max: float = CUI_EPS_MAX,
    eps_min: float = CUI_EPS_MIN,
) -> float:
    """Cui Table 4 ε schedule, scaled to ``total`` training steps."""
    horizon = cui_eps_horizon(total)
    t = max(int(step), 0)
    if t >= horizon:
        return float(eps_min)
    return float(eps_max + (eps_min - eps_max) * (t / horizon))


def scaled_replay(total_steps: int, *, base: int = CUI_REPLAY) -> int:
    """Replay size grown with horizon vs Cui's 10k @ 200k steps."""
    total = max(int(total_steps), 1)
    return max(int(base), int(base * total / CUI_TRAIN_STEPS_REF))
