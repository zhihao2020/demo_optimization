"""CAES project: random legal samples stay in nonconvex legal set."""
from __future__ import annotations

import numpy as np

from actions import CaesMode
from actions.caes_u import is_legal_u_caes, u_from_mode_mag


def test_random_mode_mag_to_u_legal():
    rng = np.random.default_rng(0)
    for _ in range(100_000):
        mode = CaesMode(int(rng.integers(0, 3)))
        mag = float(rng.random()) if mode != CaesMode.IDLE else 0.0
        u = u_from_mode_mag(mode, mag)
        assert is_legal_u_caes(u)
