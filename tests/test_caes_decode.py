"""CAES 解码：大规模随机合法 HybridAction 必须落在非凸合法集。"""

import numpy as np

from actions import CaesMode, HybridAction, HybridActionDecoder

_EPS = 1e-9


def _legal(u: float) -> bool:
    """判断 u_caes 是否落在非凸合法集（放电/空闲/充电三段）。

    Args:
        u: 解码后的 u_caes 值。

    Returns:
        在合法区间内为 True。
    """
    return (-1.0 - _EPS <= u <= -0.33 + _EPS) or (abs(u) <= _EPS) or (0.86 - _EPS <= u <= 1.0 + _EPS)


def test_caes_decode_100000_legal():
    """验证 10 万次随机合法 HybridAction 解码后 u_caes 均在非凸合法集内。"""
    rng = np.random.default_rng(0)
    dec = HybridActionDecoder()
    n = 100_000
    for _ in range(n):
        mode = CaesMode(int(rng.integers(0, 3)))
        mag = float(rng.random())
        action = HybridAction(
            u_tp=float(rng.uniform(1 / 3, 1.0)),
            u_battery=float(rng.uniform(-1.0, 1.0)),
            caes_mode=mode,
            caes_magnitude=mag,
        )
        u = dec.decode(action).u_caes
        assert _legal(u), u
        if mode == CaesMode.IDLE:
            assert abs(u) <= _EPS
