"""滚动 MILP：在 linprog 同物理代理上精确处理压空 min-load 断带。

论文基线小节须显式交代（对标 GHTD3 对 Gurobi QP 的写法）：

  1. 未来 horizon 内风光荷与电价取 forecast_provider（主矩阵 mode=perfect，
     与 PSO / 混合 SAC 观测对齐；不是随机规划）。
  2. 电池与气罐用能量线性 SoC；省略热罐/冷罐、压力–温度 DAE 与变工况效率。
  3. 压空用二元开停 + 大 M 落入 [-1,-0.33]∪{0}∪[0.86,1]；无最短运行锁
     （Cui 2024 只用启停费；min_run_steps=1）。
  4. 目标与 linprog 相同：购售电 + 燃料 + 碳 + 弃/缺电 + 电池放电磨损 + 末端库存软罚。
     CAES 能量无度电价（eval J 扣除 FMU 储能设备现金流）。
  5. 只执行第一步，真仿真闭环；求解失败则回退到当前火电、零储能。

这不是孪生上的全局最优，也不是综合收益的严格上界。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from actions import CaesMode
from actions.caes_min_run import MIN_CAES_RUN_STEPS
from actions.caes_u import u_from_mode_mag
from optimization.rolling_linprog import (
    LinprogMPCConfig,
    RollingLinprogController,
    demand_mw,
    gen_mw,
)


@dataclass
class MilpMPCConfig(LinprogMPCConfig):
    """与 LinprogMPCConfig 同字段；额外最短运行小时与求解时限。"""

    min_run_steps: int = MIN_CAES_RUN_STEPS
    time_limit_s: float = 20.0


class RollingMilpController(RollingLinprogController):
    """滚动 MILP：相对 linprog 唯一结构差别是压空整数开停与 min-load 带。"""

    def __init__(self, env: Any, cfg: MilpMPCConfig | None = None):
        super().__init__(env, cfg or MilpMPCConfig())
        self.cfg = cfg or MilpMPCConfig()
        self._apply_reward_defaults()
        self._name = "rolling_milp"
        self._prev_caes_mode: CaesMode = CaesMode.IDLE
        self._prev_caes_run: int = 0

    def on_episode_reset(self, info: dict) -> None:
        self._prev_caes_mode = CaesMode.IDLE
        self._prev_caes_run = 0

    def _solve(self) -> dict[str, float]:
        try:
            from scipy.optimize import Bounds, LinearConstraint, milp
        except ImportError as exc:  # pragma: no cover
            return {"ok": 0.0, "p_tp": 0.0, "p_bat": 0.0, "p_gas": 0.0, "note": str(exc)}

        env = self.env
        cfg = self.cfg
        h = int(cfg.horizon)
        bat = self.params["battery"]
        th = self.params["thermal"]
        caes = self.params["caes"]
        grid = self.params["grid"]

        e_bat = float(bat["E_cap_J"]) / 3.6e9
        eta = float(bat["eta"])
        p_bat_max = float(bat["P_cap_W"]) * 1e-6
        p_tp_min = float(th["P_min_W"]) * 1e-6
        p_tp_max = float(th["P_max_W"]) * 1e-6
        p_caes_max = float(caes["P_cap_W"]) * 1e-6
        u_dis_min = abs(float(caes.get("discharge_u_max", -0.33)))  # 0.33
        u_chg_min = float(caes.get("charge_u_min", 0.86))
        soc_g_lo = float(caes["gas_SOC_min"])
        soc_g_hi = float(caes["gas_SOC_max"])
        e_gas = p_caes_max * float(cfg.caes_hours_at_rated) * max(soc_g_hi - soc_g_lo, 0.05)
        p_buy_max = float(grid["P_max_buy_W"]) * 1e-6
        p_sell_max = abs(float(grid["P_max_sell_W"])) * 1e-6
        min_run = max(1, int(cfg.min_run_steps))

        outs = env.last_outputs or {}
        soc_b0 = float(outs.get("battery_soc", 0.5))
        soc_g0 = float(outs.get("caes_gas_soc", 0.85))
        p_tp_prev = gen_mw(outs.get("p_thermal", -p_tp_min * 1e6))
        if p_tp_prev < p_tp_min * 0.5:
            p_tp_prev = demand_mw(abs(float(outs.get("p_thermal", p_tp_min * 1e6))))
        p_tp_prev = float(np.clip(p_tp_prev, p_tp_min, p_tp_max))

        load, wind, pv = self._boundary_horizon()
        buy_p, sell_p = self._prices_horizon(h)
        buy_mwh = buy_p * 1000.0
        sell_mwh = sell_p * 1000.0
        c_th = cfg.fuel_yuan_per_mwh + cfg.carbon_yuan_per_t * cfg.eta_thermal_t_per_mwh
        c_buy_extra = cfg.carbon_yuan_per_t * cfg.eta_grid_t_per_mwh

        # continuous controls per step (same as linprog): 10
        # + binary z_chg, z_dis per step: 2
        n_u = 10
        n_z = 2
        n_step = n_u + n_z
        n_soc = (h + 1) * 2
        n_term = 2
        n = h * n_step + n_soc + n_term

        c = np.zeros(n)
        for k in range(h):
            base = k * n_step
            c[base + 0] = c_th
            c[base + 2] = cfg.deg_yuan_per_mwh
            # caes_ch / caes_dis (base+3/+4) stay 0: no FMU storage kWh price in eval J
            c[base + 5] = buy_mwh[k] + c_buy_extra
            c[base + 6] = -sell_mwh[k]
            c[base + 7] = cfg.nu_curt_yuan_per_mwh
            c[base + 8] = cfg.nu_curt_yuan_per_mwh
            c[base + 9] = cfg.nu_uns_yuan_per_mwh
        c[-2] = cfg.terminal_soc_yuan
        c[-1] = cfg.terminal_soc_yuan

        lb = np.zeros(n)
        ub = np.full(n, np.inf)
        integrality = np.zeros(n, dtype=np.int8)

        for k in range(h):
            base = k * n_step
            # p_tp, bat_ch, bat_dis, caes_ch, caes_dis, buy, sell, curt_w, curt_pv, uns
            lb[base + 0], ub[base + 0] = p_tp_min, p_tp_max
            lb[base + 1], ub[base + 1] = 0.0, p_bat_max
            lb[base + 2], ub[base + 2] = 0.0, p_bat_max
            lb[base + 3], ub[base + 3] = 0.0, p_caes_max
            lb[base + 4], ub[base + 4] = 0.0, p_caes_max
            lb[base + 5], ub[base + 5] = 0.0, p_buy_max
            lb[base + 6], ub[base + 6] = 0.0, p_sell_max
            lb[base + 7], ub[base + 7] = 0.0, float(wind[k])
            lb[base + 8], ub[base + 8] = 0.0, float(pv[k])
            lb[base + 9], ub[base + 9] = 0.0, float(load[k])
            # binaries
            iz_c, iz_d = base + n_u, base + n_u + 1
            lb[iz_c], ub[iz_c] = 0.0, 1.0
            lb[iz_d], ub[iz_d] = 0.0, 1.0
            integrality[iz_c] = 1
            integrality[iz_d] = 1

        sb_lo, sb_hi = float(bat["SOC_min"]), float(bat["SOC_max"])
        i_sb0 = h * n_step
        i_sg0 = h * n_step + (h + 1)
        for i in range(h + 1):
            lb[i_sb0 + i], ub[i_sb0 + i] = sb_lo, sb_hi
            lb[i_sg0 + i], ub[i_sg0 + i] = soc_g_lo, soc_g_hi
        i_eb, i_eg = n - 2, n - 1
        lb[i_eb], ub[i_eb] = 0.0, np.inf
        lb[i_eg], ub[i_eg] = 0.0, np.inf

        a_rows: list[np.ndarray] = []
        b_lo: list[float] = []
        b_hi: list[float] = []

        def _eq(row: np.ndarray, val: float) -> None:
            a_rows.append(row)
            b_lo.append(val)
            b_hi.append(val)

        def _le(row: np.ndarray, val: float) -> None:
            a_rows.append(row)
            b_lo.append(-np.inf)
            b_hi.append(val)

        # initial SoC
        row = np.zeros(n)
        row[i_sb0] = 1.0
        _eq(row, soc_b0)
        row = np.zeros(n)
        row[i_sg0] = 1.0
        _eq(row, soc_g0)

        dt = 1.0
        p_chg_lo = u_chg_min * p_caes_max
        p_dis_lo = u_dis_min * p_caes_max

        for k in range(h):
            base = k * n_step
            iz_c, iz_d = base + n_u, base + n_u + 1

            # power balance
            row = np.zeros(n)
            row[base + 0] = 1.0
            row[base + 1] = -1.0
            row[base + 2] = 1.0
            row[base + 3] = -1.0
            row[base + 4] = 1.0
            row[base + 5] = 1.0
            row[base + 6] = -1.0
            row[base + 7] = -1.0
            row[base + 8] = -1.0
            row[base + 9] = 1.0
            _eq(row, float(load[k] - wind[k] - pv[k]))

            # battery SoC
            row = np.zeros(n)
            row[i_sb0 + k + 1] = 1.0
            row[i_sb0 + k] = -1.0
            row[base + 1] = -eta * dt / max(e_bat, 1e-6)
            row[base + 2] = dt / (max(eta, 1e-6) * max(e_bat, 1e-6))
            _eq(row, 0.0)

            # gas SoC (energy proxy; thermal coupling omitted)
            row = np.zeros(n)
            row[i_sg0 + k + 1] = 1.0
            row[i_sg0 + k] = -1.0
            row[base + 3] = -dt / max(e_gas, 1e-6)
            row[base + 4] = dt / max(e_gas, 1e-6)
            _eq(row, 0.0)

            # mutual exclusion
            row = np.zeros(n)
            row[iz_c] = 1.0
            row[iz_d] = 1.0
            _le(row, 1.0)

            # charge band: p_chg <= Pmax z_c ; p_chg >= u_min Pmax z_c
            row = np.zeros(n)
            row[base + 3] = 1.0
            row[iz_c] = -p_caes_max
            _le(row, 0.0)
            row = np.zeros(n)
            row[base + 3] = -1.0
            row[iz_c] = p_chg_lo
            _le(row, 0.0)

            # discharge band
            row = np.zeros(n)
            row[base + 4] = 1.0
            row[iz_d] = -p_caes_max
            _le(row, 0.0)
            row = np.zeros(n)
            row[base + 4] = -1.0
            row[iz_d] = p_dis_lo
            _le(row, 0.0)

        # thermal ramp
        du_max = float(th["rate_max_per_s"]) * 3600.0
        dp_max = du_max * p_tp_max
        row = np.zeros(n)
        row[0] = 1.0
        _le(row, p_tp_prev + dp_max)
        row = np.zeros(n)
        row[0] = -1.0
        _le(row, -p_tp_prev + dp_max)
        for k in range(1, h):
            b0 = (k - 1) * n_step
            b1 = k * n_step
            row = np.zeros(n)
            row[b1] = 1.0
            row[b0] = -1.0
            _le(row, dp_max)
            row = np.zeros(n)
            row[b1] = -1.0
            row[b0] = 1.0
            _le(row, dp_max)

        # min-run / mode lock within horizon (UC-style).
        # Rising edge at k forces the next L=min(min_run, h-k) hours to stay on:
        #   z[k] - z[k-1] <= z[k+j]  <=>  z[k] - z[k+j] <= z[k-1]
        # with z[-1] taken from the previous closed-loop mode.
        z_c_prev = 1.0 if self._prev_caes_mode == CaesMode.CHARGE else 0.0
        z_d_prev = 1.0 if self._prev_caes_mode == CaesMode.DISCHARGE else 0.0
        rem = max(0, min_run - int(self._prev_caes_run))
        if rem > 0 and self._prev_caes_mode == CaesMode.CHARGE:
            for k in range(min(rem, h)):
                row = np.zeros(n)
                row[k * n_step + n_u] = -1.0
                _le(row, -1.0)  # z_c[k] >= 1
        if rem > 0 and self._prev_caes_mode == CaesMode.DISCHARGE:
            for k in range(min(rem, h)):
                row = np.zeros(n)
                row[k * n_step + n_u + 1] = -1.0
                _le(row, -1.0)

        for k in range(h):
            L = min(min_run, h - k)
            for j in range(L):
                row = np.zeros(n)
                row[k * n_step + n_u] = 1.0
                row[(k + j) * n_step + n_u] -= 1.0
                if k == 0:
                    _le(row, z_c_prev)
                else:
                    row[(k - 1) * n_step + n_u] -= 1.0
                    _le(row, 0.0)
                row = np.zeros(n)
                row[k * n_step + n_u + 1] = 1.0
                row[(k + j) * n_step + n_u + 1] -= 1.0
                if k == 0:
                    _le(row, z_d_prev)
                else:
                    row[(k - 1) * n_step + n_u + 1] -= 1.0
                    _le(row, 0.0)

        # terminal |soc - ref| soft
        s_ref_b = float((env.initial_soc or {}).get("battery_soc", soc_b0))
        s_ref_g = float((env.initial_soc or {}).get("caes_gas_soc", soc_g0))
        row = np.zeros(n)
        row[i_sb0 + h] = 1.0
        row[i_eb] = -1.0
        _le(row, s_ref_b)
        row = np.zeros(n)
        row[i_sb0 + h] = -1.0
        row[i_eb] = -1.0
        _le(row, -s_ref_b)
        row = np.zeros(n)
        row[i_sg0 + h] = 1.0
        row[i_eg] = -1.0
        _le(row, s_ref_g)
        row = np.zeros(n)
        row[i_sg0 + h] = -1.0
        row[i_eg] = -1.0
        _le(row, -s_ref_g)

        A = np.asarray(a_rows, dtype=np.float64)
        constraints = LinearConstraint(A, np.asarray(b_lo), np.asarray(b_hi))
        bounds = Bounds(lb, ub)
        res = milp(
            c,
            integrality=integrality,
            bounds=bounds,
            constraints=constraints,
            options={"time_limit": float(cfg.time_limit_s), "disp": False},
        )
        if not bool(getattr(res, "success", False)) or res.x is None:
            msg = str(getattr(res, "message", "milp_failed") or "").lower()
            timed_out = ("time" in msg) or ("limit" in msg)
            return {
                "ok": 0.0,
                "p_tp": p_tp_prev,
                "p_bat": 0.0,
                "p_gas": 0.0,
                "note": str(getattr(res, "message", "milp_failed")),
                "timed_out": 1.0 if timed_out else 0.0,
            }
        x = res.x
        p_gas = float(x[3] - x[4])
        z_c0 = float(x[n_u])
        z_d0 = float(x[n_u + 1])
        return {
            "ok": 1.0,
            "p_tp": float(x[0]),
            "p_bat": float(x[1] - x[2]),
            "p_gas": p_gas,
            "z_chg": z_c0,
            "z_dis": z_d0,
            "obj": float(res.fun) if res.fun is not None else 0.0,
            "timed_out": 0.0,
        }

    def predict(self, obs, deterministic: bool = True) -> dict:
        out = super().predict(obs, deterministic=deterministic)
        # update local min-run memory from executed decode
        u = float(out["u_caes"][0])
        if u >= 0.86:
            mode = CaesMode.CHARGE
        elif u <= -0.33:
            mode = CaesMode.DISCHARGE
        else:
            mode = CaesMode.IDLE
        if mode == self._prev_caes_mode and mode != CaesMode.IDLE:
            self._prev_caes_run += 1
        elif mode == CaesMode.IDLE:
            self._prev_caes_mode = CaesMode.IDLE
            self._prev_caes_run = 0
        else:
            self._prev_caes_mode = mode
            self._prev_caes_run = 1
        return out
