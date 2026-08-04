"""真滚动线性规划（scipy.optimize.linprog）。

代理模型（relaxed surrogate）：
  决策：火电功率、电池充/放、气罐等效充/放、购/售电（MW）
  约束：功率平衡、SOC 动态、容量与联络线、火电爬坡（一步）
  目标：min 购电成本 + 火电燃料代理 - 售电收入（线性）
闭环：只执行第一步，再经 env 解码为 hybrid 动作 + 真实 FMU step。

CAES 模式在代理中为连续充放功率，再映射为 charge/discharge/idle（非凸合法集外推由可行域裁剪）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import yaml
from pathlib import Path
from scipy.optimize import linprog

from actions import CaesMode


@dataclass
class LinprogMPCConfig:
    horizon: int = 8
    # 相对初始 SOC 的终端软罚（仅代理）
    terminal_soc_weight: float = 5.0
    fuel_yuan_per_mwh: float = 280.0  # 火电线性燃料代理


class RollingLinprogController:
    """滚动 linprog 控制器。"""

    def __init__(self, env: Any, cfg: LinprogMPCConfig | None = None):
        self.env = env
        self.cfg = cfg or LinprogMPCConfig()
        root = Path(__file__).resolve().parents[2]  # repo root
        with (root / "src" / "config" / "device_params.yaml").open(encoding="utf-8") as f:
            self.params = yaml.safe_load(f)
        self._name = "rolling_linprog"

    def on_episode_reset(self, info: dict) -> None:
        pass

    def _prices_horizon(self, h: int) -> tuple[np.ndarray, np.ndarray]:
        env = self.env
        H = self.cfg.horizon
        buy = np.zeros(H, dtype=np.float64)
        sell = np.zeros(H, dtype=np.float64)
        t0 = float(env.adapter.time)
        dt = float(env.config["fmu"]["decision_interval_seconds"])
        for k in range(H):
            t = t0 + k * dt
            if getattr(env, "price_profile", None) is not None:
                try:
                    b, s = env.price_profile.prices_at(t)
                    buy[k], sell[k] = float(b), float(s)
                except Exception:
                    buy[k], sell[k] = 0.6, 0.2
            else:
                buy[k], sell[k] = 0.6, 0.2
        return buy, sell

    def _boundary_horizon(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """返回 MW 的 load, wind_avail, pv_avail 前瞻（无表则用当前值重复）。"""
        env = self.env
        H = self.cfg.horizon
        outs = env.last_outputs or {}
        load0 = float(outs.get("p_load_actual", 2e8)) * 1e-6
        wind0 = float(outs.get("p_wind_available", 0.0)) * 1e-6
        pv0 = float(outs.get("p_pv_available", 0.0)) * 1e-6
        load = np.full(H, load0)
        wind = np.full(H, wind0)
        pv = np.full(H, pv0)
        # 若有 forecast_provider：按 scale 反推粗略 MW（与 env_config 一致）
        fp = getattr(env, "forecast_provider", None)
        if fp is not None:
            t0 = float(env.adapter.time)
            dt = float(env.config["fmu"]["decision_interval_seconds"])
            try:
                for k in range(H):
                    feat = fp.at_time(t0 + k * dt)  # normalized features
                    # 顺序 wind, irr, temp, load — scale 见 env_config
                    # wind: *15 不是 MW；无法精确反演 FMU 功率，仅作轻微扰动
                    # 保持当前可用功率更稳
                    _ = feat
            except Exception:
                pass
        return load, wind, pv

    def _solve(self) -> dict[str, float]:
        env = self.env
        H = int(self.cfg.horizon)
        bat = self.params["battery"]
        th = self.params["thermal"]
        caes = self.params["caes"]
        grid = self.params["grid"]

        E_bat_mwh = float(bat["E_cap_J"]) / 3.6e9  # J → MWh
        eta = float(bat["eta"])
        p_bat_max = float(bat["P_cap_W"]) * 1e-6
        p_tp_min = float(th["P_min_W"]) * 1e-6
        p_tp_max = float(th["P_max_W"]) * 1e-6
        p_caes_max = float(caes["P_cap_W"]) * 1e-6
        # 气罐等效能量：用 SOC 区间宽 * 标称（代理）
        E_gas_mwh = 80.0  # 代理容量，仅优化用
        p_buy_max = float(grid["P_max_buy_W"]) * 1e-6
        p_sell_max = abs(float(grid["P_max_sell_W"])) * 1e-6

        outs = env.last_outputs or {}
        soc_b0 = float(outs.get("battery_soc", 0.5))
        soc_g0 = float(outs.get("caes_gas_soc", 0.85))
        p_tp_prev = abs(float(outs.get("p_thermal", p_tp_min * 1e6))) * 1e-6

        load, wind, pv = self._boundary_horizon()
        buy_p, sell_p = self._prices_horizon(H)
        # 电价：CSV 常为 元/kWh 量级 → 元/MWh
        buy_mwh = buy_p * 1000.0
        sell_mwh = sell_p * 1000.0

        # 变量布局 per step h:
        # [p_tp, p_bch, p_bdis, p_gch, p_gdis, p_buy, p_sell] = 7
        n_u = 7
        # soc_bat[0..H], soc_gas[0..H]  — 用等式嵌入差分，显式变量 soc after each step: H each
        # 为简化：变量仅控制量；SOC 由递推线性等式
        # vars: u[h*7:(h+1)*7] for h=0..H-1  +  s_b[0..H] + s_g[0..H]
        n = H * n_u + (H + 1) + (H + 1)
        c = np.zeros(n)
        for h in range(H):
            base = h * n_u
            c[base + 0] = self.cfg.fuel_yuan_per_mwh  # tp
            c[base + 5] = buy_mwh[h]  # buy
            c[base + 6] = -sell_mwh[h]  # sell revenue

        # bounds
        bounds = []
        for h in range(H):
            bounds += [
                (p_tp_min, p_tp_max),
                (0.0, p_bat_max),
                (0.0, p_bat_max),
                (0.0, p_caes_max),
                (0.0, p_caes_max),
                (0.0, p_buy_max),
                (0.0, p_sell_max),
            ]
        for _ in range(H + 1):
            bounds.append((float(bat["SOC_min"]), float(bat["SOC_max"])))
        for _ in range(H + 1):
            bounds.append((float(caes["gas_SOC_min"]), float(caes["gas_SOC_max"])))

        A_eq = []
        b_eq = []
        # 初始 SOC
        i_sb0 = H * n_u
        i_sg0 = H * n_u + (H + 1)
        row = np.zeros(n)
        row[i_sb0] = 1.0
        A_eq.append(row)
        b_eq.append(soc_b0)
        row = np.zeros(n)
        row[i_sg0] = 1.0
        A_eq.append(row)
        b_eq.append(soc_g0)

        dt = 1.0  # hour
        for h in range(H):
            base = h * n_u
            # 功率平衡: tp + wind + pv + bdis - bch + gdis - gch + buy - sell = load
            row = np.zeros(n)
            row[base + 0] = 1.0
            row[base + 1] = -1.0
            row[base + 2] = 1.0
            row[base + 3] = -1.0
            row[base + 4] = 1.0
            row[base + 5] = 1.0
            row[base + 6] = -1.0
            A_eq.append(row)
            b_eq.append(float(load[h] - wind[h] - pv[h]))

            # SOC bat: s[h+1] - s[h] - eta*bch*dt/E + bdis*dt/(eta*E) = 0
            row = np.zeros(n)
            row[i_sb0 + h + 1] = 1.0
            row[i_sb0 + h] = -1.0
            row[base + 1] = -eta * dt / max(E_bat_mwh, 1e-6)
            row[base + 2] = dt / (max(eta, 1e-6) * max(E_bat_mwh, 1e-6))
            A_eq.append(row)
            b_eq.append(0.0)

            # SOC gas 代理（无效率）
            row = np.zeros(n)
            row[i_sg0 + h + 1] = 1.0
            row[i_sg0 + h] = -1.0
            row[base + 3] = -dt / max(E_gas_mwh, 1e-6)
            row[base + 4] = dt / max(E_gas_mwh, 1e-6)
            A_eq.append(row)
            b_eq.append(0.0)

        # 火电爬坡（第一步相对 prev）
        rate_mw = float(th["rate_max_per_s"]) * float(th["P_cap_W"]) * 3600.0 * 1e-6  # 每小时最大变化 MW
        # u 变化 * P_cap ≈ rate；rate_max_per_s 是标幺/秒
        du_max = float(th["rate_max_per_s"]) * 3600.0  # 标幺/小时
        dP_max = du_max * p_tp_max
        A_ub = []
        b_ub = []
        # p_tp[0] - p_prev <= dP_max ; p_prev - p_tp[0] <= dP_max
        row = np.zeros(n)
        row[0] = 1.0
        A_ub.append(row)
        b_ub.append(p_tp_prev + dP_max)
        row = np.zeros(n)
        row[0] = -1.0
        A_ub.append(row)
        b_ub.append(-p_tp_prev + dP_max)

        # 终端软罚：在目标中加 weight * |s_T - s0| 用 epigraph 可省略，改为二次不可；线性用 s_T 靠近目标
        # 简单：c 中对 s_T 不加；靠 SOC 界即可
        if env.initial_soc is not None and self.cfg.terminal_soc_weight > 0:
            # 鼓励 s_b[H] ≈ init：加辅助不可用，跳过
            pass

        res = linprog(
            c,
            A_ub=np.asarray(A_ub) if A_ub else None,
            b_ub=np.asarray(b_ub) if b_ub else None,
            A_eq=np.asarray(A_eq),
            b_eq=np.asarray(b_eq),
            bounds=bounds,
            method="highs",
            options={"presolve": True, "time_limit": 2.0},
        )
        if not res.success or res.x is None:
            return {"ok": 0.0, "p_tp": p_tp_prev, "p_bat": 0.0, "p_gas": 0.0}
        x = res.x
        p_tp = float(x[0])
        p_bch, p_bdis = float(x[1]), float(x[2])
        p_gch, p_gdis = float(x[3]), float(x[4])
        p_bat = p_bch - p_bdis  # 正=充
        p_gas = p_gch - p_gdis
        return {
            "ok": 1.0,
            "p_tp": p_tp,
            "p_bat": p_bat,
            "p_gas": p_gas,
            "obj": float(res.fun) if res.fun is not None else 0.0,
        }

    def predict(self, obs, deterministic: bool = True) -> dict:
        env = self.env
        try:
            feas = env.get_feasible_action_spec()
        except Exception:
            return {
                "u_tp": np.asarray([1.0], np.float32),
                "u_battery": np.asarray([0.0], np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], np.float32),
            }

        sol = self._solve()
        p_tp_max = float(self.params["thermal"]["P_max_W"]) * 1e-6
        p_bat_max = float(self.params["battery"]["P_cap_W"]) * 1e-6
        p_caes_max = float(self.params["caes"]["P_cap_W"]) * 1e-6

        # 功率 → 指令
        u_tp = float(np.clip(sol["p_tp"] / max(p_tp_max, 1e-6), feas.u_tp_low, feas.u_tp_high))
        # 电池：正充 → u>0
        u_bat = float(np.clip(sol["p_bat"] / max(p_bat_max, 1e-6), feas.u_battery_low, feas.u_battery_high))

        p_gas = float(sol["p_gas"])
        mode = CaesMode.IDLE
        mag = 0.0
        if p_gas > 1.0 and feas.mode_mask.charge:
            mode = CaesMode.CHARGE
            mag = float(np.clip(abs(p_gas) / max(p_caes_max, 1e-6), 0.0, 1.0))
            # 合法幅值映射到 [0.86,1] 区间外由 decoder 处理；用 magnitude 0-1
            mag = max(mag, 0.5)
        elif p_gas < -1.0 and feas.mode_mask.discharge:
            mode = CaesMode.DISCHARGE
            mag = float(np.clip(abs(p_gas) / max(p_caes_max, 1e-6), 0.0, 1.0))
            mag = max(mag, 0.5)
        elif not feas.mode_mask.idle:
            if feas.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, 0.2
            elif feas.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, 0.2

        # 回收段：与 env 硬回收一致，略抬火电
        rem = int(env.episode_steps - env.step_index)
        if rem <= 40:
            u_tp = max(u_tp, float(0.9 * feas.u_tp_high + 0.1 * feas.u_tp_low))

        return {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "caes_mode": int(mode),
            "caes_magnitude": np.asarray([mag], dtype=np.float32),
        }
