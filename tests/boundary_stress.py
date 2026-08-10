"""边界应力测试辅助模块：生成近界动作并验证预检/后验一致性。"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any, Callable
import numpy as np
from actions import CaesMode, FeasibilityOracle
from actions.validator import physical_from_dict
from envs.failures import ConstraintFailure


@dataclass
class BoundaryStressResult:
    """边界应力测试汇总结果(BoundaryStressResult)。

    统计预检拒绝、Oracle 合法、后验成功/失败及 FMU 异常次数。
    """

    n_attempted: int = 0  # 尝试次数
    n_precheck_rejected: int = 0  # 预检拒绝的次数
    n_oracle_legal: int = 0  # Oracle 判合法的次数
    n_post_step_success: int = 0  # 后验成功的次数
    n_post_step_fail: int = 0  # 后验失败的次数
    n_fmu_fail: int = 0  # FMU 失败的次数
    failures: list[dict[str, Any]] = field(default_factory=list)  # 失败详情
    scenarios: dict[str, dict[str, int]] = field(default_factory=dict)  # 场景详情

    @property
    def passed(self) -> bool:
        """是否通过：Oracle 判合法的动作不得后验硬失败，且至少尝试一次。

        Returns:
            无 Oracle 合法后验硬失败且 n_attempted > 0 时为 True。
        """
        return self.n_post_step_fail == 0 and self.n_attempted > 0

    def to_dict(self) -> dict[str, Any]:
        """序列化为可 JSON 写入的字典。

        Returns:
            含计数、passed、scenarios 与 failure_count 的字典。
        """
        return {
            "n_attempted": self.n_attempted,
            "n_precheck_rejected": self.n_precheck_rejected,
            "n_oracle_legal": self.n_oracle_legal,
            "n_post_step_success": self.n_post_step_success,
            "n_post_step_fail": self.n_post_step_fail,
            "n_fmu_fail": self.n_fmu_fail,
            "passed": self.passed,
            "scenarios": self.scenarios,
            "failure_count": len(self.failures),
        }


class BoundaryStressTester:
    """电池与 CAES 边界压力测试器(BoundaryStressTester)。

    按预定义场景生成近界/多约束混合动作，驱动真实环境步进，
    验证非法动作被预检拒绝、Oracle 合法动作后验成功。
    """

    SCENARIOS = (
        "battery_soc_near_min",  # 电池 SOC 接近最小边界
        "battery_soc_near_max",  # 电池 SOC 接近最大值
        "caes_gas_near_min",  # 压空气体接近最小值
        "caes_gas_near_max",  # 压空气体接近最大值
        "caes_hot_near_bound",  # 压空气体高温接近边界
        "caes_cold_near_bound",  # 压空气体低温接近边界
        "caes_pressure_near_bound",  # 压空气体压力接近边界
        "caes_temp_near_bound",  # 压空气体温度接近边界
        "caes_mode_switch",  # 压空气体模式切换
        "thermal_ramp_limit",  # 热电功率限制
        "grid_near_buy_cap",  # 电网接近买入容量
        "grid_near_sell_cap",  # 电网接近卖出容量
        "multi_constraint",  # 多约束
    )

    def __init__(self, oracle: FeasibilityOracle | None = None, seed: int = 0):
        """初始化测试器。

        Args:
            oracle: 可行性 Oracle(FeasibilityOracle)；默认从仓库根加载。
            seed: 随机数种子(seed)，用于可复现采样。
        """
        self.oracle = oracle or FeasibilityOracle.from_root()
        self.rng = np.random.default_rng(seed)

    def sample_boundary_action(
        self,
        outputs: dict[str, float],
        previous_thermal_w: float,
        scenario: str | None = None,
    ) -> tuple[dict, str]:
        """生成近边界或多约束应力动作。

        先在可行域内采样，再按场景贴齐边界；约 35% 概率故意越界以验证预检。

        Args:
            outputs: 当前环境输出(outputs)字典。
            previous_thermal_w: 上一时刻火电功率(previous_thermal_w)，单位 W。
            scenario: 场景名(scenario)；为 None 时随机选择 SCENARIOS 之一。

        Returns:
            (action, scenario)：混合动作字典与所用场景名。
        """
        scenario = scenario or str(self.rng.choice(self.SCENARIOS))
        feasible = self.oracle.compute(outputs, previous_thermal_w)  # 计算可行域
        # 先在可行域内采样，再按 scenario 贴齐边界；约 35% 再故意越界以验证预检
        bias_illegal = self.rng.random() < 0.35
        u_tp = float(
            self.rng.uniform(feasible.u_tp_low, feasible.u_tp_high)
        )  # 火电指令：可行域内均匀采样
        u_bat = float(
            self.rng.uniform(feasible.u_battery_low, feasible.u_battery_high)
        )  # 电池指令：可行域内均匀采样
        modes = [
            m
            for m, ok in zip(
                (CaesMode.DISCHARGE, CaesMode.IDLE, CaesMode.CHARGE),
                (
                    feasible.mode_mask.discharge,
                    feasible.mode_mask.idle,
                    feasible.mode_mask.charge,
                ),
            )
            if ok
        ] or [CaesMode.IDLE]
        mode = modes[int(self.rng.integers(0, len(modes)))]  # 在允许模式中随机选
        mag = (
            0.0 if mode == CaesMode.IDLE else float(self.rng.uniform(0.0, 1.0))
        )  # IDLE 固定 0，否则幅值 [0, 1] 均匀采样
        if scenario.startswith("battery_soc_near"):
            if "min" in scenario:
                u_bat = float(feasible.u_battery_low)  # 电池指令贴齐可行下界
            else:
                u_bat = float(feasible.u_battery_high)  # 电池指令贴齐可行上界
            if bias_illegal:
                u_bat = float(
                    np.clip(u_bat + (0.2 if "max" in scenario else -0.2), -1.5, 1.5)
                )  # 故意越出可行界
        elif scenario.startswith("caes_gas") or "caes_" in scenario:
            if feasible.mode_mask.charge and "max" in scenario:
                mode, mag = CaesMode.CHARGE, 1.0  # 满充逼近气库上界
            elif feasible.mode_mask.discharge and "min" in scenario:
                mode, mag = CaesMode.DISCHARGE, 1.0  # 满放逼近气库下界
            elif scenario == "caes_mode_switch":
                mode = (
                    CaesMode.CHARGE
                    if mode == CaesMode.DISCHARGE
                    else CaesMode.DISCHARGE
                )
                if mode == CaesMode.DISCHARGE and not feasible.mode_mask.discharge:
                    mode = CaesMode.IDLE
                if mode == CaesMode.CHARGE and not feasible.mode_mask.charge:
                    mode = CaesMode.IDLE
                mag = 1.0 if mode != CaesMode.IDLE else 0.0  # 强制切换充/放；掩码禁止则 IDLE
        elif scenario == "thermal_ramp_limit":
            u_tp = float(
                feasible.u_tp_high if self.rng.random() < 0.5 else feasible.u_tp_low
            )  # 火电贴齐爬坡上/下界（各 50%）
            if bias_illegal:
                u_tp = float(np.clip(u_tp + (0.1 if u_tp > 0.5 else -0.1), 0.0, 1.2))  # 故意越出可行界
        elif scenario.startswith("grid_near"):
            # 储能同向拉满，逼近联络线购/售电极限
            if "buy" in scenario:
                mode, mag = (
                    (CaesMode.CHARGE, 1.0)
                    if feasible.mode_mask.charge
                    else (CaesMode.IDLE, 0.0)
                )
                u_bat = float(feasible.u_battery_high)
            else:
                mode, mag = (
                    (CaesMode.DISCHARGE, 1.0)
                    if feasible.mode_mask.discharge
                    else (CaesMode.IDLE, 0.0)
                )
                u_bat = float(feasible.u_battery_low)
        elif scenario == "multi_constraint":
            # 火电上界 + 电池极端 + CAES 满功率，多约束同时加压
            u_tp = float(feasible.u_tp_high)
            u_bat = float(
                feasible.u_battery_high
                if self.rng.random() < 0.5
                else feasible.u_battery_low
            )
            if feasible.mode_mask.charge:
                mode, mag = CaesMode.CHARGE, 1.0
            elif feasible.mode_mask.discharge:
                mode, mag = CaesMode.DISCHARGE, 1.0
        action = {
            "u_tp": np.asarray([u_tp], dtype=np.float32),
            "u_battery": np.asarray([u_bat], dtype=np.float32),
            "u_caes": np.asarray([float((__import__("actions.caes_u", fromlist=["u_from_mode_mag"]).u_from_mode_mag(mode, mag)))], dtype=np.float32),
        }
        return action, scenario

    def run(
        self,
        env,
        *,
        n_actions: int = 20000,
        reset_every: int = 50,
        step_fn: Callable | None = None,
    ) -> BoundaryStressResult:
        """对真实环境执行边界偏置步进循环。

        非法动作须被预检拒绝；Oracle 判合法的动作须后验成功。

        Args:
            env: PowerSystemEnv 或兼容接口的环境实例。
            n_actions: 总尝试步数(n_actions)。
            reset_every: 每 episode 最多步数(reset_every)，超限则 reset。
            step_fn: 可选自定义步进函数(step_fn)；签名 (env, action) -> step 五元组。

        Returns:
            汇总统计 BoundaryStressResult。
        """
        result = BoundaryStressResult()
        obs, info = env.reset(seed=0)
        episode_steps = 0
        for i in range(n_actions):
            if env.last_outputs is None:
                obs, info = env.reset(seed=i)
            outputs = dict(env.last_outputs)
            prev_th = float(env.previous_thermal)
            action, scenario = self.sample_boundary_action(outputs, prev_th)
            sc = result.scenarios.setdefault(
                scenario,
                {"attempted": 0, "precheck_rejected": 0, "post_ok": 0, "post_fail": 0},
            )
            sc["attempted"] += 1
            result.n_attempted += 1
            # 预检（与 env 一致：validator + oracle）
            feasible = env.get_feasible_action_spec()
            try:
                hybrid = physical_from_dict(action)
                env.hybrid_validator.validate(hybrid, feasible)
                ok, reason = self.oracle.check_action_executable(
                    hybrid, outputs, feasible, prev_th
                )
                if not ok:
                    raise ConstraintFailure(reason or "precheck")
            except Exception:
                result.n_precheck_rejected += 1
                sc["precheck_rejected"] += 1
                episode_steps += 1
                if episode_steps >= reset_every:
                    obs, info = env.reset(seed=i + 1)
                    episode_steps = 0
                continue
            result.n_oracle_legal += 1
            if step_fn is not None:
                obs, reward, term, trunc, info = step_fn(env, action)
            else:
                obs, reward, term, trunc, info = env.step(action)
            episode_steps += 1
            if info.get("physically_valid") and info.get("transition_valid"):
                result.n_post_step_success += 1
                sc["post_ok"] += 1
            else:
                ft = info.get("failure_type") or ""
                if "Fmu" in ft or "Fmi" in ft:
                    result.n_fmu_fail += 1
                else:
                    result.n_post_step_fail += 1
                    sc["post_fail"] += 1
                    result.failures.append(
                        {
                            "scenario": scenario,
                            "failure_type": ft,
                            "fine_failure_type": info.get("fine_failure_type"),
                            "reason": info.get("failure_reason"),
                            "action": {
                                k: (float(v[0]) if hasattr(v, "__len__") else v)
                                for k, v in action.items()
                            },
                        }
                    )
                obs, info = env.reset(seed=i + 17)
                episode_steps = 0
                continue
            if term or trunc or episode_steps >= reset_every:
                obs, info = env.reset(seed=i + 3)
                episode_steps = 0
        return result
