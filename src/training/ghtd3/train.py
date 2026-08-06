"""GHTD3 训练：高层 goal + 底层 Hybrid-GiveSafe 执行。"""

from __future__ import annotations

import json
import math
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from actions import CaesMode
from controllers.price_aware_rule import PriceAwareRuleController
from controllers.rule_based_controller import RuleBasedController
from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from safety import GiveSafeController, NoSafeActionFoundError, load_givesafe_config
from training.evaluate_td3 import evaluate_policy
from training.hybrid_td3.train import annual_episode_start_seconds

from .agent import GHTD3Agent
from .buffers import HighTransition, LowTransition
from .goals import (
    DEFAULT_SOC_KEYS,
    blend_goal_with_prior,
    extract_soc,
    extract_soc_from_obs,
    extract_plant_state,
    goal_transition,
    goal_transition_intent,
    plant_intent_vector,
    structured_intrinsic_reward,
    market_conditioned_goal_prior,
    residual_scale_from_goal,
    achieved_goal_from_cycle,
)


def load_ghtd3_config(path: str | Path | None = None) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    p = Path(path) if path else root / "src" / "config" / "ghtd3_config.yaml"
    with Path(p).open(encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


class GHTD3PolicyWrapper:
    """评估：5 维 Modelica goal + Hybrid 锚定残差 + GiveSafe。"""

    def __init__(self, agent: GHTD3Agent, env: PowerSystemEnv, controller: GiveSafeController, cfg: dict):
        self.agent = agent
        self.env = env
        self.controller = controller
        self.cfg = dict(cfg)
        self.c = int(cfg.get("subgoal_interval", 8))
        self.step_in_cycle = 0
        self.goal = np.zeros(agent.goal_dim, dtype=np.float32)
        self._pending_intent = None

    def on_episode_reset(self, info: dict[str, Any]) -> None:
        self.step_in_cycle = 0
        self.goal = np.zeros(self.agent.goal_dim, dtype=np.float32)
        self._pending_intent = None

    def _select_goal_with_prior(self, obs: np.ndarray) -> np.ndarray:
        goal = self.agent.select_goal(obs, deterministic=True, random=False)
        if not bool(self.cfg.get("market_goal_prior", True)):
            return goal
        buy = None
        if getattr(self.env, "price_profile", None) is not None:
            try:
                buy, _ = self.env.price_profile.prices_at(float(self.env.adapter.time))
            except Exception:
                buy = None
        soc_now = extract_soc_from_obs(obs, 2)
        soc_init = None
        if self.env.initial_soc is not None:
            soc_init = extract_soc(self.env.initial_soc, DEFAULT_SOC_KEYS)
        rem = int(self.env.episode_steps - self.env.step_index)
        recovery = rem <= int(self.cfg.get("recovery_goal_horizon_steps", 36) or 0)
        th_mean = None
        outs = self.env.last_outputs or {}
        if outs:
            th_mean = 0.5 * (float(outs.get("caes_hot_soc", 0.5)) + float(outs.get("caes_cold_soc", 0.5)))
        prior = market_conditioned_goal_prior(
            buy,
            soc_now,
            soc_init,
            goal_low=self.agent.goal_low,
            goal_high=self.agent.goal_high,
            charge_threshold=float(self.cfg.get("charge_threshold", 0.40)),
            discharge_threshold=float(self.cfg.get("discharge_threshold", 0.90)),
            recovery=recovery,
            strength=float(self.cfg.get("market_prior_strength", 0.14)),
            th_mean=th_mean,
        )
        w = float(self.cfg.get("market_prior_weight_end", self.cfg.get("market_prior_weight", 0.2)))
        if recovery:
            w = max(w, float(self.cfg.get("recovery_prior_weight", 0.92)))
        return blend_goal_with_prior(
            goal, prior, prior_weight=w, goal_low=self.agent.goal_low, goal_high=self.agent.goal_high
        )

    def predict(self, obs, deterministic: bool = True):
        if self.step_in_cycle % self.c == 0:
            self.goal = self._select_goal_with_prior(obs)
        try:
            feasible = self.env.get_feasible_action_spec()
        except FeasibleSetEmpty:
            return {
                "u_tp": np.asarray([1.0], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }

        def propose():
            return self.agent.select_composed_action(
                obs, self.goal, feasible, deterministic=deterministic
            )

        try:
            gs = self.controller.select_safe_action(
                self.env.last_outputs,
                self.env.previous_thermal,
                propose,
                deterministic=deterministic,
                feasible_override=feasible,
            )
            action = gs.safe_action
        except NoSafeActionFoundError:
            action = {
                "u_tp": np.asarray([float(feasible.u_tp_high)], dtype=np.float32),
                "u_battery": np.asarray([0.0], dtype=np.float32),
                "caes_mode": int(CaesMode.IDLE),
                "caes_magnitude": np.asarray([0.0], dtype=np.float32),
            }
        self.step_in_cycle += 1
        outs0 = self.env.last_outputs or {}
        self._pending_intent = plant_intent_vector(outs0) if outs0 else None
        return action

    def on_transition(self, info: dict[str, Any]) -> None:
        if not info.get("transition_valid"):
            return
        outs = info.get("observations") or self.env.last_outputs or {}
        if not outs or self._pending_intent is None:
            return
        intent1 = plant_intent_vector(outs)
        self.goal = goal_transition_intent(
            self._pending_intent, self.goal, intent1, self.agent.goal_low, self.agent.goal_high
        )
        self._pending_intent = intent1


def _acquire_run_lock(run_dir: Path) -> object | None:
    """同一 run_dir 仅允许一个训练进程（防 ComfyUI/venv 双开写坏）。

    Returns:
        保持打开的文件句柄；失败返回 None。
    """
    import os
    import sys

    lock_path = run_dir / "train" / "instance.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)

    def _pid_alive(pid: int) -> bool:
        if pid <= 0:
            return False
        if sys.platform.startswith("win"):
            try:
                import ctypes

                k = ctypes.windll.kernel32
                h = k.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
                if not h:
                    return False
                k.CloseHandle(h)
                return True
            except Exception:
                return False
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False

    if lock_path.is_file():
        try:
            old = int((lock_path.read_text(encoding="utf-8") or "0").strip().split()[0])
        except Exception:
            old = 0
        if _pid_alive(old) and old != os.getpid():
            print(f"[single-instance] blocked: run_dir locked by pid={old}", flush=True)
            return None
        try:
            lock_path.unlink()
        except OSError:
            pass
    try:
        fd = os.open(str(lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, f"{os.getpid()}\n".encode("utf-8"))
        # keep fd open for process lifetime
        return os.fdopen(fd, "a", encoding="utf-8")
    except FileExistsError:
        print("[single-instance] blocked: lock race", flush=True)
        return None


def run_ghtd3_training(
    total_valid_steps: int = 10000,
    run_dir: str | Path = "runs/ghtd3_smoke",
    seed: int = 0,
    config_path: str | Path | None = None,
    annual_evaluation: bool = False,
    resume_from: str | Path | None = None,
    skip_bc: bool = False,
) -> dict[str, Any]:
    run_dir = Path(run_dir).resolve()
    for name in ("config", "train", "checkpoints", "trajectories"):
        (run_dir / name).mkdir(parents=True, exist_ok=True)
    _run_lock = _acquire_run_lock(run_dir)
    if _run_lock is None:
        return {
            "status": "blocked_single_instance",
            "run_dir": str(run_dir),
            "error": "another train_ghtd3 holds instance.lock for this run_dir",
        }
    root = Path(__file__).resolve().parents[3]
    full_cfg = load_ghtd3_config(config_path)
    cfg = dict(full_cfg.get("ghtd3") or full_cfg)
    cfg_src = Path(config_path) if config_path else (root / "src/config/ghtd3_config.yaml")
    if not cfg_src.is_file():
        cfg_src = root / "src/config/ghtd3_config.yaml"
    shutil.copy2(cfg_src, run_dir / "config" / "ghtd3_config.yaml")
    for name in ("env_config.yaml", "reward_config.yaml", "givesafe_config.yaml", "device_params.yaml"):
        src = root / "src/config" / name
        if src.exists():
            shutil.copy2(src, run_dir / "config" / name)

    np.random.seed(seed)
    env = PowerSystemEnv(run_id=run_dir.name, forecast_enabled=True)
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = GHTD3Agent(obs_dim, cfg)
    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    controller = GiveSafeController(oracle=env.oracle, shadow=None, config=gs_cfg)

    # Hybrid 锚定（冻结 raw-obs 执行器）；abs 主线 hybrid_anchor=false 跳过
    hybrid_path = cfg.get("hybrid_anchor_path")
    if bool(cfg.get("hybrid_anchor", False)) and hybrid_path:
        hp = Path(hybrid_path)
        if not hp.is_file():
            hp = root / hybrid_path
        if hp.is_file():
            from .hybrid_anchor import HybridAnchor

            anchor = HybridAnchor(obs_dim, hp, device=str(agent.device))
            # action_residual：永不移植绝对头；仅 hybrid_init_low 时移植（旧路径）
            _mode = str(cfg.get("execution_mode", "action_residual")).lower()
            do_tx = bool(cfg.get("hybrid_init_low", False)) and _mode not in ("action_residual", "tea")
            trep = agent.attach_hybrid_anchor(anchor, transplant=do_tx)
            print(f"[hybrid-init] {trep}")
        else:
            print(f"[warn] hybrid_anchor_path missing: {hybrid_path}; fallback pure residual low")
            agent.hybrid_anchor_enabled = False
    else:
        agent.hybrid_anchor_enabled = False
        agent._hybrid_anchor = None
        print(
            f"[abs-gc] no hybrid anchor; execution_mode={cfg.get('execution_mode')} "
            f"her={cfg.get('goal_relabel_mode', 'her_mix')} goal_scale={cfg.get('goal_input_scale')}"
        )

    bc_summary: dict[str, Any] | None = None
    if resume_from:
        try:
            agent.load(resume_from, strict=False)
        except Exception as exc:
            print(f"[warn] resume load partial: {exc}")
            agent.load(resume_from, strict=False)
        skip_bc = True
    # 预热：F-MLE（绝对 GC，无教师）优先；否则 residual MLE / 规则 BC
    if bool(cfg.get("bc_pretrain", True)) and not skip_bc:
        low_bc = None
        high_bc = None
        mle_stats = None
        f_mle_stats = None
        demos_n = 0
        # F-MLE：可行规则轨迹逆动力学（Safe Market-GHTD3 主线）
        if bool(cfg.get("f_mle_pretrain", False)) or (
            str(cfg.get("execution_mode", "")).lower() == "goal_conditioned"
            and bool(cfg.get("f_mle_pretrain", True))
            and not bool(cfg.get("hybrid_anchor", False))
        ):
            from .feasible_mle import f_mle_pretrain

            f_mle_stats = f_mle_pretrain(env, agent, cfg=cfg, seed=seed)
            demos_n = int(f_mle_stats.get("n_demos", 0))
            low_bc = f_mle_stats.get("low")
            high_bc = f_mle_stats.get("high")
            print(f"[F-MLE] {f_mle_stats}")
        else:
            from .bc_pretrain import (
                behavior_clone_low_actor,
                bc_pretrain_high_goals,
                collect_hierarchical_demos,
            )

            demos = collect_hierarchical_demos(
                env,
                agent,
                n_windows=int(cfg.get("bc_windows", 4)),
                seed=seed,
                price_aware=True,
                cfg=cfg,
            )
            demos_n = int(demos["obs"].shape[0])
            # 逆动力学残差 MLE（action_residual + Hybrid 锚，非主线）
            if (
                str(cfg.get("execution_mode", "")).lower() in ("action_residual", "tea")
                and bool(cfg.get("mle_pretrain_residual", True))
                and agent._hybrid_anchor is not None
            ):
                from .residual_mle import collect_residual_mle_demos, mle_pretrain_residual

                mle_demos = collect_residual_mle_demos(
                    env,
                    agent,
                    n_windows=int(cfg.get("mle_windows", cfg.get("bc_windows", 4))),
                    seed=seed,
                    cfg=cfg,
                )
                mle_stats = mle_pretrain_residual(
                    agent,
                    mle_demos,
                    epochs=int(cfg.get("mle_epochs", 25)),
                    fit_mode=bool(cfg.get("mle_fit_mode", False)),
                )
                print(f"[residual-mle] {mle_stats}")
            elif bool(cfg.get("bc_pretrain_low", not agent.hybrid_anchor_enabled)):
                low_bc = behavior_clone_low_actor(
                    agent,
                    demos,
                    epochs=int(cfg.get("bc_epochs_low", 30)),
                )
            if bool(cfg.get("bc_pretrain_high", True)) and f_mle_stats is None:
                high_bc = bc_pretrain_high_goals(
                    agent,
                    demos,
                    epochs=int(cfg.get("bc_epochs_high", 20)),
                )
        bc_summary = {
            "low": low_bc,
            "high": high_bc,
            "residual_mle": mle_stats,
            "f_mle": f_mle_stats,
            "n_demos": demos_n,
            "execution_mode": str(cfg.get("execution_mode")),
            "principles": {
                "F-MLE": f_mle_stats is not None,
                "MSGP": bool(cfg.get("market_goal_prior", True)),
                "MS-HER": str(cfg.get("goal_relabel_mode", "")).lower() in ("ms_her", "her_mix")
                or bool(cfg.get("ms_her_weighting", False)),
            },
        }
        (run_dir / "train" / "bc_summary.json").write_text(
            json.dumps(bc_summary, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
        )

    c = int(cfg.get("subgoal_interval", 8))
    alpha0 = float(cfg.get("intrinsic_alpha", 0.35))
    alpha1 = float(cfg.get("intrinsic_alpha_end", cfg.get("intrinsic_alpha", 0.25)))
    eps0 = float(cfg.get("epsilon_start", 0.3))
    eps1 = float(cfg.get("epsilon_end", 0.05))
    learn_lo = int(cfg.get("learning_starts_low", 512))
    learn_hi = int(cfg.get("learning_starts_high", 64))
    bs_lo = int(cfg.get("low_batch_size", 128))
    bs_hi = int(cfg.get("high_batch_size", 64))
    n_grad_lo = int(cfg.get("gradient_steps_low", 2))
    n_grad_hi = int(cfg.get("gradient_steps_high", 3))
    # 两阶段：前 phase_a_steps 仅 prior/随机高层目标，不更新高层网络
    # 续训时默认跳过 Phase-A（底层已稳），除非配置 force_phase_a_on_resume
    phase_a_steps = int(cfg.get("phase_a_steps", 0) or 0)
    if resume_from and not bool(cfg.get("force_phase_a_on_resume", False)):
        phase_a_steps = 0
    prior_w0 = float(cfg.get("market_prior_weight", 0.55))
    prior_w1 = float(cfg.get("market_prior_weight_end", 0.15))
    prior_anneal_start = float(cfg.get("market_prior_anneal_start", 0.30))  # progress 之后开始退火
    goal_dropout = float(cfg.get("goal_dropout", 0.0) or 0.0)
    log_every = int(cfg.get("log_every", 500) or 500)
    residual_train_start = int(cfg.get("residual_train_start", 0) or 0)
    intrinsic_weights = cfg.get("intrinsic_weights") or [1.5, 1.2, 0.25, 0.4]
    # residual 可用更小 lr
    if agent.hybrid_anchor_enabled:
        for g in agent.lo_actor_opt.param_groups:
            g["lr"] = float(cfg.get("residual_actor_lr", cfg.get("actor_lr", 1e-4)))

    valid_steps = 0
    episode = 0
    cycle_idx_ep = 0
    step_log: list[dict] = []
    # 续训合并完整训练曲线
    if resume_from:
        for cand in (
            run_dir / "train" / "step_log.json",
            Path(resume_from).resolve().parents[1] / "train" / "step_log.json",
        ):
            if cand.is_file():
                try:
                    prev = json.loads(cand.read_text(encoding="utf-8"))
                    if isinstance(prev, list) and prev:
                        step_log = list(prev)
                        break
                except Exception:
                    pass
    log_resume_base = max((int(r.get("valid_step", 0)) for r in step_log), default=0)
    stats = {
        "high_goal_count": 0,
        "low_step_count": 0,
        "givesafe_reject": 0,
        "physical_ok": 0,
        "her_mix": str(cfg.get("goal_relabel_mode", "her_mix")),
        "phase_a_steps": phase_a_steps,
    }

    def reset_ep(idx: int):
        nonlocal cycle_idx_ep
        start = annual_episode_start_seconds(env.config["fmu"], env.episode_steps, idx)
        obs, info = env.reset(seed=seed + idx, options={"start_time": start})
        cycle_idx_ep = 0
        return obs, info

    def scheduled_prior_weight(progress: float, *, recovery: bool) -> float:
        if recovery:
            return max(prior_w0, float(cfg.get("recovery_prior_weight", 0.92)))
        if progress <= prior_anneal_start:
            return prior_w0
        # 线性退火 prior_w0 → prior_w1
        t = (progress - prior_anneal_start) / max(1e-6, 1.0 - prior_anneal_start)
        t = float(np.clip(t, 0.0, 1.0))
        return float(prior_w0 + (prior_w1 - prior_w0) * t)

    def scheduled_alpha(progress: float) -> float:
        return float(alpha0 + (alpha1 - alpha0) * progress)

    obs, _ = reset_ep(episode)
    goal = np.zeros(agent.goal_dim, dtype=np.float32)
    cycle_ext = 0.0
    cycle_start_obs = obs.copy()
    outs_i = env.last_outputs or {}
    cycle_soc_seq: list[np.ndarray] = [
        plant_intent_vector(outs_i) if outs_i else extract_soc_from_obs(obs, 2)
    ]
    cycle_act_seq: list[dict] = []
    cycle_u_tp: list[float] = []
    cycle_u_tp_h: list[float] = []
    steps_in_cycle = 0
    cycle_goal = goal.copy()
    if bool(cfg.get("ltar_enabled", False)):
        _algo = "LTAR-TD3"
    elif bool(cfg.get("stfr_enabled", False)) or str(
        cfg.get("high_level_mode", "")
    ).lower() in ("prior_only", "prior", "stfr_a", "stfr"):
        _algo = "STFR"
    else:
        _algo = "SafeMarketGHTD3"
    result: dict[str, Any] = {
        "status": "running",
        "algorithm": _algo,
        "requested_valid_steps": total_valid_steps,
        "ghtd3": cfg,
        "observation_dim": obs_dim,
        "bc_pretrain": bc_summary,
    }

    try:
        while valid_steps < total_valid_steps:
            progress = valid_steps / max(total_valid_steps, 1)
            # TEA：课程扩张 ρ(t) + mode 解锁
            agent.set_progress(progress)
            try:
                agent.set_episode_context(
                    rem_steps=int(env.episode_steps - env.step_index),
                    episode_steps=int(env.episode_steps),
                )
            except Exception:
                pass
            eps = eps0 + (eps1 - eps0) * progress

            try:
                feasible = env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                cycle_ext = 0.0
                cycle_start_obs = obs.copy()
                cycle_soc_seq = [extract_soc_from_obs(obs, agent.goal_dim)]
                cycle_act_seq = []
                continue

            # 高层：周期起点采样 goal
            # STFR Stage A：high_level_mode=prior_only → 仅市场/回收 prior，不采样可学高层
            high_level_mode = str(cfg.get("high_level_mode", "learned")).lower()
            prior_only = high_level_mode in ("prior_only", "prior", "stfr_a", "stfr")
            if steps_in_cycle == 0:
                in_phase_a = valid_steps < phase_a_steps
                random_goal = (
                    (not prior_only)
                    and (
                        in_phase_a
                        or (valid_steps < learn_lo)
                        or (np.random.rand() < eps)
                    )
                )
                if prior_only:
                    goal = np.zeros(agent.goal_dim, dtype=np.float32)
                elif in_phase_a and bool(cfg.get("market_goal_prior", True)):
                    goal = np.zeros(agent.goal_dim, dtype=np.float32)
                else:
                    goal = agent.select_goal(obs, deterministic=False, random=random_goal)
                if bool(cfg.get("market_goal_prior", True)) or prior_only:
                    buy = None
                    if getattr(env, "price_profile", None) is not None:
                        try:
                            buy, _ = env.price_profile.prices_at(float(env.adapter.time))
                        except Exception:
                            buy = None
                    soc_now = extract_soc_from_obs(obs, 2)
                    soc_init = None
                    if env.initial_soc is not None:
                        soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS)
                    rem = int(env.episode_steps - env.step_index)
                    recovery = rem <= int(cfg.get("recovery_goal_horizon_steps", 36) or 0)
                    outs0 = env.last_outputs or {}
                    th_mean = None
                    if outs0:
                        th_mean = 0.5 * (
                            float(outs0.get("caes_hot_soc", 0.5)) + float(outs0.get("caes_cold_soc", 0.5))
                        )
                    prior = market_conditioned_goal_prior(
                        buy,
                        soc_now,
                        soc_init,
                        goal_low=agent.goal_low,
                        goal_high=agent.goal_high,
                        charge_threshold=float(cfg.get("charge_threshold", 0.40)),
                        discharge_threshold=float(cfg.get("discharge_threshold", 0.90)),
                        recovery=recovery,
                        strength=float(cfg.get("market_prior_strength", 0.14)),
                        th_mean=th_mean,
                    )
                    if prior_only:
                        w = 1.0
                    else:
                        w = scheduled_prior_weight(progress, recovery=recovery)
                        if in_phase_a:
                            w = max(w, 0.85)
                    goal = blend_goal_with_prior(
                        goal, prior, prior_weight=w, goal_low=agent.goal_low, goal_high=agent.goal_high
                    )
                # STFR：库存意图维主导 — 可选清零非库存/非套利维，避免假 goal 维干扰
                if bool(cfg.get("stfr_inventory_goal_focus", prior_only)) and goal.size >= 5:
                    # 保留 bat, gas, arb；压低 th / u_tp 意图（由残差+教师承担）
                    goal = goal.copy()
                    goal[2] *= 0.25  # th_mean intent
                    goal[3] *= 0.25  # u_tp bias intent
                cycle_goal = goal.copy()
                cycle_start_obs = obs.copy()
                cycle_ext = 0.0
                outs0 = env.last_outputs or {}
                cycle_soc_seq = [plant_intent_vector(outs0) if outs0 else extract_soc_from_obs(obs, 2)]
                cycle_act_seq = []
                cycle_u_tp = []
                cycle_u_tp_h = []
                stats["high_goal_count"] += 1

            obs_before = obs.copy()
            outs_before = env.last_outputs or {}
            intent_before = plant_intent_vector(outs_before) if outs_before else extract_soc_from_obs(obs_before, 2)
            g_before = goal.copy()
            rem = int(env.episode_steps - env.step_index)
            recovery_now = rem <= int(cfg.get("recovery_goal_horizon_steps", 36) or 0)
            gd = 0.0 if recovery_now else goal_dropout
            if gd > 0 and np.random.rand() < gd:
                g_exec = np.zeros_like(g_before)
            else:
                g_exec = g_before

            # Hybrid 基线动作（用于 u_tp 跟踪与 α_res=0 下界）
            a_h_scalars = None
            if agent._hybrid_anchor is not None:
                try:
                    a_h_scalars = agent._hybrid_anchor.act_scalars(obs_before, feasible, deterministic=True)
                except Exception:
                    a_h_scalars = None
            u_tp_h = float(a_h_scalars["u_tp"]) if a_h_scalars else 0.7

            def propose():
                if (
                    not agent.hybrid_anchor_enabled
                    and valid_steps < learn_lo
                    and np.random.rand() < 0.5
                ):
                    if bool(cfg.get("price_aware_bootstrap", True)) and getattr(env, "market_enabled", False):
                        return PriceAwareRuleController(env).predict(obs_before)
                    return RuleBasedController(env).predict(obs_before)
                # goal_conditioned：始终 a=π(s,g)；blend 模式才用 residual_scale
                scale = None
                if str(cfg.get("execution_mode", "action_residual")).lower() == "blend":
                    if valid_steps < residual_train_start:
                        scale = 0.0
                    else:
                        scale = residual_scale_from_goal(
                            g_exec,
                            alpha0=float(cfg.get("residual_alpha0", 0.0)),
                            alpha_max=float(cfg.get("residual_alpha_max", 0.28)),
                        )
                # 价差门控 β：把当前买电价挂到 agent
                if bool(cfg.get("residual_beta_price_gate", False)) and getattr(env, "price_profile", None) is not None:
                    try:
                        bp, _ = env.price_profile.prices_at(float(env.adapter.time))
                        agent._last_buy_price = float(bp) if bp is not None else None
                    except Exception:
                        agent._last_buy_price = None
                return agent.select_composed_action(
                    obs_before,
                    g_exec,
                    feasible,
                    deterministic=False,
                    residual_scale=scale,
                )
            def _on_rej(*_args):
                stats["givesafe_reject"] += 1

            try:
                gs = controller.select_safe_action(
                    env.last_outputs,
                    env.previous_thermal,
                    propose,
                    deterministic=False,
                    feasible_override=feasible,
                    on_rejection=_on_rej,
                )
                action = gs.safe_action
            except NoSafeActionFoundError:
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                continue

            next_obs, r_ext, terminated, truncated, info = env.step(action)
            stats["low_step_count"] += 1

            if not info.get("transition_valid"):
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                continue

            outs = info.get("observations") or env.last_outputs or {}
            intent_after = plant_intent_vector(outs) if outs else intent_before
            alpha = scheduled_alpha(progress)
            try:
                u_tp_act = float(np.asarray(action["u_tp"]).reshape(-1)[0])
            except Exception:
                u_tp_act = float(u_tp_h)
            r_int, shape_terms = structured_intrinsic_reward(
                intent_before,
                g_before,
                intent_after,
                u_tp_act,
                u_tp_h,
                float(r_ext),
                alpha=alpha,
                weights=intrinsic_weights,
            )
            # TEA 安全：相对 Hybrid 的 CAES 幅度偏离惩罚，抑制无意义过吞吐
            thr_pen = float(cfg.get("tea_mag_dev_penalty", 0.0) or 0.0)
            if thr_pen > 0.0 and a_h_scalars is not None:
                try:
                    mag_act = float(np.asarray(action["caes_magnitude"]).reshape(-1)[0])
                    mag_h = float(a_h_scalars["caes_magnitude"])
                    mode_act = int(action["caes_mode"])
                    mode_h = int(a_h_scalars["caes_mode"])
                    dev = abs(mag_act - mag_h) + (0.35 if mode_act != mode_h else 0.0)
                    r_int = float(r_int) - thr_pen * dev
                    shape_terms = {**shape_terms, "mag_dev_penalty": thr_pen * dev}
                except Exception:
                    pass
            # 终端 SOC 软屏障：回收窗内惩罚偏离初始能量库存（根因：多种子 SOC 门控失败）
            soc_pen_w = float(cfg.get("tea_terminal_soc_penalty", 0.0) or 0.0)
            # LTAR：乘上 λ，约束越紧惩罚越强
            if bool(getattr(agent, "ltar_enabled", False)):
                soc_pen_w = soc_pen_w + float(cfg.get("ltar_reward_cost_coef", 0.35)) * float(
                    getattr(agent, "lambda_soc", 0.0)
                )
            if soc_pen_w > 0.0:
                try:
                    H = int(cfg.get("recovery_goal_horizon_steps", 40) or 40)
                    rem_now = int(env.episode_steps - env.step_index)
                    if rem_now <= H and env.initial_soc is not None:
                        soc_now = extract_soc_from_obs(np.asarray(next_obs, dtype=np.float32), 2)
                        soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS)
                        n = min(soc_now.size, soc_init.size, 2)
                        err = float(np.sum(np.abs(soc_now[:n] - soc_init[:n])))
                        w = 1.0 - float(rem_now) / max(H, 1)
                        pen = soc_pen_w * w * err
                        r_int = float(r_int) - pen
                        shape_terms = {**shape_terms, "terminal_soc_penalty": pen}
                except Exception:
                    pass
            goal_next = goal_transition_intent(
                intent_before, g_before, intent_after, agent.goal_low, agent.goal_high
            )
            bounds = {
                "u_tp_low": float(info.get("u_tp_dynamic_low", feasible.u_tp_low)),
                "u_tp_high": float(info.get("u_tp_dynamic_high", feasible.u_tp_high)),
                "u_battery_low": float(info.get("u_battery_dynamic_low", feasible.u_battery_low)),
                "u_battery_high": float(info.get("u_battery_dynamic_high", feasible.u_battery_high)),
            }
            try:
                next_feas = env.get_feasible_action_spec()
                next_bounds = {
                    "u_tp_low": next_feas.u_tp_low,
                    "u_tp_high": next_feas.u_tp_high,
                    "u_battery_low": next_feas.u_battery_low,
                    "u_battery_high": next_feas.u_battery_high,
                }
                next_mask = next_feas.mode_mask.as_bool_array()
            except Exception:
                next_bounds = dict(bounds)
                next_mask = np.ones(3, dtype=bool)

            hybrid = info.get("hybrid_action") or {
                "u_tp": float(np.asarray(action["u_tp"]).ravel()[0]),
                "u_battery": float(np.asarray(action["u_battery"]).ravel()[0]),
                "caes_mode": int(action["caes_mode"]),
                "caes_magnitude": float(np.asarray(action["caes_magnitude"]).ravel()[0]),
            }
            done_flag = bool(terminated or truncated)
            agent.lo_buffer.add(
                LowTransition(
                    observation=obs_before.astype(np.float32),
                    goal=g_before.astype(np.float32),
                    hybrid_action={
                        "u_tp": float(hybrid["u_tp"]),
                        "u_battery": float(hybrid["u_battery"]),
                        "caes_mode": int(hybrid["caes_mode"]),
                        "caes_magnitude": float(hybrid.get("caes_magnitude", 0.0)),
                    },
                    reward_int=float(r_int),
                    next_observation=np.asarray(next_obs, dtype=np.float32),
                    next_goal=goal_next.astype(np.float32),
                    terminated=done_flag,
                    valid_mode_mask=feasible.mode_mask.as_bool_array(),
                    next_valid_mode_mask=next_mask,
                    dynamic_action_bounds=bounds,
                    next_dynamic_action_bounds=next_bounds,
                    reward_terms={**dict(info.get("reward_terms") or {}), **shape_terms},
                )
            )
            stats["physical_ok"] += 1
            valid_steps += 1
            cycle_ext += float(r_ext)
            cycle_soc_seq.append(intent_after.copy())
            cycle_act_seq.append(dict(hybrid))
            try:
                cycle_u_tp.append(u_tp_act)
                cycle_u_tp_h.append(u_tp_h)
            except NameError:
                pass
            steps_in_cycle += 1
            goal = goal_next
            obs = next_obs

            # 周期结束：写高层转移（SMDP 外在奖励；prior_only 时跳过 hi buffer）
            cycle_done = steps_in_cycle >= c or done_flag
            if cycle_done:
                _hl = str(cfg.get("high_level_mode", "learned")).lower()
                _store_hi = _hl not in ("prior_only", "prior", "stfr_a", "stfr")
                if _store_hi:
                    if bool(cfg.get("high_reward_normalize", True)) and steps_in_cycle > 0:
                        hi_r = float(cycle_ext) / float(steps_in_cycle)
                    else:
                        hi_r = float(cycle_ext)
                    # λ-SoC 高层：仅在外在回报上扣库存对偶项（不改底层残差）
                    if bool(getattr(agent, "high_lambda_soc", False)) or bool(
                        cfg.get("high_lambda_soc", False)
                    ):
                        try:
                            lam = float(getattr(agent, "lambda_soc", 0.0) or 0.0)
                            if lam > 0.0 and env.initial_soc is not None:
                                soc_now = extract_soc_from_obs(np.asarray(obs, dtype=np.float32), 2)
                                soc_init = extract_soc(env.initial_soc, DEFAULT_SOC_KEYS)
                                n = min(soc_now.size, soc_init.size, 2)
                                err = float(np.sum(np.abs(soc_now[:n] - soc_init[:n])))
                                hi_r = float(hi_r) - 0.15 * lam * err
                        except Exception:
                            pass
                    ach = None
                    if len(cycle_soc_seq) >= 2:
                        thr = 0.0
                        for ha in cycle_act_seq:
                            thr += abs(float(ha.get("u_battery", 0.0))) + abs(
                                float(ha.get("caes_magnitude", 0.0))
                            )
                        mu_tp = float(np.mean(cycle_u_tp)) if cycle_u_tp else u_tp_act
                        mu_h = float(np.mean(cycle_u_tp_h)) if cycle_u_tp_h else u_tp_h
                        ach = achieved_goal_from_cycle(
                            cycle_soc_seq[0],
                            cycle_soc_seq[-1],
                            mu_tp,
                            mu_h,
                            thr,
                            goal_low=agent.goal_low,
                            goal_high=agent.goal_high,
                        )
                    agent.hi_buffer.add(
                        HighTransition(
                            observation=np.asarray(cycle_start_obs, dtype=np.float32),
                            goal=cycle_goal.astype(np.float32),
                            reward_ext_sum=hi_r,
                            next_observation=np.asarray(obs, dtype=np.float32),
                            terminated=done_flag,
                            soc_seq=list(cycle_soc_seq),
                            action_seq=list(cycle_act_seq),
                            episode_id=int(episode),
                            cycle_idx=int(cycle_idx_ep),
                            achieved_delta=ach,
                        )
                    )
                cycle_idx_ep += 1
                steps_in_cycle = 0
                cycle_u_tp = []
                cycle_u_tp_h = []

            # 更新：residual 开训前只训高层；之后 lo+hi
            metrics: dict[str, float] = {}
            in_phase_a = valid_steps < phase_a_steps
            # goal_conditioned：底层始终更新（Hybrid 移植后小 lr 微调）
            # blend：residual_train_start 之后才更新 lo
            train_lo = (
                str(cfg.get("execution_mode", "action_residual")).lower() != "blend"
                or valid_steps >= residual_train_start
            )
            if train_lo and len(agent.lo_buffer) >= learn_lo:
                for _ in range(n_grad_lo):
                    metrics.update(agent.update_low(min(bs_lo, len(agent.lo_buffer))))
            # STFR Stage A / prior_only：不更新高层（慢层为解析 prior）
            _hl_mode = str(cfg.get("high_level_mode", "learned")).lower()
            _train_high = _hl_mode not in ("prior_only", "prior", "stfr_a", "stfr")
            if (
                _train_high
                and not in_phase_a
                and len(agent.hi_buffer) >= learn_hi
                and (valid_steps % c == 0)
                and n_grad_hi > 0
            ):
                hi_grads = n_grad_hi
                for _ in range(hi_grads):
                    metrics.update(agent.update_high(min(bs_hi, len(agent.hi_buffer))))

            if valid_steps % log_every == 0 or valid_steps == total_valid_steps:
                display_step = int(log_resume_base) + int(valid_steps)
                entry = {
                    "valid_step": int(display_step),
                    "r_ext": float(r_ext),
                    "r_int": float(r_int),
                    "goal": goal.tolist(),
                    "eps": eps,
                    "prior_w": scheduled_prior_weight(progress, recovery=False),
                    "alpha": alpha,
                    "phase_a": bool(in_phase_a),
                    **metrics,
                }
                if step_log and int(step_log[-1].get("valid_step", -1)) == int(display_step):
                    step_log[-1] = entry
                else:
                    step_log.append(entry)
                # 可观测进度：打印 + 落盘（训练中也可判断是否在跑）
                try:
                    prog = {
                        "valid_steps": int(valid_steps),
                        "total_steps": int(total_valid_steps),
                        "frac": float(valid_steps) / max(float(total_valid_steps), 1.0),
                        "episode": int(episode),
                        "r_ext": float(r_ext),
                        "r_int": float(r_int),
                        "eps": float(eps),
                        "tea_progress": float(progress),
                        "lambda_soc": float(getattr(agent, "lambda_soc", 0.0)),
                    }
                    (run_dir / "train" / "progress.json").write_text(
                        json.dumps(prog, ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    print(
                        f"[progress] step={valid_steps}/{total_valid_steps} "
                        f"({100.0 * prog['frac']:.1f}%) ep={episode} "
                        f"r_ext={r_ext:.3f} r_int={r_int:.3f}",
                        flush=True,
                    )
                except Exception as _exc:
                    print(f"[progress-warn] {_exc}", flush=True)
            if done_flag:
                # LTAR / high_lambda_soc：终端库存成本 → 更新 Lagrangian λ
                if bool(getattr(agent, "ltar_enabled", False)) or bool(
                    getattr(agent, "high_lambda_soc", False)
                ) or bool(cfg.get("high_lambda_soc", False)):
                    try:
                        terms = dict(info.get("reward_terms") or {})
                        # 优先用环境给出的 L1 误差；失败则 1.0 惩罚
                        l1 = terms.get("terminal_soc_l1_error")
                        if l1 is None:
                            l1 = terms.get("terminal_soc_l1_energy")
                        tol = float(
                            terms.get("terminal_soc_tolerance")
                            or 0.06
                        )
                        satisfied = bool(
                            info.get("terminal_soc_satisfied")
                            or terms.get("terminal_soc_satisfied")
                            or False
                        )
                        if l1 is not None:
                            cost = max(0.0, float(l1) - tol)
                        else:
                            cost = 0.0 if satisfied else 1.0
                        if not satisfied:
                            cost = max(cost, 0.5)
                        lam = agent.update_lambda_soc(cost)
                        stats["lambda_soc"] = float(lam)
                        stats["ltar_ep_cost"] = float(cost)
                    except Exception:
                        pass
                episode += 1
                obs, _ = reset_ep(episode)
                steps_in_cycle = 0
                goal = np.zeros(agent.goal_dim, dtype=np.float32)

        result["status"] = "completed"
        result["valid_steps"] = valid_steps
        result["episodes"] = episode
        result["stats"] = stats
        result["last_metrics"] = agent.last_metrics
        agent.save(run_dir / "checkpoints" / "ghtd3.pt")

        # 评估：基线规则 + 峰谷规则 + Safe Market-GHTD3
        rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_rule")
        rule_res = evaluate_policy(rule_env, RuleBasedController(rule_env), run_dir / "trajectories" / "rule.csv")
        rule_env.close()

        price_rule_env = PowerSystemEnv(run_id=f"{run_dir.name}_price_rule")
        price_rule_res = evaluate_policy(
            price_rule_env,
            PriceAwareRuleController(price_rule_env),
            run_dir / "trajectories" / "price_rule.csv",
        )
        price_rule_env.close()

        eval_env = PowerSystemEnv(run_id=f"{run_dir.name}_eval")
        eval_ctrl = GiveSafeController(oracle=eval_env.oracle, shadow=None, config=gs_cfg)
        policy = GHTD3PolicyWrapper(agent, eval_env, eval_ctrl, cfg)
        eval_res = evaluate_policy(eval_env, policy, run_dir / "trajectories" / "eval.csv")
        eval_env.close()

        result["eval"] = eval_res
        result["rule"] = rule_res
        result["price_rule"] = price_rule_res
        result["innovations"] = {
            "principle_pack": "Safe Market-GHTD3",
            "MSGP": bool(cfg.get("market_goal_prior", True)),
            "MS_HER": str(cfg.get("goal_relabel_mode", "")).lower() in ("ms_her", "her_mix")
            or bool(cfg.get("ms_her_weighting", False)),
            "F_MLE": bool(cfg.get("f_mle_pretrain", False))
            or (
                str(cfg.get("execution_mode", "")).lower() == "goal_conditioned"
                and not bool(cfg.get("hybrid_anchor", False))
            ),
            "GiveSafe": True,
            "IDD": "scripts/diagnose_ghtd3_goal_sensitivity.py + eval_idd_decoupling.py",
            "high_lambda_soc": bool(cfg.get("high_lambda_soc", False)),
            "tariff_aligned_c": bool(cfg.get("tariff_aligned_c", False)),
            "ltar_enabled": bool(cfg.get("ltar_enabled", False)),
            "lambda_soc_final": float(getattr(agent, "lambda_soc", 0.0)),
            "stfr_enabled": bool(cfg.get("stfr_enabled", False)),
            "high_level_mode": str(cfg.get("high_level_mode", "learned")),
            "trust_region_residual": str(cfg.get("execution_mode", "")).lower()
            in ("action_residual", "tea", "ltar")
            and not bool(cfg.get("tea_expandable", False)),
            "mode_factorized_teacher_lock": not bool(cfg.get("residual_mode_override", False)),
            "smdp_gamma_c": True,
            "modelica_goal_dim": int(agent.goal_dim),
            "hybrid_anchor": bool(agent.hybrid_anchor_enabled),
            "execution_mode": str(cfg.get("execution_mode", "")),
            "market_goal_prior": bool(cfg.get("market_goal_prior", True)),
            "market_prior_annealing": True,
            "goal_relabel_mode": str(cfg.get("goal_relabel_mode", "her_mix")),
            "phase_a_steps": phase_a_steps,
            "residual_train_start": residual_train_start,
            "intrinsic_alpha_schedule": [alpha0, alpha1],
            "recovery_goal_horizon_steps": int(cfg.get("recovery_goal_horizon_steps", 36) or 0),
            "price_aware_bootstrap": bool(cfg.get("price_aware_bootstrap", True)),
            "high_reward_normalize": bool(cfg.get("high_reward_normalize", True)),
            "hierarchical_bc_pretrain": bool(cfg.get("bc_pretrain", True)),
            "givesafe_low_level": True,
            "huber_q_clip_critics": True,
            "full_step_log": True,
            "obs_norm": bool(cfg.get("obs_norm", True)),
        }
        if annual_evaluation:
            from training.evaluate_td3 import evaluate_annual_policy

            ann_env = PowerSystemEnv(run_id=f"{run_dir.name}_annual")
            ann_ctrl = GiveSafeController(oracle=ann_env.oracle, shadow=None, config=gs_cfg)
            ann_pol = GHTD3PolicyWrapper(agent, ann_env, ann_ctrl, cfg)
            result["annual_eval"] = evaluate_annual_policy(
                ann_env,
                ann_pol,
                annual_horizon_hours=int(ann_env.config["fmu"]["annual_horizon_hours"]),
                output_dir=run_dir / "trajectories" / "annual_eval",
            )
            ann_env.close()
    finally:
        env.close()

    (run_dir / "train").mkdir(parents=True, exist_ok=True)
    (run_dir / "train" / "step_log.json").write_text(
        json.dumps(step_log, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (run_dir / "summary.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2, default=str), encoding="utf-8"
    )
    return result


def run_smoke(**kwargs) -> dict[str, Any]:
    return run_ghtd3_training(
        total_valid_steps=kwargs.pop("total_valid_steps", 3000),
        run_dir=kwargs.pop("run_dir", "runs/ghtd3_smoke"),
        **kwargs,
    )


def run_short(**kwargs) -> dict[str, Any]:
    return run_ghtd3_training(
        total_valid_steps=kwargs.pop("total_valid_steps", 20000),
        run_dir=kwargs.pop("run_dir", "runs/ghtd3_short"),
        **kwargs,
    )
