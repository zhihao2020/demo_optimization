"""FS-HSAC training entry (independent from HybridSAC)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from envs.failures import FeasibleSetEmpty
from envs.power_system_env import PowerSystemEnv
from envs.reward_calculator import IncompleteRewardConfigError
from fmu import FmuAdapter
from replay.fs_hsac_replay import FSHSACReplayBuffer
from safety import GiveSafeController, ShadowFmuValidator, load_givesafe_config
from training.episode_starts import training_start_seconds
from training.fs_hsac.algorithm import FSHSAC
from training.fs_hsac.collector import FSHSACCollector
from training.fs_hsac.compute import configure_torch_compute
from training.fs_hsac.feasibility import FeasibilityTrainer, ResidualFeasibilityNet
from training.hybrid_common.eval_and_save import (
    finalize_training_run,
    prepare_run_dir,
    write_summary_and_report,
)
from training.hybrid_common.explore import (
    CUI_BATCH,
    CUI_LR,
    CUI_TAU,
    explore_epsilon,
    scaled_replay,
)
from training.hybrid_common.policy_wrapper import RandomFeasiblePolicy
from training.hybrid_td3.train import (
    _soft_shell_enabled,
    annual_episode_start_seconds,
    check_formal_gates,
    load_givesafe_gates,
)


def run_fs_hsac_training(
    total_valid_steps: int = 5000,
    run_dir: str | Path = "runs/fs_hsac_smoke",
    seed: int = 0,
    learning_starts: int = 256,
    batch_size: int = CUI_BATCH,
    formal: bool = False,
    enable_shadow: bool | None = None,
    forecast_enabled: bool | None = None,
    annual_evaluation: bool = False,
    resume_from: str | Path | None = None,
    use_feasibility_penalty: bool = True,
    feasibility_beta: float = 0.1,
    soft_shell: bool | None = None,
) -> dict[str, Any]:
    """Train FS-HSAC v2 on the FMU twin with split Bellman / feasibility replay.

    Paper mainline is support-only: ``use_feasibility_penalty=False`` or
    ``FS_HSAC_NO_FEAS=1`` (``scripts/train_seasonal.py --method fs_hsac --support``).
    Residual C_psi (this function's default True) is appendix-only.
    Soft shell stays off for the paper protocol.
    """
    import os

    if os.environ.get("FS_HSAC_NO_FEAS", "").strip() in ("1", "true", "True"):
        use_feasibility_penalty = False
    compute = configure_torch_compute()
    _ = _soft_shell_enabled(False if soft_shell is None else soft_shell)
    run_dir = Path(run_dir)
    root = Path(__file__).resolve().parents[3]
    prepare_run_dir(run_dir, root)

    gs_cfg = load_givesafe_config(root / "src/config/givesafe_config.yaml")
    if gs_cfg.get("givesafe", {}).get("use_fallback", False):
        raise RuntimeError("禁止 use_fallback")
    shadow_cfg = (gs_cfg.get("givesafe") or {}).get("shadow_validation") or {}
    use_shadow = bool(shadow_cfg.get("enabled", True)) if enable_shadow is None else bool(enable_shadow)

    np.random.seed(seed)
    try:
        env = PowerSystemEnv(
            require_complete_reward=formal, run_id=run_dir.name, forecast_enabled=forecast_enabled
        )
    except IncompleteRewardConfigError as exc:
        return {"status": "blocked_incomplete_reward", "error": str(exc)}

    gates_cfg = load_givesafe_gates(root)
    if formal and gates_cfg.get("formal_default_blocked", True):
        env.close()
        return {"status": "blocked_formal_gates", "blockers": ["formal_default_blocked=true"]}

    shadow = None
    if use_shadow:
        fmu_path = env.root / env.config["fmu"]["path"]
        step = float(env.config["fmu"]["communication_step_seconds"])
        registry = env.registry

        def factory():
            return FmuAdapter(fmu_path, step, registry)

        shadow = ShadowFmuValidator(
            factory=factory,
            oracle=env.oracle,
            enabled=True,
            mode=str(shadow_cfg.get("mode", "always")),
            near_boundary_fraction=float(shadow_cfg.get("near_boundary_fraction", 0.15)),
        )

    controller = GiveSafeController(oracle=env.oracle, shadow=shadow, config=gs_cfg)
    buffer = FSHSACReplayBuffer(capacity=scaled_replay(int(total_valid_steps)))
    collector = FSHSACCollector(buffer, controller)
    obs_dim = int(np.prod(env.observation_space.shape))
    agent = FSHSAC(
        obs_dim=obs_dim,
        gamma=float(env.reward_calculator.config.get("gamma", 0.99)),
        tau=CUI_TAU,
        actor_lr=CUI_LR,
        critic_lr=CUI_LR,
        alpha_lr=CUI_LR,
        use_feasibility_penalty=use_feasibility_penalty,
        feasibility_beta=feasibility_beta,
        device=str(compute["device"]),
    )
    feas_net = ResidualFeasibilityNet(obs_dim)
    feas_trainer = FeasibilityTrainer(feas_net, device=agent.device)
    agent.feasibility_net = feas_net

    resumed = None
    if resume_from is not None:
        ckpt = Path(resume_from)
        if not ckpt.is_file():
            env.close()
            if shadow is not None:
                shadow.close()
            raise FileNotFoundError(f"FS-HSAC resume checkpoint not found: {ckpt}")
        agent.load(ckpt)
        resumed = str(ckpt.resolve())
        learning_starts = min(int(learning_starts), 64)

    random_policy = RandomFeasiblePolicy(env)
    valid_steps = 0
    episode = 0
    caes_nonzero = 0
    step_log: list[dict] = []
    episode_start_times: list[float] = []
    progress_path = run_dir / "train" / "progress.json"

    def reset_training_episode(index: int):
        start_time = training_start_seconds(
            env.config["fmu"],
            env.episode_steps,
            index,
            annual_episode_start_seconds=annual_episode_start_seconds,
        )
        next_obs, reset_info = env.reset(seed=seed + index, options={"start_time": start_time})
        actual_start = float(reset_info.get("time", start_time) or start_time)
        episode_start_times.append(actual_start)
        collector.on_episode_reset(actual_start)
        return next_obs, reset_info

    obs, _info0 = reset_training_episode(episode)
    result: dict[str, Any] = {
        "algo": "fs_hsac_v2",
        "algorithm_version": "fs_hsac_v2",
        "use_feasibility_penalty": bool(use_feasibility_penalty),
        "feasibility_beta": float(feasibility_beta),
        "requested_valid_steps": total_valid_steps,
        "status": "running",
        "formal": formal,
        "givesafe": True,
        "use_fallback": False,
        "shadow_validation": shadow.capabilities() if shadow else {"enabled": False},
        "oracle_version": env.oracle.oracle_version,
        "forecast_enabled": env.forecast_enabled,
        "forecast_horizon_hours": env.forecast_provider.horizon_hours if env.forecast_provider else 0,
        "observation_dim": obs_dim,
        "resume_from": resumed,
        "agent_total_it_at_start": int(agent.total_it),
        "device": str(agent.device),
        "torch_threads": int(compute["threads"]),
        "shaping_audit": {
            "terminal_soc_shaping_mode": (env.reward_calculator.config.get("terminal_soc") or {})
            .get("shaping", {})
            .get("mode"),
            "absolute_coef": (env.reward_calculator.config.get("terminal_soc") or {})
            .get("shaping", {})
            .get("absolute_coef"),
            "weekend_soft_anchor": (env.reward_calculator.config.get("terminal_soc") or {})
            .get("shaping", {})
            .get("weekend_soft_anchor"),
            "note": "potential term is policy-invariant up to shaping; absolute/weekend are explicit preferences",
        },
    }

    try:
        pbar = tqdm(
            total=total_valid_steps,
            desc=f"FS-HSAC/{agent.device}",
            unit="step",
            dynamic_ncols=True,
            mininterval=2.0,
            miniters=20,
        )
        while valid_steps < total_valid_steps:
            try:
                feasible = env.get_feasible_action_spec()
            except FeasibleSetEmpty:
                episode += 1
                obs, _info0 = reset_training_episode(episode)
                continue

            def propose():
                if valid_steps < learning_starts:
                    return random_policy.predict(obs, feasible=feasible)
                remain = max(int(total_valid_steps) - int(learning_starts), 1)
                eps = explore_epsilon(valid_steps - learning_starts, remain)
                if np.random.rand() < eps:
                    return random_policy.predict(obs, feasible=feasible)
                return agent.select_action(obs, feasible, deterministic=False)

            obs, reward, terminated, truncated, info = collector.step_with_givesafe(
                env, propose, deterministic=False, feasible=feasible
            )
            if info.get("transition_type") == "physical" and info.get("transition_valid"):
                valid_steps += 1
                metrics: dict[str, float] = {}
                raw_u = info.get("requested_u_caes", 0.0)
                try:
                    u_caes = abs(float(np.asarray(raw_u).reshape(-1)[0]))
                except (TypeError, ValueError):
                    u_caes = 0.0
                if u_caes > 1e-6:
                    caes_nonzero += 1
                caes_frac = caes_nonzero / max(valid_steps, 1)
                if buffer.bellman_size >= learning_starts:
                    metrics = agent.update(buffer, batch_size=min(batch_size, len(buffer)))
                    if buffer.feasibility_size >= 64:
                        fmet = feas_trainer.update(buffer.sample_feasibility(min(batch_size, buffer.feasibility_size)))
                        metrics.update({f"feas_{k}": v for k, v in fmet.items()})
                        agent.use_feasibility_penalty = bool(
                            use_feasibility_penalty and feas_trainer.enabled
                        )
                if valid_steps % 500 == 0 or valid_steps == total_valid_steps:
                    entry = {
                        "valid_step": valid_steps,
                        "reward": reward,
                        "attempts": info.get("givesafe_attempt_count"),
                        "rejected": info.get("givesafe_rejected_attempts"),
                        "caes_nonzero": caes_nonzero,
                        "caes_frac": caes_frac,
                        **metrics,
                    }
                    step_log.append(entry)
                    progress_path.parent.mkdir(parents=True, exist_ok=True)
                    progress_path.write_text(
                        json.dumps(entry, indent=2, ensure_ascii=False, default=str),
                        encoding="utf-8",
                    )
                pbar.update(1)
                remain = max(int(total_valid_steps) - int(learning_starts), 1)
                eps_now = (
                    1.0
                    if valid_steps < learning_starts
                    else explore_epsilon(valid_steps - learning_starts, remain)
                )
                pbar.set_postfix(
                    ep=episode,
                    r=f"{float(reward):.3f}",
                    ad=f"{float(agent.last_metrics.get('alpha_d', 0.0)):.3f}",
                    ac=f"{float(agent.last_metrics.get('alpha_c', 0.0)):.3f}",
                    eps=f"{eps_now:.2f}",
                    caes=f"{caes_frac:.2f}",
                    refresh=False,
                )
            if terminated or truncated:
                episode += 1
                obs, _info0 = reset_training_episode(episode)
        else:
            result.update(status="completed", valid_steps=valid_steps)
        pbar.close()

        result = finalize_training_run(
            run_dir=run_dir,
            agent=agent,
            checkpoint_name="fs_hsac_v2.pt",
            gs_cfg=gs_cfg,
            use_shadow=use_shadow,
            forecast_enabled=forecast_enabled,
            annual_evaluation=annual_evaluation,
            result=result,
            step_log=step_log,
            collector_stats=collector.stats,
            soft_shell=False,
        )
        if formal:
            gate = check_formal_gates(result, gates_cfg)
            result["formal_gate"] = gate
        write_summary_and_report(run_dir, result)
        return result
    finally:
        env.close()
        if shadow is not None:
            shadow.close()
