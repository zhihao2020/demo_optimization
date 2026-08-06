#!/usr/bin/env python
"""GHTD3 结构诊断：goal 是否真正驱动底层动作。

检查项
------
1. Hybrid 移植后 g=0 是否复现 Hybrid 动作
2. 固定 s，单维扫 g：动作响应幅度 / 单调性 / mode 切换
3. encoder 第一层 goal 列权重范数（是否仍≈0）
4. 旧 blend 路径下 α 封顶后动作被 Hybrid 主导的程度
5. （可选）加载已训 ckpt 对比训练后 goal 敏感度

用法
----
  PYTHONPATH=src python scripts/diagnose_ghtd3_goal_sensitivity.py
  PYTHONPATH=src python scripts/diagnose_ghtd3_goal_sensitivity.py \\
      --ckpt runs/ghtd3_modelica_ft_30k/checkpoints/ghtd3.pt
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from config.paths import apply_process_cache_env  # noqa: E402

apply_process_cache_env()

from envs.power_system_env import PowerSystemEnv  # noqa: E402
from training.ghtd3.agent import GHTD3Agent  # noqa: E402
from training.ghtd3.goals import GOAL_NAMES, residual_scale_from_goal  # noqa: E402
from training.ghtd3.hybrid_anchor import HybridAnchor  # noqa: E402


def _load_cfg(path: str | Path | None = None) -> dict[str, Any]:
    p = Path(path) if path else ROOT / "src/config/ghtd3_config.yaml"
    if not p.is_file():
        p = ROOT / p
    with open(p, encoding="utf-8") as f:
        full = yaml.safe_load(f) or {}
    return dict(full.get("ghtd3") or full)


def _action_vec(a: dict) -> np.ndarray:
    return np.asarray(
        [
            float(np.asarray(a["u_tp"]).reshape(-1)[0]),
            float(np.asarray(a["u_battery"]).reshape(-1)[0]),
            float(int(np.asarray(a["caes_mode"]).reshape(-1)[0])),
            float(np.asarray(a["caes_magnitude"]).reshape(-1)[0]),
        ],
        dtype=np.float64,
    )


def _l1(a: dict, b: dict) -> float:
    return float(np.sum(np.abs(_action_vec(a) - _action_vec(b))))


def _goal_col_stats(lo_actor: torch.nn.Module, obs_dim: int, goal_dim: int) -> dict[str, Any]:
    w = lo_actor.encoder[0].weight.detach().cpu().numpy()  # [H, obs+goal]
    obs_part = w[:, :obs_dim]
    goal_part = w[:, obs_dim : obs_dim + goal_dim]
    per_dim = []
    for i in range(goal_dim):
        col = goal_part[:, i]
        per_dim.append(
            {
                "name": GOAL_NAMES[i] if i < len(GOAL_NAMES) else f"g{i}",
                "l2": float(np.linalg.norm(col)),
                "abs_max": float(np.max(np.abs(col))),
                "abs_mean": float(np.mean(np.abs(col))),
            }
        )
    return {
        "obs_cols_l2_mean": float(np.mean(np.linalg.norm(obs_part, axis=0))),
        "goal_cols_l2_mean": float(np.mean(np.linalg.norm(goal_part, axis=0))),
        "goal_over_obs_ratio": float(
            np.mean(np.linalg.norm(goal_part, axis=0))
            / max(np.mean(np.linalg.norm(obs_part, axis=0)), 1e-12)
        ),
        "per_goal_dim": per_dim,
    }


def _forward_pack(
    agent: GHTD3Agent,
    obs: np.ndarray,
    goal: np.ndarray,
    *,
    u_tp_low: float,
    u_tp_high: float,
    u_bat_low: float,
    u_bat_high: float,
    mode_mask: np.ndarray,
) -> dict[str, Any]:
    """动作 + pre-squash logit，便于判断饱和。"""
    device = agent.device
    o = torch.as_tensor(obs, dtype=torch.float32, device=device).view(1, -1)
    if agent.low_use_raw_obs:
        pass
    else:
        o = agent._prep_obs(o)
    g = torch.as_tensor(goal, dtype=torch.float32, device=device).view(1, -1)
    if g.shape[-1] != agent.goal_dim:
        gg = torch.zeros(1, agent.goal_dim, device=device)
        n = min(agent.goal_dim, g.shape[-1])
        gg[:, :n] = g[:, :n]
        g = gg
    mask = torch.as_tensor(mode_mask, dtype=torch.bool, device=device).view(1, 3)
    with torch.no_grad():
        logits = agent.lo_actor.forward_logits(o, g)
        out = agent.lo_actor.act(
            o,
            g,
            torch.tensor([u_tp_low], device=device),
            torch.tensor([u_tp_high], device=device),
            torch.tensor([u_bat_low], device=device),
            torch.tensor([u_bat_high], device=device),
            mask,
            deterministic=True,
            explore_noise_std=0.0,
        )
    return {
        "u_tp": float(out["u_tp"][0].cpu()),
        "u_battery": float(out["u_battery"][0].cpu()),
        "caes_mode": int(out["caes_mode"][0].cpu()),
        "caes_magnitude": float(out["caes_magnitude"][0].cpu()),
        "z_tp": float(logits["z_tp"][0].cpu()),
        "z_bat": float(logits["z_bat"][0].cpu()),
        "z_discharge": float(logits["z_discharge"][0].cpu()),
        "z_charge": float(logits["z_charge"][0].cpu()),
        "logits_mode": [float(x) for x in logits["logits_mode"][0].cpu().tolist()],
    }


def _sweep_dim(
    agent: GHTD3Agent,
    obs: np.ndarray,
    feasible,
    dim: int,
    n: int = 11,
    base: np.ndarray | None = None,
    *,
    loose_bounds: bool = False,
) -> dict[str, Any]:
    low, high = float(agent.goal_low[dim]), float(agent.goal_high[dim])
    grid = np.linspace(low, high, n, dtype=np.float32)
    if base is None:
        base = np.zeros(agent.goal_dim, dtype=np.float32)
    if loose_bounds:
        tp_lo, tp_hi = 1.0 / 3.0, 1.0
        bat_lo, bat_hi = -1.0, 1.0
        mask = np.ones(3, dtype=bool)
    else:
        tp_lo, tp_hi = float(feasible.u_tp_low), float(feasible.u_tp_high)
        bat_lo, bat_hi = float(feasible.u_battery_low), float(feasible.u_battery_high)
        mask = feasible.mode_mask.as_bool_array()

    packs = []
    for v in grid:
        g = base.copy()
        g[dim] = v
        packs.append(
            _forward_pack(
                agent,
                obs,
                g,
                u_tp_low=tp_lo,
                u_tp_high=tp_hi,
                u_bat_low=bat_lo,
                u_bat_high=bat_hi,
                mode_mask=mask,
            )
        )
    A = np.asarray(
        [[p["u_tp"], p["u_battery"], p["caes_mode"], p["caes_magnitude"]] for p in packs],
        dtype=np.float64,
    )
    Z = np.asarray(
        [[p["z_tp"], p["z_bat"], p["z_discharge"], p["z_charge"]] for p in packs],
        dtype=np.float64,
    )
    mid = A[n // 2]
    span = A.max(axis=0) - A.min(axis=0)
    zspan = Z.max(axis=0) - Z.min(axis=0)
    mono = {}
    for j, name in enumerate(["u_tp", "u_battery", "mode", "mag"]):
        d = np.diff(A[:, j])
        if np.allclose(d, 0):
            mono[name] = {"flat": True, "agree_pos": 0.0, "agree_neg": 0.0}
        else:
            mono[name] = {
                "flat": False,
                "agree_pos": float(np.mean(d > 0)),
                "agree_neg": float(np.mean(d < 0)),
                "range": float(span[j]),
            }
    # 饱和：logit 在动但 sigmoid 后动作不动
    saturated = bool(float(zspan.max()) > 0.5 and float(span[:2].sum() + span[3]) < 1e-4)
    return {
        "dim": dim,
        "name": GOAL_NAMES[dim] if dim < len(GOAL_NAMES) else f"g{dim}",
        "loose_bounds": loose_bounds,
        "grid": grid.tolist(),
        "u_tp": A[:, 0].tolist(),
        "u_battery": A[:, 1].tolist(),
        "caes_mode": A[:, 2].tolist(),
        "caes_magnitude": A[:, 3].tolist(),
        "z_tp": Z[:, 0].tolist(),
        "z_bat": Z[:, 1].tolist(),
        "span": {
            "u_tp": float(span[0]),
            "u_battery": float(span[1]),
            "mode": float(span[2]),
            "mag": float(span[3]),
            "l1_total": float(span.sum()),
        },
        "logit_span": {
            "z_tp": float(zspan[0]),
            "z_bat": float(zspan[1]),
            "z_discharge": float(zspan[2]),
            "z_charge": float(zspan[3]),
            "max": float(zspan.max()),
        },
        "saturated_head": saturated,
        "max_l1_from_mid": float(np.max(np.sum(np.abs(A - mid), axis=1))),
        "monotonicity": mono,
        "mode_unique": int(len(np.unique(A[:, 2].astype(int)))),
        "z_tp_abs_mean": float(np.mean(np.abs(Z[:, 0]))),
        "z_bat_abs_mean": float(np.mean(np.abs(Z[:, 1]))),
    }


def _blend_vs_gc(
    agent: GHTD3Agent,
    obs: np.ndarray,
    feasible,
    goals: list[np.ndarray],
    alpha_max: float = 0.28,
) -> dict[str, Any]:
    """对比同一 g 下 goal_conditioned vs blend 的动作差。"""
    rows = []
    saved = agent.execution_mode
    for g in goals:
        agent.execution_mode = "goal_conditioned"
        a_gc = agent.select_composed_action(obs, g, feasible, deterministic=True)
        agent.execution_mode = "blend"
        alpha = residual_scale_from_goal(g, alpha0=0.0, alpha_max=alpha_max)
        a_bl = agent.select_composed_action(obs, g, feasible, deterministic=True, residual_scale=alpha)
        a_h = agent._hybrid_anchor.act_scalars(obs, feasible, deterministic=True) if agent._hybrid_anchor else None
        row = {
            "goal": g.tolist(),
            "alpha_blend": float(alpha),
            "gc": _action_vec(a_gc).tolist(),
            "blend": _action_vec(a_bl).tolist(),
            "l1_gc_vs_blend": _l1(a_gc, a_bl),
        }
        if a_h is not None:
            a_h_d = {
                "u_tp": a_h["u_tp"],
                "u_battery": a_h["u_battery"],
                "caes_mode": a_h["caes_mode"],
                "caes_magnitude": a_h["caes_magnitude"],
            }
            row["hybrid"] = _action_vec(a_h_d).tolist()
            row["l1_blend_vs_hybrid"] = _l1(a_bl, a_h_d)
            row["l1_gc_vs_hybrid"] = _l1(a_gc, a_h_d)
            # blend 相对 Hybrid 的位移 / GC 相对 Hybrid 的位移
            row["blend_captures_gc_frac"] = float(
                row["l1_blend_vs_hybrid"] / max(row["l1_gc_vs_hybrid"], 1e-9)
            )
        rows.append(row)
    agent.execution_mode = saved
    return {"alpha_max": alpha_max, "rows": rows}


def _infer_goal_dim(ckpt: Path, obs_dim: int) -> int | None:
    try:
        data = torch.load(ckpt, map_location="cpu", weights_only=False)
        w0 = data["lo_actor"]["encoder.0.weight"]
        return int(w0.shape[1] - obs_dim)
    except Exception:
        return None


def _build_agent(
    obs_dim: int,
    cfg: dict[str, Any],
    *,
    transplant: bool,
    ckpt: Path | None,
) -> tuple[GHTD3Agent, dict[str, Any]]:
    local_cfg = dict(cfg)
    if ckpt is not None and ckpt.is_file():
        gd = _infer_goal_dim(ckpt, obs_dim)
        if gd is not None and gd > 0:
            local_cfg["goal_dim"] = gd
    agent = GHTD3Agent(obs_dim, local_cfg)
    meta: dict[str, Any] = {"transplant": False, "ckpt": None, "goal_dim": agent.goal_dim}
    hp = Path(cfg.get("hybrid_anchor_path") or "")
    if not hp.is_file():
        hp = ROOT / str(cfg.get("hybrid_anchor_path") or "")
    mode = str(local_cfg.get("execution_mode", "action_residual")).lower()
    use_anchor = bool(local_cfg.get("hybrid_anchor", False)) and hp.is_file()
    if use_anchor:
        anchor = HybridAnchor(obs_dim, hp, device=str(agent.device))
        # action_residual：永不移植；goal_conditioned 仅 fresh+transplant 标志时移植
        do_tx = bool(transplant) and mode not in ("action_residual", "tea") and ckpt is None
        meta = agent.attach_hybrid_anchor(anchor, transplant=do_tx)
        meta["hybrid_path"] = str(hp)
        meta["execution_mode"] = mode
    else:
        meta["execution_mode"] = mode
        meta["hybrid_path"] = None
    if ckpt is not None and ckpt.is_file():
        agent.load(ckpt, strict=False)
        agent.execution_mode = mode
        if mode in ("action_residual", "tea", "goal_conditioned"):
            agent.low_use_raw_obs = bool(local_cfg.get("low_use_raw_obs", False))
        else:
            agent.low_use_raw_obs = bool(local_cfg.get("low_use_raw_obs", True))
        if use_anchor and agent._hybrid_anchor is None and hp.is_file():
            agent._hybrid_anchor = HybridAnchor(obs_dim, hp, device=str(agent.device))
            agent.hybrid_anchor_enabled = True
        meta["ckpt"] = str(ckpt)
        meta["lo_it"] = agent.lo_it
        meta["hi_it"] = agent.hi_it
    return agent, meta


def diagnose_one(
    label: str,
    agent: GHTD3Agent,
    obs: np.ndarray,
    feasible,
    obs_dim: int,
    *,
    n_grid: int = 11,
) -> dict[str, Any]:
    g0 = np.zeros(agent.goal_dim, dtype=np.float32)
    # 诊断一律走 select_composed_action（含 action_residual）
    a0 = agent.select_composed_action(obs, g0, feasible, deterministic=True)
    hybrid_match = None
    if agent._hybrid_anchor is not None:
        a_h = agent._hybrid_anchor.act_scalars(obs, feasible, deterministic=True)
        a_h_d = {
            "u_tp": a_h["u_tp"],
            "u_battery": a_h["u_battery"],
            "caes_mode": a_h["caes_mode"],
            "caes_magnitude": a_h["caes_magnitude"],
        }
        hybrid_match = {
            "g0": _action_vec(a0).tolist(),
            "hybrid": _action_vec(a_h_d).tolist(),
            "l1": _l1(a0, a_h_d),
            "u_tp_diff": float(np.asarray(a0["u_tp"]).ravel()[0] - a_h["u_tp"]),
            "u_bat_diff": float(np.asarray(a0["u_battery"]).ravel()[0] - a_h["u_battery"]),
            "mode_same": int(a0["caes_mode"]) == int(a_h["caes_mode"]),
            "mag_diff": float(np.asarray(a0["caes_magnitude"]).ravel()[0] - a_h["caes_magnitude"]),
            "match_ok": _l1(a0, a_h_d) < 1e-3,
        }

    def _sweep_composed(dim: int, loose: bool = False) -> dict[str, Any]:
        low, high = float(agent.goal_low[dim]), float(agent.goal_high[dim])
        grid = np.linspace(low, high, n_grid, dtype=np.float32)
        actions = []
        for v in grid:
            g = g0.copy()
            g[dim] = v
            a = agent.select_composed_action(obs, g, feasible, deterministic=True)
            actions.append(_action_vec(a))
        A = np.stack(actions)
        span = A.max(axis=0) - A.min(axis=0)
        # residual 路径下再测 logits（norm obs）
        zspan_max = 0.0
        z_abs = 0.0
        if agent.execution_mode != "action_residual":
            pack = _sweep_dim(agent, obs, feasible, dim, n=n_grid, base=g0, loose_bounds=loose)
            return {
                **pack,
                "span": {
                    "u_tp": float(span[0]),
                    "u_battery": float(span[1]),
                    "mode": float(span[2]),
                    "mag": float(span[3]),
                    "l1_total": float(span.sum()),
                },
            }
        for v in grid:
            g = g0.copy()
            g[dim] = v
            with torch.no_grad():
                o = torch.as_tensor(obs, dtype=torch.float32, device=agent.device).view(1, -1)
                o = agent._prep_obs_low(o)
                gg = torch.as_tensor(g, dtype=torch.float32, device=agent.device).view(1, -1)
                logits = agent.lo_actor.forward_logits(o, gg)
                zspan_max = max(
                    zspan_max,
                    abs(float(logits["z_tp"][0])),
                    abs(float(logits["z_bat"][0])),
                )
                z_abs = max(z_abs, abs(float(logits["z_tp"][0])))
        return {
            "dim": dim,
            "name": GOAL_NAMES[dim] if dim < len(GOAL_NAMES) else f"g{dim}",
            "loose_bounds": loose,
            "grid": grid.tolist(),
            "u_tp": A[:, 0].tolist(),
            "u_battery": A[:, 1].tolist(),
            "caes_mode": A[:, 2].tolist(),
            "caes_magnitude": A[:, 3].tolist(),
            "span": {
                "u_tp": float(span[0]),
                "u_battery": float(span[1]),
                "mode": float(span[2]),
                "mag": float(span[3]),
                "l1_total": float(span.sum()),
            },
            "logit_span": {"max": float(zspan_max), "z_tp": float(zspan_max), "z_bat": 0.0, "z_discharge": 0.0, "z_charge": 0.0},
            "saturated_head": False,
            "mode_unique": int(len(np.unique(A[:, 2].astype(int)))),
            "z_tp_abs_mean": float(z_abs),
            "z_bat_abs_mean": 0.0,
        }

    sweeps = []
    sweeps_loose = []
    for d in range(agent.goal_dim):
        sweeps.append(_sweep_composed(d, loose=False))
        sweeps_loose.append(_sweep_composed(d, loose=True))

    # 联合极端 goal
    g_extreme = []
    for sign in (-1.0, 1.0):
        g = np.zeros(agent.goal_dim, dtype=np.float32)
        for d in range(min(4, agent.goal_dim)):  # 前 4 维带符号；arb 单独
            mid = 0.5 * (agent.goal_low[d] + agent.goal_high[d])
            half = 0.5 * (agent.goal_high[d] - agent.goal_low[d])
            g[d] = float(mid + sign * half)
        if agent.goal_dim > 4:
            g[4] = agent.goal_high[4] if sign > 0 else agent.goal_low[4]
        g_extreme.append(g)
    g_extreme.append(np.asarray(agent.goal_high, dtype=np.float32))
    g_extreme.append(np.asarray(agent.goal_low, dtype=np.float32))

    blend_cmp = _blend_vs_gc(agent, obs, feasible, g_extreme, alpha_max=0.28)

    # 汇总敏感度（动作级 + logit 级）
    sens = {}
    for s, sl in zip(sweeps, sweeps_loose):
        sens[s["name"]] = {
            "span_l1": s["span"]["l1_total"],
            "span_u_tp": s["span"]["u_tp"],
            "span_u_battery": s["span"]["u_battery"],
            "span_mag": s["span"]["mag"],
            "logit_span_max": s["logit_span"]["max"],
            "logit_span_loose_max": sl["logit_span"]["max"],
            "span_l1_loose": sl["span"]["l1_total"],
            "mode_changes": s["mode_unique"] > 1,
            # action_residual：动作 O(β)~0.1 即有效；绝对头路径仍用原阈值
            "alive_action": s["span"]["l1_total"] > 1e-3 or sl["span"]["l1_total"] > 1e-3,
            "alive_logit": s["logit_span"]["max"] > 0.05 or sl["logit_span"]["max"] > 0.05,
            "saturated": bool(s["saturated_head"] or sl["saturated_head"]),
            "z_tp_abs_mean": s["z_tp_abs_mean"],
            "z_bat_abs_mean": s["z_bat_abs_mean"],
        }
    n_alive = sum(1 for v in sens.values() if v["alive_action"])
    n_logit = sum(1 for v in sens.values() if v["alive_logit"])
    n_sat = sum(1 for v in sens.values() if v["saturated"])

    # 诊断 verdict
    issues = []
    if hybrid_match is not None and not hybrid_match["match_ok"]:
        issues.append(f"g=0 未复现 Hybrid (L1={hybrid_match['l1']:.4g})")
    wstats = _goal_col_stats(agent.lo_actor, obs_dim, agent.goal_dim)
    if wstats["goal_cols_l2_mean"] < 1e-6:
        issues.append("encoder goal 列权重≈0：π(s,g) 对任意 g 恒等于 π(s,0)")
    if n_alive == 0 and n_logit == 0:
        issues.append("全部 goal 维：logit 与动作均无响应（死通路）")
    elif n_alive == 0 and n_logit > 0:
        issues.append(
            f"logit 有响应({n_logit}维)但动作全饱和(saturated={n_sat})：sigmoid 头卡死，分层意图传不到 a"
        )
    elif n_alive < max(1, agent.goal_dim // 2):
        issues.append(f"仅 {n_alive}/{agent.goal_dim} 维 goal 有动作响应")
    if agent.goal_dim > 4 and not sens.get("arb", {}).get("alive_action", False):
        issues.append("arb 维在 goal_conditioned 下无动作响应（死维/或仅进 blend α）")
    # blend 截断
    if agent.execution_mode == "blend":
        fracs = [r.get("blend_captures_gc_frac", 0.0) for r in blend_cmp["rows"] if "blend_captures_gc_frac" in r]
        if fracs:
            mean_frac = float(np.mean(fracs))
            if mean_frac < 0.5 and n_alive > 0:
                issues.append(
                    f"blend 仅捕获 GC 相对 Hybrid 位移的 {mean_frac:.0%}（α≤0.28 架空高层）"
                )

    if not issues:
        verdict = "OK：goal 通路存活，g=0≈Hybrid"
    elif wstats["goal_cols_l2_mean"] < 1e-6 or (n_alive == 0 and n_logit == 0):
        verdict = "CRITICAL：goal 通路未激活（分层无效）"
    elif n_alive == 0 and n_logit > 0:
        verdict = "CRITICAL：goal→logit 活着但动作头饱和（分层意图到不了执行）"
    else:
        verdict = "WARN：" + "；".join(issues[:3])

    return {
        "label": label,
        "execution_mode": agent.execution_mode,
        "low_use_raw_obs": agent.low_use_raw_obs,
        "goal_dim": agent.goal_dim,
        "goal_box": {"low": agent.goal_low.tolist(), "high": agent.goal_high.tolist()},
        "hybrid_match_g0": hybrid_match,
        "goal_weight_stats": wstats,
        "per_dim_sensitivity": sens,
        "n_dims_alive_action": n_alive,
        "n_dims_alive_logit": n_logit,
        "n_dims_saturated": n_sat,
        "sweeps": sweeps,
        "sweeps_loose": sweeps_loose,
        "blend_vs_gc": blend_cmp,
        "issues": issues,
        "verdict": verdict,
    }


def _print_report(rep: dict[str, Any]) -> None:
    print("\n" + "=" * 72)
    print(f"[{rep['label']}]  {rep['verdict']}")
    print("=" * 72)
    print(f"  mode={rep['execution_mode']}  raw_obs_low={rep['low_use_raw_obs']}")
    hm = rep.get("hybrid_match_g0")
    if hm:
        print(
            f"  g=0 vs Hybrid: L1={hm['l1']:.6g}  mode_same={hm['mode_same']}  "
            f"Δu_tp={hm['u_tp_diff']:.4g}  Δu_bat={hm['u_bat_diff']:.4g}  "
            f"match_ok={hm['match_ok']}"
        )
    ws = rep["goal_weight_stats"]
    print(
        f"  encoder: obs_col_L2_mean={ws['obs_cols_l2_mean']:.4g}  "
        f"goal_col_L2_mean={ws['goal_cols_l2_mean']:.4g}  "
        f"goal/obs={ws['goal_over_obs_ratio']:.3g}"
    )
    for d in ws["per_goal_dim"]:
        print(f"    {d['name']:10s}  L2={d['l2']:.4g}  |max|={d['abs_max']:.4g}")
    gd = rep.get("goal_dim", 5)
    print(
        f"  alive action: {rep.get('n_dims_alive_action', 0)}/{gd}  "
        f"alive logit: {rep.get('n_dims_alive_logit', 0)}/{gd}  "
        f"saturated: {rep.get('n_dims_saturated', 0)}"
    )
    for name, s in rep["per_dim_sensitivity"].items():
        if s.get("alive_action"):
            flag = "ACT  "
        elif s.get("alive_logit"):
            flag = "LOGIT"
        else:
            flag = "DEAD "
        sat = " SAT" if s.get("saturated") else ""
        print(
            f"    {flag}{sat} {name:10s}  act_L1={s['span_l1']:.4g}  "
            f"loose_L1={s.get('span_l1_loose', 0):.4g}  "
            f"Δz_max={s.get('logit_span_max', 0):.4g}  "
            f"|z_tp|={s.get('z_tp_abs_mean', 0):.3g}  |z_bat|={s.get('z_bat_abs_mean', 0):.3g}"
        )
    rows = rep["blend_vs_gc"]["rows"]
    if rows:
        fracs = [r.get("blend_captures_gc_frac") for r in rows if r.get("blend_captures_gc_frac") is not None]
        l1b = [r.get("l1_blend_vs_hybrid", 0) for r in rows]
        l1g = [r.get("l1_gc_vs_hybrid", 0) for r in rows]
        print(
            f"  blend(α≤0.28) vs GC: mean |a_bl-a_H|={np.mean(l1b):.4g}  "
            f"mean |a_gc-a_H|={np.mean(l1g):.4g}  "
            f"capture_frac={np.mean(fracs) if fracs else float('nan'):.2%}"
        )
    for iss in rep.get("issues") or []:
        print(f"  ! {iss}")


def main() -> None:
    p = argparse.ArgumentParser(description="Diagnose GHTD3 goal→action pathway")
    p.add_argument(
        "--ckpt",
        action="append",
        default=[],
        help="可选 ghtd3.pt；可多次指定。默认额外扫 modelica_ft / her 若存在",
    )
    p.add_argument(
        "--config",
        type=str,
        default=None,
        help="ghtd3 yaml（默认 src/config/ghtd3_config.yaml；abs 用 ghtd3_config_abs.yaml）",
    )
    p.add_argument("--out", type=str, default="runs/diagnose_goal_sensitivity.json")
    p.add_argument("--n-grid", type=int, default=11)
    p.add_argument("--no-default-ckpts", action="store_true")
    p.add_argument(
        "--fresh-only",
        action="store_true",
        help="只诊断随机初始化（门禁 abs/GC 用，不扫旧 ckpt）",
    )
    args = p.parse_args()

    cfg = _load_cfg(args.config)
    mode = str(cfg.get("execution_mode", "action_residual")).lower()
    # 未指定 config 时保持旧 residual 默认；指定 abs 则尊重 yaml
    if args.config is None:
        cfg.setdefault("execution_mode", "action_residual")
        cfg["hybrid_init_low"] = False
        cfg["low_use_raw_obs"] = False
        cfg["residual_init"] = True
        mode = "action_residual"
    else:
        cfg.setdefault("low_use_raw_obs", False)
        if mode == "goal_conditioned":
            cfg.setdefault("residual_init", False)
            cfg.setdefault("hybrid_anchor", False)
            cfg.setdefault("goal_input_scale", 4.0)
            cfg.setdefault("obs_norm", True)

    print("[diag] building env for one obs ...")
    env = PowerSystemEnv(run_id="diag_goal_sens", forecast_enabled=True)
    obs, _ = env.reset(seed=0)
    obs = np.asarray(obs, dtype=np.float32)
    obs_dim = int(obs.shape[0])
    feasible = env.get_feasible_action_spec()
    print(f"[diag] obs_dim={obs_dim}  feasible u_tp=[{feasible.u_tp_low:.3g},{feasible.u_tp_high:.3g}]")
    print(f"[diag] execution_mode={cfg.get('execution_mode')} hybrid_anchor={cfg.get('hybrid_anchor')}")

    cases: list[tuple[str, bool, Path | None]] = [
        (f"fresh_{mode}", False, None),
    ]
    ckpts: list[Path] = [Path(c) for c in args.ckpt]
    if args.fresh_only or args.no_default_ckpts:
        ckpts = [Path(c) for c in args.ckpt if Path(c).is_file()]
    elif not args.no_default_ckpts:
        for rel in (
            "runs/ghtd3_gc_hybrid_35k/checkpoints/ghtd3.pt",
            "runs/ghtd3_modelica_ft_30k/checkpoints/ghtd3.pt",
            "runs/ghtd3_modelica_goal_40k/checkpoints/ghtd3.pt",
            "runs/ghtd3_her_anneal_50k/checkpoints/ghtd3.pt",
            "runs/ghtd3_anchor_smoke_2k/checkpoints/ghtd3.pt",
        ):
            path = ROOT / rel
            if path.is_file() and path not in ckpts:
                ckpts.append(path)
    for path in ckpts:
        if path.is_file():
            cases.append((f"ckpt:{path.parent.parent.name}", True, path))

    reports = []
    for label, transplant, ckpt in cases:
        print(f"\n[diag] case={label} ...")
        agent, meta = _build_agent(obs_dim, cfg, transplant=transplant, ckpt=ckpt)
        rep = diagnose_one(label, agent, obs, feasible, obs_dim, n_grid=args.n_grid)
        rep["meta"] = meta
        reports.append(rep)
        _print_report(rep)

    # 跨 case 对比表
    print("\n" + "=" * 72)
    print("SUMMARY")
    print("=" * 72)
    print(f"{'case':36s} {'act':>5s} {'logit':>5s} {'g0=H':>5s} {'goalL2':>8s}  verdict")
    for r in reports:
        hm = r.get("hybrid_match_g0") or {}
        g0ok = "Y" if hm.get("match_ok") else ("?" if not hm else "N")
        gl2 = r["goal_weight_stats"]["goal_cols_l2_mean"]
        gd = r.get("goal_dim", 5)
        print(
            f"{r['label'][:36]:36s} "
            f"{r.get('n_dims_alive_action', 0):>2d}/{gd} "
            f"{r.get('n_dims_alive_logit', 0):>2d}/{gd} "
            f"{g0ok:>5s} {gl2:>8.3g}  {r['verdict'][:52]}"
        )

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "obs_dim": obs_dim,
        "obs_preview": obs[:8].tolist(),
        "feasible": {
            "u_tp_low": float(feasible.u_tp_low),
            "u_tp_high": float(feasible.u_tp_high),
            "u_battery_low": float(feasible.u_battery_low),
            "u_battery_high": float(feasible.u_battery_high),
        },
        "reports": reports,
    }
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[diag] wrote {out}")


if __name__ == "__main__":
    main()
