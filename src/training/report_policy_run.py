"""从训练 run 目录生成中文可读策略报告（Markdown + 图）。"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _setup_matplotlib_font() -> None:
    """尽量选用系统中文字体，避免图中方框。"""
    import matplotlib

    matplotlib.rcParams["axes.unicode_minus"] = False
    for name in ("Microsoft YaHei", "SimHei", "PingFang SC", "Noto Sans CJK SC", "DejaVu Sans"):
        try:
            matplotlib.rcParams["font.sans-serif"] = [name]
            break
        except Exception:  # noqa: BLE001
            continue


def _load_summary(run_dir: Path) -> dict[str, Any]:
    path = run_dir / "summary.json"
    if not path.is_file():
        raise FileNotFoundError(f"缺少 summary.json: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _load_csv(path: Path):
    import pandas as pd

    if not path.is_file():
        return None
    return pd.read_csv(path)


def _cashflow_from_block(block: dict[str, Any] | None) -> float | None:
    if not block:
        return None
    if block.get("economic_cashflow_total") is not None:
        return float(block["economic_cashflow_total"])
    terms = block.get("cost_terms") or {}
    if terms.get("economic_cashflow_delta") is not None:
        return float(terms["economic_cashflow_delta"])
    return None


def _cashflow_from_csv(df) -> float | None:
    if df is None or df.empty:
        return None
    for col in ("rt_economic_cashflow_total", "economic_cashflow_total"):
        if col in df.columns:
            series = df[col].astype(float)
            return float(series.iloc[-1])
    if "rt_economic_cashflow_delta" in df.columns:
        return float(df["rt_economic_cashflow_delta"].astype(float).sum())
    return None


def _action_stats(df) -> dict[str, Any]:
    if df is None or df.empty:
        return {}
    stats: dict[str, Any] = {}
    if "decoded_u_tp" in df.columns:
        tp = df["decoded_u_tp"].astype(float)
        stats["u_tp_mean"] = float(tp.mean())
        stats["u_tp_min"] = float(tp.min())
        stats["u_tp_max"] = float(tp.max())
    if "decoded_u_battery" in df.columns:
        bat = df["decoded_u_battery"].astype(float)
        stats["battery_charge_hours"] = int((bat > 1e-6).sum())
        stats["battery_discharge_hours"] = int((bat < -1e-6).sum())
        stats["battery_idle_hours"] = int((np.abs(bat) <= 1e-6).sum())
    if "decoded_u_caes" in df.columns:
        c = df["decoded_u_caes"].astype(float)
        stats["caes_charge_hours"] = int((c > 1e-6).sum())
        stats["caes_discharge_hours"] = int((c < -1e-6).sum())
        stats["caes_idle_hours"] = int((np.abs(c) <= 1e-6).sum())
    elif "requested_caes_mode" in df.columns:
        m = df["requested_caes_mode"].astype(int)
        stats["caes_discharge_hours"] = int((m == 0).sum())
        stats["caes_idle_hours"] = int((m == 1).sum())
        stats["caes_charge_hours"] = int((m == 2).sum())
    return stats


def _components(block: dict[str, Any] | None) -> dict[str, float]:
    if not block:
        return {}
    comps = block.get("economic_cashflow_components") or {}
    if comps:
        return {k: float(v) for k, v in comps.items()}
    terms = block.get("cost_terms") or {}
    out = {}
    for name in ("wind", "pv", "thermal", "battery", "caes", "load", "grid"):
        key = f"economic_cashflow_{name}_delta"
        if key in terms:
            out[name] = float(terms[key])
    return out


def _plot_actions(eval_df, rule_df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib_font()
    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)
    series = [
        ("decoded_u_tp", "火电 u_tp"),
        ("decoded_u_battery", "电池 u_battery"),
        ("decoded_u_caes", "CAES u_caes"),
    ]
    for ax, (col, title) in zip(axes, series):
        if eval_df is not None and col in eval_df.columns:
            ax.plot(eval_df["step"], eval_df[col].astype(float), label="策略", color="C0")
        if rule_df is not None and col in rule_df.columns:
            ax.plot(
                rule_df["step"],
                rule_df[col].astype(float),
                label="规则",
                color="C1",
                ls="--",
                alpha=0.8,
            )
        ax.set_ylabel(title)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc="upper right")
    axes[-1].set_xlabel("step (h)")
    fig.suptitle("调度指令时序")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_cashflow(eval_df, rule_df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib_font()
    fig, ax = plt.subplots(figsize=(11, 4))
    col = "rt_economic_cashflow_total"
    if eval_df is not None and col in eval_df.columns:
        ax.plot(eval_df["step"], eval_df[col].astype(float), label="策略", color="C0")
    if rule_df is not None and col in rule_df.columns:
        ax.plot(
            rule_df["step"],
            rule_df[col].astype(float),
            label="规则",
            color="C1",
            ls="--",
        )
    ax.set_xlabel("step (h)")
    ax.set_ylabel("累计现金流 (元)")
    ax.set_title("累计经济现金流")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _plot_soc(eval_df, out_path: Path) -> None:
    import matplotlib.pyplot as plt

    _setup_matplotlib_font()
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True)
    series = [
        ("obs_battery_soc", "电池 SOC"),
        ("obs_caes_gas_soc", "CAES 气罐"),
        ("obs_caes_hot_soc", "CAES 热罐"),
        ("obs_caes_cold_soc", "CAES 冷罐"),
    ]
    for ax, (col, title) in zip(axes.ravel(), series):
        if eval_df is not None and col in eval_df.columns:
            y = eval_df[col].astype(float)
            ax.plot(eval_df["step"], y, color="C0")
            ax.axhline(y.iloc[0], color="C1", ls="--", alpha=0.6, label="初值")
            ax.axhline(y.iloc[-1], color="C2", ls=":", alpha=0.6, label="终值")
            ax.legend(fontsize=7)
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
    axes[1, 0].set_xlabel("step (h)")
    axes[1, 1].set_xlabel("step (h)")
    fig.suptitle("储能 SOC")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


def _fmt_money(x: float | None) -> str:
    if x is None:
        return "N/A"
    return f"{x:,.2f}"


def generate_policy_report(run_dir: str | Path) -> Path:
    """生成 ``run_dir/report/report.md`` 与三张 PNG，返回报告路径。"""
    run_dir = Path(run_dir)
    summary = _load_summary(run_dir)
    eval_df = _load_csv(run_dir / "trajectories" / "eval.csv")
    rule_df = _load_csv(run_dir / "trajectories" / "rule.csv")
    eval_block = summary.get("eval") or {}
    rule_block = summary.get("rule") or {}

    profit_eval = _cashflow_from_block(eval_block)
    if profit_eval is None:
        profit_eval = _cashflow_from_csv(eval_df)
    profit_rule = _cashflow_from_block(rule_block)
    if profit_rule is None:
        profit_rule = _cashflow_from_csv(rule_df)
    delta = None
    if profit_eval is not None and profit_rule is not None:
        delta = profit_eval - profit_rule

    report_dir = run_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    actions_png = report_dir / "actions.png"
    cashflow_png = report_dir / "cashflow.png"
    soc_png = report_dir / "soc.png"
    _plot_actions(eval_df, rule_df, actions_png)
    _plot_cashflow(eval_df, rule_df, cashflow_png)
    _plot_soc(eval_df, soc_png)

    act = _action_stats(eval_df)
    comps_e = _components(eval_block)
    comps_r = _components(rule_block)

    algo = summary.get("algo") or run_dir.name
    lines = [
        f"# 策略评估报告 · `{run_dir.name}`",
        "",
        f"- 算法 / 运行：`{algo}`",
        f"- 状态：`{summary.get('status', 'unknown')}`",
        f"- 有效训练步：`{summary.get('valid_steps', summary.get('requested_valid_steps', 'N/A'))}`",
        "",
        "## 1. 收益摘要（元）",
        "",
        "| 对象 | 累计现金流 (元) | episode 奖励 |",
        "|------|-----------------|-------------|",
        f"| 训练策略 | {_fmt_money(profit_eval)} | {eval_block.get('episode_reward', 'N/A')} |",
        f"| 规则基线 | {_fmt_money(profit_rule)} | {rule_block.get('episode_reward', 'N/A')} |",
        f"| 策略 − 规则 | {_fmt_money(delta)} | — |",
        "",
        "> 现金流取自 FMU `economic_cashflow_total`（评估窗口末值或窗口增量合计）。正值表示净收益。",
        "",
        "## 2. 现金流分项",
        "",
    ]
    if comps_e or comps_r:
        keys = sorted(set(comps_e) | set(comps_r))
        lines += ["| 分项 | 策略 (元) | 规则 (元) |", "|------|-----------|-----------|"]
        for k in keys:
            lines.append(
                f"| {k} | {_fmt_money(comps_e.get(k))} | {_fmt_money(comps_r.get(k))} |"
            )
        lines.append("")
    else:
        lines += ["（summary 中无分项字段；详见轨迹 CSV 的 `rt_economic_cashflow_*` 列。）", ""]

    def _metric(block: dict, key: str):
        if key in block:
            return block.get(key)
        return (block.get("metrics") or {}).get(key, "N/A")

    lines += [
        "## 3. 运行质量",
        "",
        f"- 弃电能量：策略 `{_metric(eval_block, 'curtailment_energy_mwh')}` MWh，"
        f"规则 `{_metric(rule_block, 'curtailment_energy_mwh')}` MWh",
        f"- 缺供能量：策略 `{_metric(eval_block, 'unserved_energy_mwh')}` MWh，"
        f"规则 `{_metric(rule_block, 'unserved_energy_mwh')}` MWh",
        f"- 非法/失败步（策略）：forbidden=`{eval_block.get('forbidden_action_count', 'N/A')}`，"
        f"invalid=`{eval_block.get('invalid_transition_count', 'N/A')}`",
        f"- GiveSafe 提案拒绝率：`{summary.get('proposal_rejection_rate', 'N/A')}`",
        "",
        "## 4. 策略行为摘要（评估窗口）",
        "",
    ]
    if act:
        lines += [
            f"- 火电负荷率：均值 `{act.get('u_tp_mean', float('nan')):.3f}`，"
            f"范围 `[{act.get('u_tp_min', float('nan')):.3f}, {act.get('u_tp_max', float('nan')):.3f}]`",
            f"- 电池：充电 `{act.get('battery_charge_hours', 0)}` h，"
            f"放电 `{act.get('battery_discharge_hours', 0)}` h，"
            f"待机 `{act.get('battery_idle_hours', 0)}` h",
            f"- CAES：充电 `{act.get('caes_charge_hours', 0)}` h，"
            f"放电 `{act.get('caes_discharge_hours', 0)}` h，"
            f"待机 `{act.get('caes_idle_hours', 0)}` h",
            "",
        ]
    else:
        lines += ["（无评估轨迹，跳过动作统计。）", ""]

    lines += [
        "## 5. 图示",
        "",
        "### 调度指令",
        "",
        "![actions](actions.png)",
        "",
        "### 累计现金流",
        "",
        "![cashflow](cashflow.png)",
        "",
        "### 储能 SOC",
        "",
        "![soc](soc.png)",
        "",
        "## 6. 原始产物",
        "",
        f"- [`summary.json`](../summary.json)",
        f"- [`trajectories/eval.csv`](../trajectories/eval.csv)",
        f"- [`trajectories/rule.csv`](../trajectories/rule.csv)",
        "",
    ]

    report_path = report_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path
