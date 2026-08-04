import json
from pathlib import Path
runs = Path(r"D:\Code\0622\optimal_demo\runs")
rows = []
for d in sorted(runs.iterdir()):
    if not d.is_dir():
        continue
    for name in ["summary.json", "pipeline_summary.json"]:
        p = d/name
        if not p.exists():
            continue
        try:
            s = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        ev = s.get("eval") or s.get("final_eval") or {}
        rew = ev.get("episode_reward") or ev.get("weekly_episode_reward")
        cost = ev.get("weekly_raw_total_cost")
        rule = s.get("rule") or s.get("baseline") or {}
        if isinstance(rule, dict):
            rule_rew = rule.get("episode_reward") or rule.get("weekly_episode_reward")
        else:
            rule_rew = None
        rows.append((d.name, rew, cost, rule_rew, s.get("algorithm") or s.get("status"), ev.get("terminal_soc_satisfied")))
        break
print(f"{'run':45} {'eval_rew':>10} {'rule_rew':>10} {'termSOC':>8} algo")
for r in rows:
    print(f"{r[0][:45]:45} {str(r[1])[:10]:>10} {str(r[3])[:10]:>10} {str(r[5]):>8} {r[4]}")
