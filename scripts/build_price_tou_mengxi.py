#!/usr/bin/env python
"""生成蒙西电网工商业分时购售电价 CSV（默认写入 data/price_tou.csv）。

输配电度价：第四监管周期 · 蒙西 110kV（介子九维 transmission-tariffs / NDRC 表）。
分时时段与比价：内蒙古完善蒙西工商业分时政策（大风季/小风季 + 6–8 月尖峰深谷）。
代理购电浮动基数 B 与基金附加为算例参数（illustrative），非当月代理购电真值。
"""
from __future__ import annotations

import csv
import json
from calendar import monthrange
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

DOC = {
    "province": "mengxi",
    "province_name": "蒙西（内蒙古电力）",
    "year": 2026,
    "doc_no": (
        "内蒙古自治区发展和改革委员会关于完善蒙西电网工商业分时电价政策"
        "（2024-02-01 起口径；大/小风季时段与比价）；"
        "输配电：国家发展改革委第四监管周期省级电网输配电价表 · 蒙西"
    ),
    "effective_from": "2024-02-01",
    "path": "proxy_purchase_tou",
    "review_status": "tou-from-nmg-policy-public-summary; td-from-ndrc-fourth-cycle-via-jiezijiuwei",
    "source_url": "https://www.jiezijiuwei.com/tools/transmission-tariffs",
    "transmission_source": (
        "https://www.jiezijiuwei.com/tools/transmission-tariffs (mengxi) · "
        "NDRC fourth-cycle provincial TD schedule · 110kV td_energy=0.052"
    ),
    "tou_policy_note": (
        "大风季 1–5、9–12：峰 06–08/18–22 +68%，谷 00–04/11–16 −52%；"
        "小风季 6–8：峰同 +54%，谷 11–16 −56%；"
        "6–8 月尖峰 19–21=峰×1.2，深谷 13–15=谷×0.8"
    ),
}

# 浮动基数（参与分时浮动）— 算例参数
proxy_price = 0.4
capacity_comp_price = 0.0705
line_loss_price = 0.01
sys_op_price = 0.0209
float_base = proxy_price + capacity_comp_price + line_loss_price + sys_op_price

# 不参与浮动
td_energy_price = 0.052  # 蒙西 110kV 输配电度价
fund_price = 0.02717
non_float = td_energy_price + fund_price

# 相对平段的比价；S/D 仅小风季在峰/谷上再乘
ratios_windy = {"P": 1.68, "F": 1.0, "V": 0.48}  # 大风季
ratios_calm = {"P": 1.54, "F": 1.0, "V": 0.44}  # 小风季
# 小风季尖峰/深谷相对平段
ratio_S_calm = ratios_calm["P"] * 1.2  # 1.848
ratio_D_calm = ratios_calm["V"] * 0.8  # 0.352

# 售电（平坦算例）
export_mech_price = 0.225
export_market_price = 0.15
export_mech_ratio = 0.5
sell_flat = round(
    export_mech_ratio * export_mech_price + (1 - export_mech_ratio) * export_market_price, 6
)

# 24h 模板：hour index 0=00:00–01:00
# 大风季
# 峰 6–8, 18–22；谷 0–4, 11–16；其余平
WINDY_HOURS = []
for h in range(24):
    if h in (6, 7) or h in (18, 19, 20, 21):
        WINDY_HOURS.append("P")
    elif h in (0, 1, 2, 3) or h in (11, 12, 13, 14, 15):
        WINDY_HOURS.append("V")
    else:
        WINDY_HOURS.append("F")
assert len(WINDY_HOURS) == 24
assert WINDY_HOURS.count("P") == 6 and WINDY_HOURS.count("V") == 9 and WINDY_HOURS.count("F") == 9

# 小风季底稿：峰同；谷仅 11–16；其余平（含 0–6 为平）
CALM_BASE = []
for h in range(24):
    if h in (6, 7) or h in (18, 19, 20, 21):
        CALM_BASE.append("P")
    elif h in (11, 12, 13, 14, 15):
        CALM_BASE.append("V")
    else:
        CALM_BASE.append("F")
assert CALM_BASE.count("P") == 6 and CALM_BASE.count("V") == 5 and CALM_BASE.count("F") == 13

# 小风季叠加尖峰 19–21、深谷 13–15
CALM_HOURS = list(CALM_BASE)
for h in (19, 20):
    CALM_HOURS[h] = "S"
for h in (13, 14):
    CALM_HOURS[h] = "D"


def ratio_for(band: str, *, calm: bool) -> float:
    if calm:
        if band == "S":
            return ratio_S_calm
        if band == "D":
            return ratio_D_calm
        return float(ratios_calm[band])
    if band in ("S", "D"):
        raise ValueError("大风季无尖峰/深谷")
    return float(ratios_windy[band])


def buy_price(band: str, *, calm: bool) -> float:
    return round(float_base * ratio_for(band, calm=calm) + non_float, 6)


def main() -> None:
    year = 2026
    assert sum(monthrange(year, m)[1] for m in range(1, 13)) == 365

    band_buy_windy = {b: buy_price(b, calm=False) for b in ("P", "F", "V")}
    band_buy_calm = {
        b: buy_price(b, calm=True) for b in ("S", "P", "F", "V", "D")
    }

    rows: list[dict[str, str | float | int]] = []
    counts: dict[str, int] = {"S": 0, "P": 0, "F": 0, "V": 0, "D": 0}
    h = 0
    for month in range(1, 13):
        calm = month in (6, 7, 8)
        bands = CALM_HOURS if calm else WINDY_HOURS
        days = monthrange(year, month)[1]
        for _day in range(days):
            for hour in range(24):
                band = bands[hour]
                buy = buy_price(band, calm=calm)
                rows.append(
                    {
                        "time": h * 3600,
                        "buy_yuan_per_kwh": buy,
                        "sell_yuan_per_kwh": sell_flat,
                        "band": band,
                    }
                )
                counts[band] += 1
                h += 1
    assert h == 8760, h

    out = DATA / "price_tou.csv"
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(
            f, fieldnames=["time", "buy_yuan_per_kwh", "sell_yuan_per_kwh", "band"]
        )
        w.writeheader()
        w.writerows(rows)

    meta = {
        **DOC,
        "float_base_components": {
            "proxy_price": proxy_price,
            "capacity_comp_price": capacity_comp_price,
            "line_loss_price": line_loss_price,
            "sys_op_price": sys_op_price,
            "float_base_sum": float_base,
            "note": "illustrative proxy-purchase composition; not monthly Mengxi settlement",
        },
        "non_float_components": {
            "td_energy_price": td_energy_price,
            "fund_price": fund_price,
            "non_float_sum": non_float,
            "td_voltage_note": "蒙西 110kV 输配电度价 0.052 元/kWh（第四监管周期 / jiezijiuwei）",
            "td_demand_yuan_per_kw_month": 31.2,
            "td_capacity_yuan_per_kva_month": 19.5,
            "two_part_not_in_rl": True,
        },
        "ratios_windy_season": ratios_windy,
        "ratios_calm_season": {
            **ratios_calm,
            "S": ratio_S_calm,
            "D": ratio_D_calm,
        },
        "buy_yuan_per_kwh_by_band_windy": band_buy_windy,
        "buy_yuan_per_kwh_by_band_calm": band_buy_calm,
        "buy_formula": "buy = B * r_band(season) + td_energy + fund",
        "sell_yuan_per_kwh": sell_flat,
        "sell_formula": "0.5*export_mech_price + 0.5*export_market_price (illustrative flat)",
        "export_components": {
            "export_mech_ratio": export_mech_ratio,
            "export_mech_price": export_mech_price,
            "export_market_price": export_market_price,
        },
        "band_hours_in_8760": counts,
        "calendar_year": year,
        "n_hours": 8760,
        "templates": {
            "windy_hours": WINDY_HOURS,
            "calm_hours": CALM_HOURS,
        },
        "note": (
            "Price-taker volumetric TOU for Mengxi grid. "
            "Not ISO clearing. Capacity/demand two-part charges listed in meta but not in RL reward. "
            "Replaces Shandong default; old table: data/price_tou_shandong_backup.csv"
        ),
    }
    (DATA / "price_tou_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    # sanity
    jan = [r for r in rows if int(r["time"]) < 24 * 3600]
    jun_start = sum(monthrange(year, m)[1] for m in range(1, 6)) * 24
    jun = rows[jun_start : jun_start + 24]
    assert all(r["band"] != "S" for r in jan)
    assert any(r["band"] == "S" for r in jun)
    assert any(r["band"] == "D" for r in jun)
    avg_buy = sum(float(r["buy_yuan_per_kwh"]) for r in rows) / len(rows)
    print("wrote", out)
    print("bands hours", counts)
    print("buy windy", band_buy_windy)
    print("buy calm", band_buy_calm)
    print("sell", sell_flat)
    print("avg buy", round(avg_buy, 6))
    print("Jan bands", [r["band"] for r in jan])
    print("Jun bands", [r["band"] for r in jun])


if __name__ == "__main__":
    main()
