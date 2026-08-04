from pathlib import Path
import json
from calendar import monthrange

ROOT = Path(r"D:\Code\0622\optimal_demo")
DATA = ROOT / "data"

# 国网山东 2026 工商业分时 + 介子九维引擎 REFERENCE_INPUTS（山东代理购电·园区微电网算例）
# 来源：jiezijiuwei smart-microgrid priceData.tou.provinces[shandong] + REFERENCE_INPUTS
DOC = {
    "province": "shandong",
    "province_name": "山东",
    "year": 2026,
    "doc_no": "国网山东省电力公司2026年工商业分时电价公告；框架依据鲁发改价格〔2023〕914号",
    "effective_from": "2026-01-01",
    "path": "proxy_purchase_tou",  # 电网代理购电分时，非 ISO 出清
    "review_status": "provincial-published (via jiezijiuwei compiled table)",
    "source_url": "https://www.jiezijiuwei.com/tools/smart-microgrid",
    "transmission_source": "国家发展改革委第四监管周期省级电网输配电价表（山东 110kV 档示例 td_energy=0.106）",
}

# 浮动基数（参与分时浮动）
proxy_price = 0.4
capacity_comp_price = 0.0705
line_loss_price = 0.01
sys_op_price = 0.0209
float_base = proxy_price + capacity_comp_price + line_loss_price + sys_op_price  # 0.5014

# 不参与浮动
td_energy_price = 0.106  # 山东 110kV 输配电度价（算例）
fund_price = 0.02717
non_float = td_energy_price + fund_price  # 0.13317

ratios = {"S": 2.0, "P": 1.7, "F": 1.0, "V": 0.3, "D": 0.1}
band_buy = {k: round(float_base * r + non_float, 6) for k, r in ratios.items()}

# 余电上网：机制电价与市场均价各半（REFERENCE_INPUTS）
export_mech_price = 0.225
export_market_price = 0.15
export_mech_ratio = 0.5
sell_flat = round(export_mech_ratio * export_mech_price + (1 - export_mech_ratio) * export_market_price, 6)

# 分月 24h 时段表（S尖峰 P高峰 F平 V低谷 D深谷）
monthly_hours = [
    {"months": [1, 2, 12], "hours": list("FFVVVVFPPFVDVDV FSSSPPFFF".replace(" ", ""))},
    {"months": [3, 4, 5], "hours": list("FFFFFFF FFVDVDVFFSSSPPFF".replace(" ", ""))},
    {"months": [6], "hours": list("FFFFFFFVVVVVFFFFPSSSSSPF".replace(" ", ""))},
    {"months": [7, 8], "hours": list("FVVVVVFFFFFFFFFFPSSSSSPF".replace(" ", ""))},
    {"months": [9, 10, 11], "hours": list("FFFFFFFFFFVDVDVFPSSPPFFF".replace(" ", ""))},
]
# fix hours strings carefully from original
monthly_hours = [
    {"months": [1, 2, 12], "hours": ["F","F","V","V","V","V","F","P","P","F","V","D","D","D","V","F","S","S","S","P","P","F","F","F"]},
    {"months": [3, 4, 5], "hours": ["F","F","F","F","F","F","F","F","F","F","V","D","D","D","V","F","F","S","S","S","P","P","F","F"]},
    {"months": [6], "hours": ["F","F","F","F","F","F","F","V","V","V","V","V","F","F","F","F","P","S","S","S","S","S","P","F"]},
    {"months": [7, 8], "hours": ["F","V","V","V","V","V","F","F","F","F","F","F","F","F","F","F","P","S","S","S","S","S","P","F"]},
    {"months": [9, 10, 11], "hours": ["F","F","F","F","F","F","F","F","F","F","V","D","D","D","V","F","P","S","S","P","P","F","F","F"]},
]
assert all(len(m["hours"]) == 24 for m in monthly_hours)

month_to_hours = {}
for block in monthly_hours:
    for mo in block["months"]:
        month_to_hours[mo] = block["hours"]

# 2026 非闰年，8760 = 365*24
year = 2026
assert sum(monthrange(year, m)[1] for m in range(1, 13)) == 365

rows = ["time,buy_yuan_per_kwh,sell_yuan_per_kwh,band"]
counts = {k: 0 for k in ratios}
h = 0
for month in range(1, 13):
    days = monthrange(year, month)[1]
    bands = month_to_hours[month]
    for _day in range(days):
        for hour in range(24):
            band = bands[hour]
            buy = band_buy[band]
            sell = sell_flat
            rows.append(f"{h*3600},{buy},{sell},{band}")
            counts[band] += 1
            h += 1
assert h == 8760, h

out = DATA / "price_tou.csv"
out.write_text("\n".join(rows) + "\n", encoding="utf-8")

meta = {
    **DOC,
    "float_base_components": {
        "proxy_price": proxy_price,
        "capacity_comp_price": capacity_comp_price,
        "line_loss_price": line_loss_price,
        "sys_op_price": sys_op_price,
        "float_base_sum": float_base,
    },
    "non_float_components": {
        "td_energy_price": td_energy_price,
        "fund_price": fund_price,
        "non_float_sum": non_float,
        "td_voltage_note": "算例取山东 110kV 输配电度价 0.106 元/kWh（第四监管周期）",
    },
    "ratios": ratios,
    "buy_yuan_per_kwh_by_band": band_buy,
    "buy_formula": "buy = (proxy+capacity_comp+line_loss+sys_op) * ratio[band] + td_energy + fund",
    "sell_yuan_per_kwh": sell_flat,
    "sell_formula": "0.5*export_mech_price + 0.5*export_market_price",
    "export_components": {
        "export_mech_ratio": export_mech_ratio,
        "export_mech_price": export_mech_price,
        "export_market_price": export_market_price,
    },
    "band_hours_in_8760": counts,
    "calendar_year": year,
    "n_hours": 8760,
    "note": "Price-taker 外生价；非现货出清结果。直接参与市场交易用户电能量价可由现货形成，本文件采用代理购电分时路径。",
}
(DATA / "price_tou_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

print("wrote", out)
print("bands hours", counts)
print("buy by band", band_buy)
print("sell", sell_flat)
print("sample Jan hour0-23:", [bands for bands in [month_to_hours[1]]][0])
# sanity: first 24 buys
import csv
with out.open(encoding="utf-8") as f:
    r = list(csv.DictReader(f))
print("row0", r[0])
print("row16 peak S?", r[16])  # Jan hour 16 should be S
print("avg buy", sum(float(x["buy_yuan_per_kwh"]) for x in r)/len(r))
