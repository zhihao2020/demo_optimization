"""Build data/price_tou.csv.

Default: monthly 110 kV two-part agency-purchase TOU from
data/price_tou_monthly_official.csv (year=2026). Missing months are
not filled with the old year-constant 0.4 proxy.

--legacy: previous constructive five-level path (archive only).
"""
from __future__ import annotations

import argparse
import csv
import json
from calendar import monthrange
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

RATIOS = {"S": 2.0, "P": 1.7, "F": 1.0, "V": 0.3, "D": 0.1}
MONTHLY_HOURS = {
    1: ["F", "F", "V", "V", "V", "V", "F", "P", "P", "F", "V", "D", "D", "D", "V", "F", "S", "S", "S", "P", "P", "F", "F", "F"],
    2: ["F", "F", "V", "V", "V", "V", "F", "P", "P", "F", "V", "D", "D", "D", "V", "F", "S", "S", "S", "P", "P", "F", "F", "F"],
    3: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "F", "S", "S", "S", "P", "P", "F", "F"],
    4: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "F", "S", "S", "S", "P", "P", "F", "F"],
    5: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "F", "S", "S", "S", "P", "P", "F", "F"],
    6: ["F", "F", "F", "F", "F", "F", "F", "V", "V", "V", "V", "V", "F", "F", "F", "F", "P", "S", "S", "S", "S", "S", "P", "F"],
    7: ["F", "V", "V", "V", "V", "V", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "P", "S", "S", "S", "S", "S", "P", "F"],
    8: ["F", "V", "V", "V", "V", "V", "F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "P", "S", "S", "S", "S", "S", "P", "F"],
    9: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "P", "S", "S", "P", "P", "F", "F", "F"],
    10: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "P", "S", "S", "P", "P", "F", "F", "F"],
    11: ["F", "F", "F", "F", "F", "F", "F", "F", "F", "F", "V", "D", "D", "D", "V", "F", "P", "S", "S", "P", "P", "F", "F", "F"],
    12: ["F", "F", "V", "V", "V", "V", "F", "P", "P", "F", "V", "D", "D", "D", "V", "F", "S", "S", "S", "P", "P", "F", "F", "F"],
}
assert all(len(v) == 24 for v in MONTHLY_HOURS.values())

SELL_FLAT = 0.1875


def _write_8760(year: int, month_buy: dict[int, dict[str, float]], meta: dict) -> None:
    days_ok = sum(monthrange(year, m)[1] for m in range(1, 13))
    assert days_ok in (365, 366)
    rows = ["time,buy_yuan_per_kwh,sell_yuan_per_kwh,band"]
    counts = {k: 0 for k in RATIOS}
    h = 0
    for month in range(1, 13):
        bands_h = MONTHLY_HOURS[month]
        buy_m = month_buy[month]
        for _day in range(monthrange(year, month)[1]):
            for hour in range(24):
                band = bands_h[hour]
                buy = buy_m[band]
                rows.append(f"{h * 3600},{buy},{SELL_FLAT},{band}")
                counts[band] += 1
                h += 1
    n_hours = days_ok * 24
    assert h == n_hours, h
    out = DATA / "price_tou.csv"
    out.write_text("\n".join(rows) + "\n", encoding="utf-8")
    meta = {
        **meta,
        "sell_yuan_per_kwh": SELL_FLAT,
        "band_hours": counts,
        "calendar_year": year,
        "n_hours": n_hours,
        "buy_by_month": {str(m): month_buy[m] for m in range(1, 13)},
    }
    (DATA / "price_tou_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print("wrote", out)
    print("bands hours", counts)


def build_monthly(year: int = 2026) -> None:
    path = DATA / "price_tou_monthly_official.csv"
    with path.open(encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    by_month: dict[int, dict] = {}
    for row in table:
        if int(row["year"]) != year:
            continue
        m = int(row["month"])
        by_month[m] = row
    missing_rows = [m for m in range(1, 13) if m not in by_month]
    empty_bands = [
        m
        for m, r in by_month.items()
        if not str(r.get("band_S") or "").strip()
    ]
    if missing_rows or empty_bands:
        raise SystemExit(
            "monthly official 2026 TOU is incomplete; will not overwrite price_tou.csv "
            f"with the legacy 0.4 proxy. missing_rows={missing_rows} empty_bands={empty_bands}. "
            "Fill data/price_tou_monthly_official.csv from official PDFs, or pass --legacy."
        )
    last_o: dict[str, float] | None = None
    month_buy: dict[int, dict[str, float]] = {}
    grades = {}
    for m in range(1, 13):
        r = by_month[m]
        bands = {
            "S": round(float(r["band_S"]), 8),
            "P": round(float(r["band_P"]), 8),
            "F": round(float(r["band_F"]), 8),
            "V": round(float(r["band_V"]), 8),
            "D": round(float(r["band_D"]), 8),
        }
        grade = (r.get("evidence_grade") or "S").strip()
        if grade == "S" and last_o is not None and bands == last_o:
            pass
        if grade == "O":
            last_o = bands
        month_buy[m] = bands
        grades[m] = grade
    _write_8760(
        year,
        month_buy,
        {
            "province": "shandong",
            "year": year,
            "path": "monthly_agency_purchase_tou",
            "review_status": "official monthly 110kV two-part delivered TOU + official clocks",
            "claim_level": "Buy = monthly agency-purchase tables; sell remains S-grade flat",
            "evidence_grade_by_month": {str(k): v for k, v in grades.items()},
            "ledger": str(path.relative_to(ROOT)).replace("\\", "/"),
            "windows": "国网山东 2026 工商业分时公告; 鲁发改价格〔2023〕914号",
        },
    )


def build_legacy() -> None:
    """Archive: year-constant constructive five-level path (not the paper default)."""
    proxy_price = 0.4
    capacity_comp_price = 0.0705
    line_loss_price = 0.01
    sys_op_price = 0.0209
    float_base = proxy_price + capacity_comp_price + line_loss_price + sys_op_price
    td_energy_price = 0.106
    fund_price = 0.02717
    non_float = td_energy_price + fund_price
    band_buy = {k: round(float_base * r + non_float, 6) for k, r in RATIOS.items()}
    month_buy = {m: band_buy for m in range(1, 13)}
    _write_8760(
        2026,
        month_buy,
        {
            "province": "shandong",
            "year": 2026,
            "path": "legacy_constructive_proxy_purchase_tou",
            "review_status": "LEGACY year-constant constructive levels; do not cite as official 2026",
            "claim_level": "Do not cite absolute buy levels as official 2026 settlement",
            "float_base": float_base,
            "non_float": non_float,
            "buy_yuan_per_kwh_by_band": band_buy,
        },
    )


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--legacy", action="store_true", help="year-constant constructive path (archive)")
    p.add_argument("--year", type=int, default=2026)
    args = p.parse_args()
    if args.legacy:
        build_legacy()
    else:
        build_monthly(args.year)


if __name__ == "__main__":
    main()
