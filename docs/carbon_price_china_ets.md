# China National ETS carbon price & factors

Used by `src/config/reward_config.yaml` → `carbon.*`.  
Full ledger: [`docs/parameter_evidence.md`](parameter_evidence.md).  
**Active profile:** `official-2024-ets-sd-grid-v1`.

## Operating point (main tables)

| Item | Value | Evidence |
|------|--------|----------|
| **Default π_CO2** | **97.49 CNY / tCO2e** | MEE: 2024 year-end composite closing price |
| 2024 band | 69–106 CNY/t | MEE National Carbon Market Development Report |
| 2024 peak | 105.65 CNY/t | same |
| Legacy OP (archived runs) | 80 CNY/t | between ICAP 2025 avg (~70.78) and YE close |
| GHTD3 lit cross-check | ≈86.4 CNY/t | \(0.012\) USD/kg × 7.2 |

Sensitivity set: `{69.30, 80, 86.4, 97.49, 105.65}`.

## Intensity bookkeeping

| Stream | Symbol | Default | Role |
|--------|--------|---------|------|
| Thermal direct EF | \(\eta_{\mathrm{th}}\) | **0.85** t/MWh | **Scenario** plant emission intensity (not the allowance benchmark) |
| Allowance benchmark | \(\beta\) | **0.8049** t/MWh | Official 2024 Class-II (≤300 MW-class) coal **generation** benchmark |
| Grid import EF | \(\eta_g\) | **0.6191** t/MWh | Official **2023 Shandong** location-based electricity CO₂ factor |

Quota: \(A+=\beta E_{\mathrm{th}}\), \(E+=\eta_{\mathrm{th}}E_{\mathrm{th}}\); settle \(-\pi Q\) at episode end.  
Grid import: step tax \(\pi\eta_g E_{\mathrm{buy}}\) (`grid_in_quota: false`).

Do **not** equate \(\eta_{\mathrm{th}}\) with \(\beta\).

## Legacy archived configs

Runs under `runs/**/config/reward_config.yaml` that still show `0.5703` / `0.82` / `80` are tagged  
`legacy-2022-grid-factor/proxy-benchmark`. Do not rewrite those snapshots. New main tables require re-eval (and retrain if carbon observations change policy) under `official-2024-ets-sd-grid-v1`.

## Primary official URLs

- https://www.mee.gov.cn/ywgz/ydqhbh/syqhbh/202501/t20250105_1099975.shtml  
- https://www.mee.gov.cn/ywgz/ydqhbh/wsqtkz/202509/W020250927515316322073.pdf  
- https://www.mee.gov.cn/xxgk2018/xxgk/xxgk03/202410/W020241021392230468687.pdf  
- https://www.mee.gov.cn/xxgk2018/xxgk/xxgk01/202512/W020251231726284332528.pdf  

## Disable

```yaml
carbon:
  enabled: false
```
