# China National ETS carbon price (Scheme B)

Used by `src/config/reward_config.yaml` → `carbon.price_cny_per_t`.

## Operating point

| Item | Value |
|------|--------|
| **Default π_CO2** | **80 CNY / tCO2e** |
| Market | China National ETS, CEA secondary market |
| Rationale | Between 2025 ICAP average (~70.78) and 2024 year-end close (~97–98) |

## Reference levels (for citation)

- ICAP China National ETS factsheet: 2025 average secondary-market price **~70.78 CNY/tCO2e**.
- National carbon market progress / year-end reporting: 2024 composite closing band about **69–106 CNY/t**, year-end close near **97–98 CNY/t**.

## Emission factors (Python proxy; no Modelica change)

| Stream | Default η | Unit |
|--------|-----------|------|
| Thermal generation | 0.85 | tCO2 / MWh |
| Grid import | 0.5703 | tCO2 / MWh |

Mass: Δm = η_th·E_th + η_grid·E_buy (import only).  
Cost: π·Δm deducted from cash-flow increment (external ETS-style cost; fuel cash already in FMU thermal cashflow).

## Disable

```yaml
carbon:
  enabled: false
```
