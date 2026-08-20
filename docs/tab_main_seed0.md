# seasonal_v1 seed 0 — GHTD3-style cost tables (hybrid SAC identity)

Source: `runs/seasonal_v1/**/train_result.json` and `docs/matrix_status_hybrid_sac.md`.
Only `valid_steps=168` rows enter `tab:main`. Truncated weeks are not ranked on cash.
**Main method: hybrid SAC (`sac_param`).** No HMSD. `milp` = binary CAES + min-load on energy surrogate.

Cost signs: **lower is better** for CC / C_ET / C_ops / C_CUT / C_CO2 / C_DEG (CC = −J^gen). Jgen kept for cross-check (higher better).

## tab:main (full 168 h; cost breakdown)

| season | method | CC (CNY) | C_ET | C_ops | C_CUT | C_CO2 | C_DEG | Jgen |
|--------|--------|----------|------|-------|-------|-------|-------|------|
| winter | pso | -1.436e+07 | — | — | 0.0 | — | — | 1.436e+07 |
| winter | linprog | -1.019e+07 | -1.964e+06 | 2.967e+05 | 0.0 | 88406.5 | 729.2 | 1.019e+07 |
| transition | linprog | -2.660e+06 | 1.604e+06 | 1.404e+06 | 0.0 | 414812.7 | 506.3 | 2.660e+06 |
| summer | linprog | -3.256e+06 | 1.758e+06 | 1.819e+06 | 0.0 | 289936.8 | 417.2 | 3.256e+06 |

## tab:run (executability; auxiliary)

| season | method | status | hours | full_week | note |
|--------|--------|--------|-------|-----------|------|
| winter | hybrid SAC | missing | — | N |  |
| winter | hybrid TD3 | missing | — | N |  |
| winter | proj. SAC | eval_failed | — | N | NoSafeActionFoundError |
| winter | proj. TD3 | completed | 13 | N | incomplete 13/168 h |
| winter | pso | completed | 168 | Y |  |
| winter | linprog | completed | 168 | Y |  |
| winter | milp | missing | — | N |  |
| transition | hybrid SAC | missing | — | N |  |
| transition | hybrid TD3 | missing | — | N |  |
| transition | proj. SAC | eval_failed | — | N | NoSafeActionFoundError |
| transition | proj. TD3 | completed | 5 | N | incomplete 5/168 h |
| transition | pso | completed | 143 | N | incomplete 143/168 h |
| transition | linprog | completed | 168 | Y |  |
| transition | milp | missing | — | N |  |
| summer | hybrid SAC | missing | — | N |  |
| summer | hybrid TD3 | missing | — | N |  |
| summer | proj. SAC | missing | — | N |  |
| summer | proj. TD3 | completed | 76 | N | incomplete 76/168 h |
| summer | pso | completed | 117 | N | incomplete 117/168 h |
| summer | linprog | completed | 168 | Y |  |
| summer | milp | missing | — | N |  |

## Forbidden claims

- Do not rank truncated weeks against full weeks on cash.
- Do not claim 8760 h RL safe + best economics.
- Do not mix obs=163 archives with this matrix.
- Do not report HMSD in the paper (body or appendix).
- Do not claim unconditional RL > MILP; if MILP CC is lower, explain energy linearization / missing thermal coupling / twin non-executability.
