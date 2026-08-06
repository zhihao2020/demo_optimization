# Surrogate weekly optimality gap (same-proxy)

J_surr_star: optimal linear cash-flow proxy (weekly LP, perfect foresight, continuous thermal/storage/grid, must-take RE, terminal SoC band). gap_surr compares methods on the SAME proxy (trajectory replay). J_fmu is closed-loop twin cash flow and is NOT bounded by J_surr_star.

- Horizon: 168 h | fuel: 400.0 CNY/MWh | terminal ε=0.06

| Season | \(J^*_{surr}\) (10⁶) | B0 gap% | linprog gap% | PSO gap% | TD3 gap% | HMSD gap% | HMSD \(J_{surr}/J^*\) | HMSD \(J_{FMU}\) (10⁶) |
|--------|---------------------:|--------:|-------------:|---------:|---------:|----------:|----------------------:|------------------------:|
| winter | 19.27 | 57.0 | 59.0 | nan | 31.9 | 8.1 | 0.92 | 17.45 |
| transition | 17.91 | 65.5 | 62.5 | 27.9 | nan | 23.3 | 0.77 | 13.23 |
| summer | 15.41 | 91.3 | 92.4 | nan | 53.0 | 19.9 | 0.80 | 11.89 |
