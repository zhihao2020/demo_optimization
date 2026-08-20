<!-- ARCHIVED: superseded by FS-HSAC; see docs/paper_outline_and_figures.md -->

# Fair seasonal results aggregate

Source root: `runs\seasonal_v1`
n_results: 14

## Per seed

| season | method | seed | R | Jgen | CF | unserved | reject_rate |
|--------|--------|------|---|------|----|----------|-------------|
| summer | hmsd | 0 | 68.574 | 1.173e+07 | 1.260e+07 | 0.000 | 0.332 |
| summer | linprog | 0 | 11.591 | 3.256e+06 | 5.074e+06 | 0.000 | — |
| summer | pso | 0 | 55.890 | 9.386e+06 | 9.438e+06 | 0.000 | — |
| summer | td3 | 0 | 11.360 | 2.695e+06 | 2.824e+06 | 0.000 | — |
| transition | hmsd | 0 | 43.432 | 8.320e+06 | 1.092e+07 | 0.000 | 0.306 |
| transition | linprog | 0 | 13.198 | 2.660e+06 | 4.064e+06 | 0.000 | — |
| transition | pso | 0 | 52.657 | 8.538e+06 | 9.379e+06 | 0.000 | — |
| transition | sac | 0 | — | — | — | — | — |
| transition | td3 | 0 | 1.086 | 5.102e+05 | 7.557e+05 | 0.000 | — |
| winter | hmsd | 0 | 80.547 | 1.334e+07 | 1.426e+07 | 0.000 | 0.332 |
| winter | linprog | 0 | 59.992 | 1.019e+07 | 1.048e+07 | 0.000 | — |
| winter | pso | 0 | 102.720 | 1.436e+07 | 1.480e+07 | 0.000 | — |
| winter | sac | 0 | — | — | — | — | — |
| winter | td3 | 0 | 1.422 | 6.820e+05 | 7.058e+05 | 0.000 | — |

## Mean ± std (by season × method)

| season | method | n | R mean±std | Jgen mean±std |
|--------|--------|---|------------|---------------|
| summer | hmsd | 1 | 68.57±0.00 | 1.173e+07±0.000e+00 |
| summer | linprog | 1 | 11.59±0.00 | 3.256e+06±0.000e+00 |
| summer | pso | 1 | 55.89±0.00 | 9.386e+06±0.000e+00 |
| summer | td3 | 1 | 11.36±0.00 | 2.695e+06±0.000e+00 |
| transition | hmsd | 1 | 43.43±0.00 | 8.320e+06±0.000e+00 |
| transition | linprog | 1 | 13.20±0.00 | 2.660e+06±0.000e+00 |
| transition | pso | 1 | 52.66±0.00 | 8.538e+06±0.000e+00 |
| transition | sac | 1 | — | — |
| transition | td3 | 1 | 1.09±0.00 | 5.102e+05±0.000e+00 |
| winter | hmsd | 1 | 80.55±0.00 | 1.334e+07±0.000e+00 |
| winter | linprog | 1 | 59.99±0.00 | 1.019e+07±0.000e+00 |
| winter | pso | 1 | 102.72±0.00 | 1.436e+07±0.000e+00 |
| winter | sac | 1 | — | — |
| winter | td3 | 1 | 1.42±0.00 | 6.820e+05±0.000e+00 |

## HMSD vs TD3 (mean R)

- winter: HMSD=80.55, TD3=1.42 → **HMSD higher**
- transition: HMSD=43.43, TD3=1.09 → **HMSD higher**
- summer: HMSD=68.57, TD3=11.36 → **HMSD higher**
