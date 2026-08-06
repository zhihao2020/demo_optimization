# GiveSafe / feasibility offline stats (from dispatch CSVs)

Projection proxy: continuous command change between `requested_*` and `decoded_*` (shield/decoder).

| Method | Proj. rate | Action L2 mean | Invalid trans. rate | FMU fail / week (mean) | CAES lock frac |
|--------|------------|----------------|---------------------|------------------------|----------------|
| ghtd3 | 0.143 | 0.1291 | 0.0000 | 0.00 | 0.083 |
| td3 | 0.658 | 0.7107 | 0.0065 | 0.33 | 0.404 |
| b0 | 0.046 | 0.0305 | 0.0000 | 0.00 | 0.028 |
| pso | 0.222 | 0.0680 | 0.0499 | 0.00 | 0.067 |
| linprog | 0.510 | 0.3998 | 0.0000 | 0.00 | 0.353 |
